# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    """Return the 6 vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _min_outer_side_for_inner(inner_hex_data, outer_center):
    """
    Compute the minimum side length of a regular hexagon (centered at outer_center, angle 0)
    that contains all inner hexagons (unit side).
    Returns (side, ok) where ok is True if all vertices are inside.
    """
    cx, cy, _ = outer_center
    normals = []
    for k in range(6):
        ang = np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    normals = np.array(normals)  # shape (6, 2)
    
    max_proj = 0.0
    all_finite = True
    for i in range(len(inner_hex_data)):
        x, y, a_deg = inner_hex_data[i]
        verts = _hex_vertices(x, y, 1.0, a_deg)
        if not np.all(np.isfinite(verts)):
            all_finite = False
            break
        shifted = verts - np.array([cx, cy])
        projs = shifted @ normals.T
        m = np.max(projs)
        if m > max_proj:
            max_proj = m
    
    if not all_finite:
        return float('inf'), False
    
    side = max_proj * 2.0 / np.sqrt(3.0)
    return side, True


def _check_validity(inner_hex_data, outer_hex_data, side):
    """
    Check all validity conditions. Returns True if valid.
    """
    if not (np.all(np.isfinite(inner_hex_data)) and np.all(np.isfinite(outer_hex_data))):
        return False
    if not np.isfinite(side):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    
    ox, oy, oa = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, side, oa)
    
    # Check every inner vertex is inside outer hexagon
    for i in range(len(inner_hex_data)):
        x, y, a_deg = inner_hex_data[i]
        verts = _hex_vertices(x, y, 1.0, a_deg)
        for p in verts:
            for j in range(6):
                p1 = outer_verts[j]
                p2 = outer_verts[(j + 1) % 6]
                e = p2 - p1
                v = p - p1
                cross = e[0] * v[1] - e[1] * v[0]
                if cross < -1e-6:
                    return False
    
    # Check every pair of inner hexagons is separated
    n_inner = len(inner_hex_data)
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            x1, y1, a1 = inner_hex_data[i]
            x2, y2, a2 = inner_hex_data[j]
            v1 = _hex_vertices(x1, y1, 1.0, a1)
            v2 = _hex_vertices(x2, y2, 1.0, a2)
            
            def get_normals(verts):
                norms = []
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k + 1) % 6]
                    e = p2 - p1
                    n = np.array([e[1], -e[0]])
                    n = n / np.linalg.norm(n)
                    norms.append(n)
                return np.array(norms)
            
            n1 = get_normals(v1)
            n2 = get_normals(v2)
            all_normals = np.vstack([n1, n2])
            
            separated = False
            for n in all_normals:
                proj1 = v1 @ n
                proj2 = v2 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
    
    return True


def _compute_overlap_penalty(inner_hex_data):
    """
    Compute a penalty for overlapping inner hexagons.
    For each pair, we check the maximum separation on the 12 edge normals.
    If max_separation <= 1e-6, they are considered intersecting or too close.
    Penalty is (1e-6 - max_separation)^2 if max_separation < 1e-6, else 0.
    """
    n_inner = len(inner_hex_data)
    total_penalty = 0.0
    
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            x1, y1, a1 = inner_hex_data[i]
            x2, y2, a2 = inner_hex_data[j]
            v1 = _hex_vertices(x1, y1, 1.0, a1)
            v2 = _hex_vertices(x2, y2, 1.0, a2)
            
            def get_normals(verts):
                norms = []
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k + 1) % 6]
                    e = p2 - p1
                    n = np.array([e[1], -e[0]])
                    n = n / np.linalg.norm(n)
                    norms.append(n)
                return np.array(norms)
            
            n1 = get_normals(v1)
            n2 = get_normals(v2)
            all_normals = np.vstack([n1, n2])
            
            max_sep = -float('inf')
            for n in all_normals:
                proj1 = v1 @ n
                proj2 = v2 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # Separation on this normal
                # If max1 < min2, sep = min2 - max1
                # If max2 < min1, sep = min1 - max2
                # Else, sep = min(max1, max2) - max(min1, min2) (which is negative, representing overlap depth)
                if max1 < min2:
                    sep = min2 - max1
                elif max2 < min1:
                    sep = min1 - max2
                else:
                    sep = min(max1, max2) - max(min1, min2)
                
                if sep > max_sep:
                    max_sep = sep
            
            # We require max_sep > 1e-6
            if max_sep < 1e-6:
                total_penalty += (1e-6 - max_sep) ** 2
    
    return total_penalty


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580.0  # 580s to leave safety margin
    
    sqrt3 = np.sqrt(3)
    
    # Initial seed: A compact hexagonal arrangement.
    # 1 center, 6 around, 4 in second ring.
    # Centers of unit hexagons in a honeycomb grid (flat top, angle 0):
    # Distance between centers is 2 * apothem = sqrt(3) for adjacent? No.
    # For flat-top hexagons (vertices at 0, 60...), the width is 2, height is sqrt(3).
    # Horizontal neighbors: dx=2, dy=0.
    # Diagonal neighbors: dx=1, dy=sqrt(3).
    
    inner_seed = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, sqrt3, 0.0],
            [-1.0, sqrt3, 0.0],
            [-2.0, 0.0, 0.0],
            [-1.0, -sqrt3, 0.0],
            [1.0, -sqrt3, 0.0],
            [3.0, sqrt3, 0.0],
            [0.0, 2*sqrt3, 0.0],
            [-3.0, sqrt3, 0.0],
            [-3.0, -sqrt3, 0.0],
        ]
    )
    
    outer_seed = np.array([0.0, 0.0, 0.0])
    side_seed, ok_seed = _min_outer_side_for_inner(inner_seed, outer_seed)
    
    if not ok_seed or not np.isfinite(side_seed):
        return inner_seed, outer_seed, 8.0
    
    best_inner = inner_seed.copy()
    best_outer = outer_seed.copy()
    best_side = side_seed
    best_score = 1.0 / best_side

    def objective(params):
        """Objective: minimize outer side + penalty for overlaps."""
        inner = params.reshape(11, 3)
        # Keep angles in range for numerical stability, though cos/sin are periodic
        inner[:, 2] = inner[:, 2] % 360.0
        
        side, ok = _min_outer_side_for_inner(inner, np.array([0.0, 0.0, 0.0]))
        if not ok or not np.isfinite(side):
            return 1e10
        
        # Add overlap penalty
        penalty = _compute_overlap_penalty(inner)
        return side + penalty * 1000.0  # Strong penalty

    # Optimization
    rng = np.random.default_rng(42)
    
    # Strategy 1: Nelder-Mead from seed
    if time.time() < deadline - 60.0:
        try:
            res = minimize(objective, inner_seed.flatten(), method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9})
            
            if np.isfinite(res.fun):
                inner_opt = res.x.reshape(11, 3)
                inner_opt[:, 2] = inner_opt[:, 2] % 360.0
                side_opt, ok_opt = _min_outer_side_for_inner(inner_opt, np.array([0.0, 0.0, 0.0]))
                
                if ok_opt and np.isfinite(side_opt) and side_opt < best_side:
                    if _check_validity(inner_opt, np.array([0.0, 0.0, 0.0]), side_opt):
                        best_inner = inner_opt
                        best_outer = np.array([0.0, 0.0, 0.0])
                        best_side = side_opt
                        best_score = 1.0 / best_side
        except Exception:
            pass

    # Strategy 2: Powell method (often better for non-smooth)
    if time.time() < deadline - 60.0:
        try:
            res = minimize(objective, best_inner.flatten(), method='Powell',
                          options={'maxiter': 500, 'xtol': 1e-7, 'ftol': 1e-9})
            
            if np.isfinite(res.fun):
                inner_opt = res.x.reshape(11, 3)
                inner_opt[:, 2] = inner_opt[:, 2] % 360.0
                side_opt, ok_opt = _min_outer_side_for_inner(inner_opt, np.array([0.0, 0.0, 0.0]))
                
                if ok_opt and np.isfinite(side_opt) and side_opt < best_side:
                    if _check_validity(inner_opt, np.array([0.0, 0.0, 0.0]), side_opt):
                        best_inner = inner_opt
                        best_outer = np.array([0.0, 0.0, 0.0])
                        best_side = side_opt
                        best_score = 1.0 / best_side
        except Exception:
            pass

    # Strategy 3: Random restarts
    for restart in range(10):
        if time.time() > deadline - 30.0:
            break
        
        # Perturb best solution
        scale = 0.5
        x0 = best_inner.flatten() + rng.normal(0, scale, size=33)
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9})
            
            if np.isfinite(res.fun):
                inner_opt = res.x.reshape(11, 3)
                inner_opt[:, 2] = inner_opt[:, 2] % 360.0
                side_opt, ok_opt = _min_outer_side_for_inner(inner_opt, np.array([0.0, 0.0, 0.0]))
                
                if ok_opt and np.isfinite(side_opt) and side_opt < best_side:
                    if _check_validity(inner_opt, np.array([0.0, 0.0, 0.0]), side_opt):
                        best_inner = inner_opt
                        best_outer = np.array([0.0, 0.0, 0.0])
                        best_side = side_opt
                        best_score = 1.0 / best_side
        except Exception:
            continue

    return best_inner, best_outer, best_side


# EVOLVE-BLOCK-END

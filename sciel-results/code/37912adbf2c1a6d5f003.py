# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, angle_deg):
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _all_vertices(inner_hexes):
    """Return array of shape (11*6, 2) with all vertices of all inner hexagons."""
    verts = []
    for i in range(len(inner_hexes)):
        cx, cy, ang = inner_hexes[i]
        v = _hex_vertices(cx, cy, 1.0, ang)
        verts.append(v)
    return np.vstack(verts)

def _check_separation_fast(inner_hexes, tol=1e-6):
    """Check all pairs are separated on at least one of 12 edge normals."""
    n = len(inner_hexes)
    verts_list = []
    normals_list = []
    for i in range(n):
        cx, cy, ang = inner_hexes[i]
        verts_list.append(_hex_vertices(cx, cy, 1.0, ang))
        an = np.radians(ang) + np.radians(30 + 60 * np.arange(6))
        normals_list.append(np.column_stack([np.cos(an), np.sin(an)]))
    
    for i in range(n):
        for j in range(i + 1, n):
            vi = verts_list[i]
            vj = verts_list[j]
            # 12 normals: 6 from i, 6 from j
            ni = vi @ normals_list[i].T  # (6,6)
            nj = vj @ normals_list[j].T  # (6,6)
            # For each of the 12 normals, check interval separation
            all_separated = False
            # Normals from hex i
            for k in range(6):
                proj_i = vi @ normals_list[i][k]
                proj_j = vj @ normals_list[i][k]
                if np.max(proj_i) < np.min(proj_j) - tol or np.max(proj_j) < np.min(proj_i) - tol:
                    all_separated = True
                    break
            if all_separated:
                continue
            # Normals from hex j
            for k in range(6):
                proj_i = vi @ normals_list[j][k]
                proj_j = vj @ normals_list[j][k]
                if np.max(proj_i) < np.min(proj_j) - tol or np.max(proj_j) < np.min(proj_i) - tol:
                    all_separated = True
                    break
            if not all_separated:
                return False
    return True

def _min_enclosing_side_fast(inner_hexes):
    """Compute minimum enclosing hexagon side (outer at origin, angle 0)."""
    normals = []
    for i in range(6):
        a = np.radians(30 + 60 * i)
        normals.append([np.cos(a), np.sin(a)])
    normals = np.array(normals)
    apothem_factor = np.sqrt(3) / 2
    
    max_proj = 0.0
    for hex_data in inner_hexes:
        cx, cy, ang = hex_data
        verts = _hex_vertices(cx, cy, 1.0, ang)
        projs = verts @ normals.T
        local_max = np.max(projs)
        if local_max > max_proj:
            max_proj = local_max
    
    return max_proj / apothem_factor

def _check_containment_fast(inner_hexes, side):
    """Check all inner vertices are inside outer hexagon of given side."""
    normals = []
    for i in range(6):
        a = np.radians(30 + 60 * i)
        normals.append([np.cos(a), np.sin(a)])
    normals = np.array(normals)
    apothem = side * np.sqrt(3) / 2
    
    for hex_data in inner_hexes:
        cx, cy, ang = hex_data
        verts = _hex_vertices(cx, cy, 1.0, ang)
        projs = verts @ normals.T
        if np.any(projs > apothem + 1e-6):
            return False
    return True

def _build_symmetric_positions(r1, r2, r3):
    """
    Build 11 hexagon positions using 6-fold symmetry.
    - 1 center at origin
    - 6 at radius r1 in directions 30, 90, 150, 210, 270, 330
    - 4 at radius r2 in directions 0, 120, 240, 300 (or similar)
    Actually, for 11 = 1 + 6 + 4, let's use:
    - 1 center
    - 6 at radius r1 (directions 30, 90, 150, 210, 270, 330)
    - 4 at radius r2 (directions 0, 90, 180, 270) -- but that breaks 6-fold symmetry
    
    Better: exploit the hexagonal symmetry properly.
    11 = 1 + 6 + 4. The 4 outer can be at 4 of the 6 directions at radius r2.
    But to maintain symmetry, we might use a different decomposition.
    
    Actually, the optimal packing of 11 hexagons in a hexagon likely has:
    - 1 center
    - 6 at first ring (distance sqrt(3))
    - 4 at second ring, but not all 6 (to save space)
    
    Let's parameterize:
    - r1: distance of 6 inner ring
    - r2: distance of 4 outer ring
    - theta: rotation of the outer 4 (but by symmetry, fix to specific angles)
    
    For 6-fold symmetry with 4 outer, we can place them at angles:
    0, 90, 180, 270 (4-fold symmetry) or at 60, 120, 180, 240, 300, 0 (all 6)
    
    Let's try: 1 center + 6 at r1 (30,90,150,210,270,330) + 4 at r2 (0, 120, 240, 300)
    This has 3-fold symmetry.
    """
    positions = np.zeros((11, 2))
    # Center
    positions[0] = [0.0, 0.0]
    # 6 inner ring
    for k in range(6):
        angle = np.radians(30 + 60 * k)
        positions[1 + k] = [r1 * np.cos(angle), r1 * np.sin(angle)]
    # 4 outer ring at angles 0, 120, 240, 300
    outer_angles = [0, 120, 240, 300]
    for k, angle_deg in enumerate(outer_angles):
        angle = np.radians(angle_deg)
        positions[7 + k] = [r2 * np.cos(angle), r2 * np.sin(angle)]
    return positions

def _separation_violation(inner_hexes):
    """Return a smooth measure of separation violation (0 if all separated)."""
    n = len(inner_hexes)
    verts_list = []
    normals_list = []
    for i in range(n):
        cx, cy, ang = inner_hexes[i]
        verts_list.append(_hex_vertices(cx, cy, 1.0, ang))
        an = np.radians(ang) + np.radians(30 + 60 * np.arange(6))
        normals_list.append(np.column_stack([np.cos(an), np.sin(an)]))
    
    total_violation = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            vi = verts_list[i]
            vj = verts_list[j]
            min_gap = np.inf
            # Check all 12 normals
            for k in range(6):
                n_vec = normals_list[i][k]
                proj_i = vi @ n_vec
                proj_j = vj @ n_vec
                gap = max(np.min(proj_j) - np.max(proj_i), np.min(proj_i) - np.max(proj_j))
                if gap < min_gap:
                    min_gap = gap
            for k in range(6):
                n_vec = normals_list[j][k]
                proj_i = vi @ n_vec
                proj_j = vj @ n_vec
                gap = max(np.min(proj_j) - np.max(proj_i), np.min(proj_i) - np.max(proj_j))
                if gap < min_gap:
                    min_gap = gap
            # If min_gap < 0, they overlap; violation = -min_gap
            if min_gap < 1e-6:
                total_violation += (1e-6 - min_gap) ** 2
    return total_violation

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Initial guess: touching hexagons
    d = np.sqrt(3)
    r1_init = d
    r2_init = 2 * d
    
    def _make_inner(r1, r2):
        positions = _build_symmetric_positions(r1, r2)
        return np.column_stack([positions, np.zeros(11)])
    
    # Initial solution
    inner_init = _make_inner(r1_init, r2_init)
    side_init = _min_enclosing_side_fast(inner_init)
    valid_init = _check_separation_fast(inner_init)
    
    best_inner = inner_init.copy()
    best_side = side_init
    
    if not valid_init:
        # Scale up until valid
        scale = 1.0
        while scale < 3.0:
            scale += 0.05
            inner_test = _make_inner(r1_init * scale, r2_init * scale)
            if _check_separation_fast(inner_test):
                best_inner = inner_test
                best_side = _min_enclosing_side_fast(inner_test)
                break
    
    # Optimization using scipy with symmetry parameterization
    def objective(params):
        r1, r2 = params
        if r1 < 0.5 or r2 < r1:
            return 1e10
        inner = _make_inner(r1, r2)
        side = _min_enclosing_side_fast(inner)
        # Add separation penalty
        viol = _separation_violation(inner)
        return side + 1000.0 * viol
    
    # Try multiple initial points
    initial_points = [
        [d, 2 * d],
        [d * 1.1, 2 * d * 1.1],
        [d * 0.95, 2 * d * 0.95],
        [d, 2 * d * 0.95],
        [d * 1.05, 2 * d],
    ]
    
    for init_pt in initial_points:
        if time.time() > deadline - 60.0:
            break
        try:
            res = minimize(
                objective,
                init_pt,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.success:
                r1_opt, r2_opt = res.x
                inner_opt = _make_inner(r1_opt, r2_opt)
                if _check_separation_fast(inner_opt):
                    side_opt = _min_enclosing_side_fast(inner_opt)
                    if side_opt < best_side:
                        best_inner = inner_opt
                        best_side = side_opt
        except Exception:
            pass
    
    # High-precision polish: coordinate descent on all 22 coordinates
    if time.time() < deadline - 30.0:
        positions = best_inner[:, :2].copy()
        angles = np.zeros(11)
        
        for iteration in range(50):
            if time.time() > deadline - 30.0:
                break
            improved = False
            # Try small perturbations in x and y for each hexagon
            for i in range(11):
                for delta in [0.001, -0.001, 0.0001, -0.0001]:
                    for axis in [0, 1]:
                        old_val = positions[i, axis]
                        positions[i, axis] = old_val + delta
                        inner_test = np.column_stack([positions, angles])
                        if _check_separation_fast(inner_test):
                            side_test = _min_enclosing_side_fast(inner_test)
                            if side_test < best_side - 1e-12:
                                best_side = side_test
                                best_inner = inner_test.copy()
                                improved = True
                            else:
                                positions[i, axis] = old_val
                        else:
                            positions[i, axis] = old_val
            if not improved:
                break
    
    # Final verification and return
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    side_length = float(best_side)
    
    return best_inner, outer_hex_data, side_length

# EVOLVE-BLOCK-END

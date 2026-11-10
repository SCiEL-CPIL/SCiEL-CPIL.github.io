# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _check_validity(inner, outer, s):
    """
    Check all validity conditions. Returns (is_valid, min_violation).
    inner: (12,3) array (x,y,angle_deg)
    outer: (3,) array (x,y,angle_deg)
    s: float side length
    """
    if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer)) and np.isfinite(s)):
        return False, -1.0

    if s < np.sqrt(12.0) - 1e-9:
        return False, -1.0

    ov = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(ov[i] - ov[j]) < 1e-6:
                return False, -1.0

    shift = np.array([outer[0], outer[1]])
    for k in range(12):
        cx, cy, ang = inner[k]
        cx2, cy2 = cx - shift[0], cy - shift[1]
        verts = _hex_vertices(cx2, cy2, 1.0, ang)
        for i in range(6):
            v1 = verts[i]
            v2 = verts[(i + 1) % 6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1.0) > 1e-8:
                return False, -1.0

    for i in range(12):
        vi = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for j in range(i + 1, 12):
            vj = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            all_normals = []
            for v in [vi, vj]:
                for e in range(6):
                    p1 = v[e]
                    p2 = v[(e + 1) % 6]
                    edge = p2 - p1
                    n = np.array([-edge[1], edge[0]])
                    n = n / (np.linalg.norm(n) + 1e-15)
                    all_normals.append(n)

            separated = False
            for n in all_normals:
                proj_i = np.dot(vi, n)
                proj_j = np.dot(vj, n)
                min_i, max_i = proj_i.min(), proj_i.max()
                min_j, max_j = proj_j.min(), proj_j.max()
                gap = max(min_j - max_i, min_i - max_j)
                if gap >= 1e-6:
                    separated = True
                    break
            if not separated:
                return False, -1.0

    for k in range(12):
        vi = _hex_vertices(inner[k, 0], inner[k, 1], 1.0, inner[k, 2])
        for v in vi:
            for e in range(6):
                p1 = ov[e]
                p2 = ov[(e + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False, -1.0

    return True, 0.0


def _compute_s_for_inner(inner, outer_angle_deg):
    """
    Given inner hexagon positions and a fixed outer hexagon angle,
    find the optimal center and minimal side length s.
    Uses LP via linear programming for exact solution.
    """
    all_verts = []
    for k in range(12):
        v = _hex_vertices(inner[k, 0], inner[k, 1], 1.0, inner[k, 2])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)  # (72, 2)
    
    a = np.deg2rad(outer_angle_deg)
    normals = np.array([[np.cos(a + np.deg2rad(30 + 60 * i)),
                         np.sin(a + np.deg2rad(30 + 60 * i))] for i in range(6)])
    
    # For fixed angle, find c minimizing max_i (M_i - dot(c, n_i))
    # where M_i = max_v dot(v, n_i)
    dots = all_verts @ normals.T  # (72, 6)
    M = dots.max(axis=0)  # (6,)
    
    # Use scipy minimize for the 2D problem: minimize max_i(M_i - c·n_i)
    def objective(c):
        vals = M - c @ normals.T
        return vals.max()
    
    c0 = all_verts.mean(axis=0)
    res = minimize(objective, c0, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
    c_opt = res.x
    s_val = objective(c_opt) / (np.sqrt(3) / 2)
    
    return s_val, np.array([c_opt[0], c_opt[1], outer_angle_deg])


def _full_objective(params, inner_fixed_angles=True):
    """
    Objective for joint optimization.
    params: [cx, cy, angle_deg, x1, y1, a1, ..., x12, y12, a12]
    Returns the outer hexagon side length s.
    """
    cx, cy, ang = params[0], params[1], params[2]
    inner = params[3:].reshape(12, 3)
    
    # Compute all inner vertices
    all_verts = []
    for k in range(12):
        v = _hex_vertices(inner[k, 0], inner[k, 1], 1.0, inner[k, 2])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    a = np.deg2rad(ang)
    normals = np.array([[np.cos(a + np.deg2rad(30 + 60 * i)),
                         np.sin(a + np.deg2rad(30 + 60 * i))] for i in range(6)])
    
    dots = all_verts @ normals.T
    M = dots.max(axis=0)
    vals = M - np.array([cx, cy]) @ normals.T
    s_val = vals.max() / (np.sqrt(3) / 2)
    
    # Add penalties for overlapping inner hexagons
    penalty = 0.0
    for i in range(12):
        vi = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for j in range(i + 1, 12):
            vj = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            # Compute minimum distance between the two hexagons via SAT
            min_gap = np.inf
            all_normals = []
            for v in [vi, vj]:
                for e in range(6):
                    p1 = v[e]
                    p2 = v[(e + 1) % 6]
                    edge = p2 - p1
                    n = np.array([-edge[1], edge[0]])
                    n = n / (np.linalg.norm(n) + 1e-15)
                    all_normals.append(n)
            
            for n in all_normals:
                proj_i = np.dot(vi, n)
                proj_j = np.dot(vj, n)
                min_i, max_i = proj_i.min(), proj_i.max()
                min_j, max_j = proj_j.min(), proj_j.max()
                gap = max(min_j - max_i, min_i - max_j)
                if gap > 0:
                    min_gap = min(min_gap, gap)
                else:
                    min_gap = min(min_gap, gap)
                    break  # Overlapping on this axis
            
            if min_gap < 1e-6:
                penalty += max(0, 1e-6 - min_gap) ** 2
    
    return s_val + penalty


def _seed_config_hex_cluster():
    """Hexagonal cluster: center + 6 ring1 + 5 ring2"""
    sqrt3 = np.sqrt(3)
    return np.array([
        [0, 0, 0],
        [2, 0, 0],
        [-2, 0, 0],
        [1, sqrt3, 0],
        [-1, sqrt3, 0],
        [1, -sqrt3, 0],
        [-1, -sqrt3, 0],
        [3, sqrt3, 0],
        [-3, sqrt3, 0],
        [3, -sqrt3, 0],
        [-3, -sqrt3, 0],
        [0, 2 * sqrt3, 0],
    ])


def _seed_config_spread():
    """More spread out arrangement"""
    sqrt3 = np.sqrt(3)
    return np.array([
        [0, 0, 0],
        [2.2, 0, 0],
        [-2.2, 0, 0],
        [1.1, 1.9, 0],
        [-1.1, 1.9, 0],
        [1.1, -1.9, 0],
        [-1.1, -1.9, 0],
        [3.3, 1.9, 0],
        [-3.3, 1.9, 0],
        [3.3, -1.9, 0],
        [-3.3, -1.9, 0],
        [0, 3.8, 0],
    ])


def _seed_config_rotated():
    """Cluster with slight rotation of outer hexagons"""
    sqrt3 = np.sqrt(3)
    base = _seed_config_hex_cluster()
    # Rotate some hexagons slightly
    base[6:, 2] = 10.0  # rotate ring 2 by 10 degrees
    return base


def _optimize_from_seed(seed_inner, start_time, deadline):
    """
    Multi-stage optimization from a seed configuration.
    Stage 1: Optimize outer hexagon (center, angle) for fixed inner.
    Stage 2: Joint Nelder-Mead optimization of all parameters.
    Stage 3: Fine-tuning with smaller steps.
    """
    # Stage 1: Optimize outer for fixed inner
    best_s = np.inf
    best_outer = None
    
    for angle_deg in np.arange(0, 60, 2.0):
        if time.time() > deadline - 30:
            break
        s_val, outer = _compute_s_for_inner(seed_inner, angle_deg)
        if s_val < best_s:
            best_s = s_val
            best_outer = outer
    
    # Refine angle
    if best_outer is not None:
        for angle_deg in np.arange(best_outer[2] - 2, best_outer[2] + 2.1, 0.25):
            if time.time() > deadline - 30:
                break
            s_val, outer = _compute_s_for_inner(seed_inner, angle_deg)
            if s_val < best_s:
                best_s = s_val
                best_outer = outer
    
    # Stage 2: Joint optimization using Nelder-Mead
    # Parameters: [cx, cy, outer_angle, x1, y1, a1, ..., x12, y12, a12]
    # We'll work in a local coordinate system centered at the outer hexagon center
    # for better conditioning
    
    cx0, cy0, ang0 = best_outer[0], best_outer[1], best_outer[2]
    
    # Convert inner to local coords
    local_inner = seed_inner.copy()
    local_inner[:, 0] -= cx0
    local_inner[:, 1] -= cy0
    
    params0 = np.concatenate([[0, 0, ang0], local_inner.flatten()])
    
    def obj(params):
        cx, cy, ang = params[0], params[1], params[2]
        inner_local = params[3:].reshape(12, 3)
        
        # Convert to global
        inner_global = inner_local.copy()
        inner_global[:, 0] += cx
        inner_global[:, 1] += cy
        
        all_verts = []
        for k in range(12):
            v = _hex_vertices(inner_global[k, 0], inner_global[k, 1], 1.0, inner_global[k, 2])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        a = np.deg2rad(ang)
        normals = np.array([[np.cos(a + np.deg2rad(30 + 60 * i)),
                             np.sin(a + np.deg2rad(30 + 60 * i))] for i in range(6)])
        
        dots = all_verts @ normals.T
        M = dots.max(axis=0)
        vals = M - np.array([cx, cy]) @ normals.T
        s_val = vals.max() / (np.sqrt(3) / 2)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(12):
            vi = _hex_vertices(inner_global[i, 0], inner_global[i, 1], 1.0, inner_global[i, 2])
            for j in range(i + 1, 12):
                vj = _hex_vertices(inner_global[j, 0], inner_global[j, 1], 1.0, inner_global[j, 2])
                min_gap = np.inf
                all_normals = []
                for v in [vi, vj]:
                    for e in range(6):
                        p1 = v[e]
                        p2 = v[(e + 1) % 6]
                        edge = p2 - p1
                        n = np.array([-edge[1], edge[0]])
                        n = n / (np.linalg.norm(n) + 1e-15)
                        all_normals.append(n)
                
                for n in all_normals:
                    proj_i = np.dot(vi, n)
                    proj_j = np.dot(vj, n)
                    min_i, max_i = proj_i.min(), proj_i.max()
                    min_j, max_j = proj_j.min(), proj_j.max()
                    gap = max(min_j - max_i, min_i - max_j)
                    if gap < 0:
                        min_gap = min(min_gap, gap)
                        break
                    else:
                        min_gap = min(min_gap, gap)
                
                if min_gap < 1e-4:
                    penalty += max(0, 1e-4 - min_gap) ** 2 * 100
        
        return s_val + penalty
    
    # Nelder-Mead optimization
    res = minimize(obj, params0, method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8,
                            'adaptive': True})
    
    if res.fun < best_s and time.time() < deadline - 10:
        best_params = res.x
        best_s = res.fun  # This includes penalty, need to recompute clean s
        
        # Recompute clean s
        cx, cy, ang = best_params[0], best_params[1], best_params[2]
        inner_local = best_params[3:].reshape(12, 3)
        inner_global = inner_local.copy()
        inner_global[:, 0] += cx
        inner_global[:, 1] += cy
        
        all_verts = []
        for k in range(12):
            v = _hex_vertices(inner_global[k, 0], inner_global[k, 1], 1.0, inner_global[k, 2])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        a = np.deg2rad(ang)
        normals = np.array([[np.cos(a + np.deg2rad(30 + 60 * i)),
                             np.sin(a + np.deg2rad(30 + 60 * i))] for i in range(6)])
        
        dots = all_verts @ normals.T
        M = dots.max(axis=0)
        vals = M - np.array([cx, cy]) @ normals.T
        clean_s = vals.max() / (np.sqrt(3) / 2)
        
        # Check if penalty was zero (no overlaps)
        has_overlap = False
        for i in range(12):
            vi = _hex_vertices(inner_global[i, 0], inner_global[i, 1], 1.0, inner_global[i, 2])
            for j in range(i + 1, 12):
                vj = _hex_vertices(inner_global[j, 0], inner_global[j, 1], 1.0, inner_global[j, 2])
                all_normals = []
                for v in [vi, vj]:
                    for e in range(6):
                        p1 = v[e]
                        p2 = v[(e + 1) % 6]
                        edge = p2 - p1
                        n = np.array([-edge[1], edge[0]])
                        n = n / (np.linalg.norm(n) + 1e-15)
                        all_normals.append(n)
                
                for n in all_normals:
                    proj_i = np.dot(vi, n)
                    proj_j = np.dot(vj, n)
                    min_i, max_i = proj_i.min(), proj_i.max()
                    min_j, max_j = proj_j.min(), proj_j.max()
                    gap = max(min_j - max_i, min_i - max_j)
                    if gap < 1e-6:
                        has_overlap = True
                        break
                if has_overlap:
                    break
            if has_overlap:
                break
        
        if not has_overlap:
            best_s = clean_s
            best_outer = np.array([cx, cy, ang])
            seed_inner = inner_global
    
    # Stage 3: Fine-tune outer only
    if time.time() < deadline - 5:
        for angle_deg in np.arange(best_outer[2] - 1, best_outer[2] + 1.1, 0.1):
            s_val, outer = _compute_s_for_inner(seed_inner, angle_deg)
            if s_val < best_s:
                best_s = s_val
                best_outer = outer
    
    return seed_inner, best_outer, best_s


def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon, minimizing the outer hexagon side length.
    Returns (inner, outer, s).
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    np.random.seed(42)
    
    # Multiple seed configurations
    seeds = [
        _seed_config_hex_cluster(),
        _seed_config_spread(),
        _seed_config_rotated(),
    ]
    
    best_inner = None
    best_outer = None
    best_s = np.inf
    
    for seed in seeds:
        if time.time() > deadline - 60:
            break
        
        inner, outer, s = _optimize_from_seed(seed, start_time, deadline)
        
        if s < best_s:
            best_s = s
            best_inner = inner.copy()
            best_outer = outer.copy()
    
    # Final verification
    if best_inner is None:
        # Fallback
        inner = _seed_config_hex_cluster()
        s, outer = _compute_s_for_inner(inner, 0.0)
        best_inner = inner
        best_outer = outer
        best_s = s
    
    valid, _ = _check_validity(best_inner, best_outer, best_s)
    if not valid:
        # Fallback to a known valid configuration
        inner = np.array([
            [0, 0, 0],
            [2.5, 0, 0],
            [-2.5, 0, 0],
            [1.25, 2.17, 0],
            [-1.25, 2.17, 0],
            [1.25, -2.17, 0],
            [-1.25, -2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, -2.17, 0],
            [-3.75, -2.17, 0],
            [0, -4, 0],
        ])
        outer = np.array([0, 0, 0])
        s = 8.0
        return inner, outer, s
    
    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END

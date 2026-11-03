import math
import time
import numpy as np
from scipy.optimize import linprog, minimize_scalar

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon that contains all points in verts.
    The container hexagon has its flat-to-flat axis at outer_angle_deg.
    Returns (cx, cy, side) or None.
    """
    base_angle = math.radians(outer_angle_deg)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    vdots = verts @ normals.T  # shape (M, 6)
    
    A_rows = []
    b_vals = []
    
    for j in range(6):
        nx, ny = normals[j]
        A_rows.append(np.tile([-nx, -ny, -1.0], (M, 1)))
        b_vals.append(-vdots[:, j])
        A_rows.append(np.tile([nx, ny, -1.0], (M, 1)))
        b_vals.append(vdots[:, j])
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    
    A = np.vstack([A, np.array([[0.0, 0.0, -1.0]])])
    b = np.concatenate([b, [0.0]])
    
    c_obj = np.array([0.0, 0.0, 1.0])
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs', options={'presolve': True})
        if res.success:
            cx, cy, R = res.x
            side = 2.0 * R * _INVSQRT3
            return cx, cy, side
    except Exception:
        pass
    return None

def _build_cluster(axial_coords, cluster_rot_deg=0.0):
    """Build positions for a hex cluster from axial coordinates with optional rotation."""
    sqrt3 = _SQRT3
    positions = []
    for q, r in axial_coords:
        x = sqrt3 * q + sqrt3/2 * r
        y = 1.5 * r
        positions.append((x, y))
    
    if abs(cluster_rot_deg) > 1e-10:
        theta = math.radians(cluster_rot_deg)
        ct, st = math.cos(theta), math.sin(theta)
        positions = [(x*ct - y*st, x*st + y*ct) for x, y in positions]
    
    return positions

def _get_configs():
    """Generate multiple cluster configurations."""
    configs = []
    
    # Standard 11-hex cluster: center + 6 ring + 4 from ring 2
    center = (0, 0)
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    # Different selections of 4 from ring 2
    ring2_options = [
        [(2, -1), (0, -2), (-2, 1), (0, 2)],
        [(2, -1), (-1, -2), (-2, 1), (1, 1)],
        [(2, 0), (-2, 0), (0, 2), (0, -2)],
        [(2, -1), (2, -2), (-2, 1), (-2, 2)],
        [(1, -2), (-2, 0), (-1, 2), (2, -1)],
        [(2, -1), (-1, -2), (0, 2), (-2, 0)],
    ]
    
    for ring2 in ring2_options:
        coords = [center] + ring1 + ring2
        configs.append(_build_cluster(coords, 0.0))
        configs.append(_build_cluster(coords, 30.0))
        configs.append(_build_cluster(coords, 15.0))
        configs.append(_build_cluster(coords, -15.0))
    
    return configs

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 0.0
    
    configs = _get_configs()
    inner_angles = [0.0, 30.0]
    
    for config_idx, positions in enumerate(configs):
        if time.time() > _DEADLINE - 10.0:
            break
        
        for inner_angle in inner_angles:
            if time.time() > _DEADLINE - 10.0:
                break
            
            all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
            
            # Coarse grid search over outer angle
            coarse_results = []
            for outer_angle_deg in np.arange(0, 30.25, 0.5):
                if time.time() > _DEADLINE - 10.0:
                    break
                res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                if res:
                    coarse_results.append((res[2], outer_angle_deg))
            
            if not coarse_results:
                continue
            
            coarse_results.sort(key=lambda x: x[0])
            best_coarse_side, best_coarse_angle = coarse_results[0]
            
            # Local optimization
            candidate_angles = [best_coarse_angle]
            for side, angle in coarse_results[:5]:
                if abs(angle - best_coarse_angle) > 1e-6:
                    candidate_angles.append(angle)
            
            best_local_side = best_coarse_side
            best_local_angle = best_coarse_angle
            best_local_center = None
            
            for start_angle in candidate_angles:
                if time.time() > _DEADLINE - 15.0:
                    break
                
                lo = max(0.0, start_angle - 3.0)
                hi = min(30.0, start_angle + 3.0)
                
                try:
                    def obj(a):
                        res = _min_enclosing_hex_side(all_vertices, a)
                        return 1e6 if res is None else res[2]
                    
                    res_opt = minimize_scalar(obj, bounds=(lo, hi), method='bounded',
                                              options={'xatol': 1e-10, 'maxiter': 200})
                    if res_opt.success and res_opt.fun < best_local_side:
                        best_local_side = res_opt.fun
                        best_local_angle = res_opt.x
                except Exception:
                    pass
            
            # High-precision polish
            if best_local_side < best_side and time.time() < _DEADLINE - 20.0:
                # Fine grid
                lo = max(0.0, best_local_angle - 0.01)
                hi = min(30.0, best_local_angle + 0.01)
                for outer_angle_deg in np.arange(lo, hi + 0.001, 0.001):
                    if time.time() > _DEADLINE - 15.0:
                        break
                    res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                    if res:
                        cx, cy, side = res
                        if side < best_local_side:
                            best_local_side = side
                            best_local_angle = outer_angle_deg
                            best_local_center = (cx, cy)
                
                # Even finer
                if time.time() < _DEADLINE - 15.0:
                    lo2 = max(0.0, best_local_angle - 0.001)
                    hi2 = min(30.0, best_local_angle + 0.001)
                    for outer_angle_deg in np.arange(lo2, hi2 + 0.0001, 0.0001):
                        if time.time() > _DEADLINE - 12.0:
                            break
                        res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                        if res:
                            cx, cy, side = res
                            if side < best_local_side:
                                best_local_side = side
                                best_local_angle = outer_angle_deg
                                best_local_center = (cx, cy)
                
                # Bounded scalar minimization
                if time.time() < _DEADLINE - 12.0:
                    lo3 = max(0.0, best_local_angle - 0.0001)
                    hi3 = min(30.0, best_local_angle + 0.0001)
                    try:
                        def obj3(a):
                            res = _min_enclosing_hex_side(all_vertices, a)
                            return 1e6 if res is None else res[2]
                        
                        res_opt2 = minimize_scalar(obj3, bounds=(lo3, hi3), method='bounded',
                                                   options={'xatol': 1e-13, 'maxiter': 500})
                        if res_opt2.success and res_opt2.fun < best_local_side:
                            best_local_side = res_opt2.fun
                            best_local_angle = res_opt2.x
                    except Exception:
                        pass
                
                # Ultra-fine grid
                if time.time() < _DEADLINE - 10.0:
                    lo4 = max(0.0, best_local_angle - 0.00005)
                    hi4 = min(30.0, best_local_angle + 0.00005)
                    for outer_angle_deg in np.arange(lo4, hi4 + 0.000005, 0.000005):
                        if time.time() > _DEADLINE - 8.0:
                            break
                        res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                        if res:
                            cx, cy, side = res
                            if side < best_local_side:
                                best_local_side = side
                                best_local_angle = outer_angle_deg
                                best_local_center = (cx, cy)
            
            if best_local_side < best_side:
                res_final = _min_enclosing_hex_side(all_vertices, best_local_angle)
                if res_final:
                    cx, cy, side = res_final
                    if side < best_side:
                        best_side = side
                        best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                        best_center = (cx, cy)
                        best_outer_angle = best_local_angle
                        best_inner_angle = inner_angle
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle

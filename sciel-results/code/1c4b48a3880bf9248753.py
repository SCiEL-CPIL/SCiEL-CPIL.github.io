import math
import time
import numpy as np
from scipy.optimize import linprog, minimize_scalar, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    """
    base_angle = math.radians(outer_angle_deg + 30)
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
                      method='highs', options={'presolve': True, 'dual_feasibility_tolerance': 1e-12, 'primal_feasibility_tolerance': 1e-12})
        if res.success:
            cx, cy, R = res.x
            side = R * _INVSQRT3 * 2.0
            return cx, cy, side
    except Exception:
        pass
    return None

def _smooth_enclosing_side(verts, outer_angle_deg, cx, cy, beta=200.0):
    """
    Smooth surrogate for the enclosing hexagon side length using log-sum-exp.
    Given a fixed center (cx, cy) and outer angle, returns a smooth approximation
    of the minimum side length needed.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    center = np.array([cx, cy])
    c_dots = center @ normals.T  # shape (6,)
    
    # For each normal direction j, the required half-width is:
    # max_i (dot(vert_i, n_j) - dot(center, n_j))
    # We use log-sum-exp to smooth the max over i for each j.
    vdots = verts @ normals.T  # shape (M, 6)
    diffs = vdots - c_dots[np.newaxis, :]  # shape (M, 6)
    
    # Soft max over i for each j
    max_diffs = np.max(diffs, axis=0)  # shape (6,)
    scaled = beta * diffs
    exp_scaled = np.exp(scaled - scaled.max(axis=0, keepdims=True))
    soft_max = np.log(exp_scaled.sum(axis=0)) / beta + diffs.max(axis=0) / beta * 0  # simplified
    # Actually use proper log-sum-exp:
    m = diffs.max(axis=0, keepdims=True)
    log_sum_exp = m.squeeze() + np.log(np.exp(diffs - m).sum(axis=0)) / beta
    
    # The required R is max over j of soft_max_j
    R = log_sum_exp.max()
    side = R * _INVSQRT3 * 2.0
    return side

def _objective_for_angle(verts, outer_angle_deg):
    """Return the side length for a given outer angle, or a large penalty."""
    res = _min_enclosing_hex_side(verts, outer_angle_deg)
    if res is None:
        return 1e6
    return res[2]

def _smooth_objective(verts, outer_angle_deg, beta=500.0):
    """
    Smooth surrogate for the enclosing hexagon side length using log-sum-exp.
    This makes the objective differentiable so gradient-based methods can work.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # For a given center (cx,cy), the required half-width in direction j is:
    # max_i (dot(vert_i, n_j) - dot((cx,cy), n_j))
    # We need to find cx,cy to minimize max over j of this.
    # Use LP to get exact value, but for smooth surrogate we approximate.
    # Actually, let's just use the LP-based objective but with a penalty for infeasibility.
    res = _min_enclosing_hex_side(verts, outer_angle_deg)
    if res is None:
        return 1e6
    return res[2]

def _nelder_mead_polish(verts, outer_angle_deg, cx, cy):
    """
    Use Nelder-Mead to jointly optimize (cx, cy, angle) for the enclosing hexagon.
    Returns improved (cx, cy, side, angle) or None.
    """
    def obj(params):
        px, py, pa = params
        pa_deg = pa * 180.0 / math.pi
        res = _min_enclosing_hex_side(verts, pa_deg)
        if res is None:
            return 1e6
        return res[2]
    
    initial = np.array([cx, cy, math.radians(outer_angle_deg)])
    try:
        res_nm = minimize(obj, initial, method='Nelder-Mead',
                          options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 500, 'maxfev': 300})
        if res_nm.success and res_nm.fun < 1e5:
            px, py, pa = res_nm.x
            pa_deg = pa * 180.0 / math.pi
            final_res = _min_enclosing_hex_side(verts, pa_deg)
            if final_res:
                return final_res[0], final_res[1], final_res[2], pa_deg
    except Exception:
        pass
    return None

def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.0000001
    u = (sqrt3*scale, 0.0)
    v = (sqrt3/2*scale, 1.5*scale)
    
    coords = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i*u[0] + j*v[0]
            y = i*u[1] + j*v[1]
            coords.append((x, y, i, j))
    
    if layout_idx == 0:
        target = [(0,0)]
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        target.extend(ring1)
        ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
        target.extend(ring2_sorted[:4])
        positions = []
        for ti, tj in target:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x,y))
        return positions[:N]
    elif layout_idx == 1:
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 2:
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 3:
        positions = []
        y1 = -3.0*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 1):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 4:
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    else:
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return [(x,y) for x,y,i,j in coords_sorted[:N]]

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
    
    num_layouts = 6
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        for inner_angle in [0.0, 30.0, 15.0, -15.0, 45.0, -45.0, 7.5, -7.5]:
            if time.time() > _DEADLINE - 10.0:
                break
            
            all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
            
            # Coarse grid search to find good starting points
            coarse_results = []
            for outer_angle_deg in np.arange(0, 30.25, 0.5):
                if time.time() > _DEADLINE - 10.0:
                    break
                res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                if res:
                    coarse_results.append((res[2], outer_angle_deg))
            
            if not coarse_results:
                continue
            
            # Find the best coarse result and its neighbors for local optimization
            coarse_results.sort(key=lambda x: x[0])
            best_coarse_side, best_coarse_angle = coarse_results[0]
            
            # Use minimize_scalar to refine around the best coarse angle
            # Try multiple starting points for robustness
            candidate_angles = [best_coarse_angle]
            # Add neighbors from coarse search
            for side, angle in coarse_results[:5]:
                if abs(angle - best_coarse_angle) > 1e-6:
                    candidate_angles.append(angle)
            
            best_local_side = best_coarse_side
            best_local_angle = best_coarse_angle
            best_local_center = None
            
            for start_angle in candidate_angles:
                if time.time() > _DEADLINE - 15.0:
                    break
                
                # Define bounds for local search
                lo = max(0.0, start_angle - 3.0)
                hi = min(30.0, start_angle + 3.0)
                
                try:
                    res_opt = minimize_scalar(
                        lambda a: _objective_for_angle(all_vertices, a),
                        bounds=(lo, hi),
                        method='bounded',
                        options={'xatol': 1e-14, 'maxiter': 300}
                    )
                    if res_opt.success:
                        opt_angle = res_opt.x
                        opt_side = res_opt.fun
                        if opt_side < best_local_side:
                            best_local_side = opt_side
                            best_local_angle = opt_angle
                except Exception:
                    pass
            
            # Final high-precision polish: very fine grid around the best angle
            if best_local_side < best_side and time.time() < _DEADLINE - 20.0:
                # First pass: fine grid
                lo = max(0.0, best_local_angle - 0.01)
                hi = min(30.0, best_local_angle + 0.01)
                for outer_angle_deg in np.arange(lo, hi + 0.0001, 0.001):
                    if time.time() > _DEADLINE - 15.0:
                        break
                    res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                    if res:
                        cx, cy, side = res
                        if side < best_local_side:
                            best_local_side = side
                            best_local_angle = outer_angle_deg
                            best_local_center = (cx, cy)
                
                # Second pass: even finer grid around the refined angle
                if time.time() < _DEADLINE - 15.0:
                    lo2 = max(0.0, best_local_angle - 0.001)
                    hi2 = min(30.0, best_local_angle + 0.001)
                    for outer_angle_deg in np.arange(lo2, hi2 + 0.00001, 0.0001):
                        if time.time() > _DEADLINE - 15.0:
                            break
                        res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                        if res:
                            cx, cy, side = res
                            if side < best_local_side:
                                best_local_side = side
                                best_local_angle = outer_angle_deg
                                best_local_center = (cx, cy)
            
            # Get the center for the best angle with final exact LP solve
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
    
    # Final ultra-precision polish: Nelder-Mead on (cx, cy, angle) with tight tolerances
    if time.time() < _DEADLINE - 8.0:
        all_vertices_final = np.vstack([_hex_vertices(px, py, best_inner_angle) for px, py in 
                                        [(p[0], p[1]) for p in best_inner]])
        
        # Nelder-Mead joint optimization
        nm_result = _nelder_mead_polish(all_vertices_final, best_outer_angle, 
                                         best_center[0], best_center[1])
        if nm_result is not None:
            nm_cx, nm_cy, nm_side, nm_angle = nm_result
            if nm_side < best_side:
                best_side = nm_side
                best_center = (nm_cx, nm_cy)
                best_outer_angle = nm_angle
        
        # One more minimize_scalar with very tight bounds for angle only
        if time.time() < _DEADLINE - 5.0:
            lo_final = max(0.0, best_outer_angle - 0.0001)
            hi_final = min(30.0, best_outer_angle + 0.0001)
            try:
                res_final_opt = minimize_scalar(
                    lambda a: _objective_for_angle(all_vertices_final, a),
                    bounds=(lo_final, hi_final),
                    method='bounded',
                    options={'xatol': 1e-14, 'maxiter': 200}
                )
                if res_final_opt.success:
                    final_angle = res_final_opt.x
                    final_res = _min_enclosing_hex_side(all_vertices_final, final_angle)
                    if final_res:
                        cx, cy, side = final_res
                        if side < best_side:
                            best_side = side
                            best_center = (cx, cy)
                            best_outer_angle = final_angle
            except Exception:
                pass
    
    return best_inner, best_center, best_side, best_outer_angle

"""
Optimized hexagon packing: 11 unit hexagons in a compact rigid hexagonal cluster.
Uses smooth optimization (Nelder-Mead) with a log-sum-exp surrogate for the
non-smooth containment objective, jointly optimizing inner positions and outer
hexagon parameters.
"""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # Safety margin before 150s limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """
    Get vertices of a unit regular hexagon centered at (cx, cy) with rotation.
    Standard orientation: vertex at angle 0.
    """
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _find_min_enclosing_hex_side_lp(vertices, outer_angle_deg):
    """
    Find the minimum side length of a regular hexagon with given angle
    that contains all given vertices using LP.
    Returns (center_x, center_y, side_length) or None if infeasible.
    """
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    A_ub = []
    b_ub = []
    
    for vx, vy in vertices:
        for nx, ny in normals:
            v_dot_n = vx * nx + vy * ny
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-v_dot_n)
            A_ub.append([nx, ny, -1.0])
            b_ub.append(v_dot_n)
    
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    c_vec = [0.0, 0.0, 1.0]
    
    try:
        res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, 
                      bounds=[(None, None), (None, None), (0, None)], 
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R * 2.0 / math.sqrt(3)
            return cx, cy, side
        else:
            return None
    except Exception:
        return None

def _smooth_objective(params, all_inner_verts, beta=50.0):
    """
    Smooth surrogate for the minimum enclosing hexagon side length.
    params: [dx_0, dy_0, ..., dx_10, dy_10, outer_cx, outer_cy, outer_angle_deg]
    Uses log-sum-exp to smooth the max over all vertex-normal pairs.
    """
    n_hex = N
    # Extract parameters
    inner_deltas = params[:2*n_hex]
    outer_cx = params[2*n_hex]
    outer_cy = params[2*n_hex+1]
    outer_angle_deg = params[2*n_hex+2]
    
    # Compute all vertices of inner hexagons with small displacements
    # Base positions are stored globally
    global _base_positions
    all_verts = []
    for i in range(n_hex):
        bx, by = _base_positions[i]
        dx, dy = inner_deltas[2*i], inner_deltas[2*i+1]
        cx, cy = bx + dx, by + dy
        verts = _get_hex_vertices(cx, cy, 0.0)
        all_verts.extend(verts)
    
    # Convert to numpy array
    all_verts = np.array(all_verts)
    
    # Outer hexagon normals
    outer_angle_rad = math.radians(outer_angle_deg)
    normals = []
    for k in range(6):
        a = outer_angle_rad + math.pi/6 + math.pi * k / 3
        normals.append((math.cos(a), math.sin(a)))
    normals = np.array(normals)  # shape (6, 2)
    
    # For each vertex v and normal n, compute distance: (v - c) . n
    # We need: max over all (v, n) of (v - c) . n <= R
    # Also: max over all (v, n) of (c - v) . n <= R
    # i.e., |max((v-c).n)| <= R for all n
    
    # Compute all projections
    # all_verts: (M, 2), normals: (6, 2)
    # For each normal, compute max projection and min projection
    R_vals = []
    for nx, ny in normals:
        projections = all_verts[:, 0] * nx + all_verts[:, 1] * ny
        center_proj = outer_cx * nx + outer_cy * ny
        # Max distance from center in direction n
        max_dist = np.max(projections) - center_proj
        min_dist = center_proj - np.min(projections)
        R_vals.append(max_dist)
        R_vals.append(min_dist)
    
    R_vals = np.array(R_vals)
    
    # Smooth max using log-sum-exp
    # max(x) ≈ (1/beta) * log(sum(exp(beta * x)))
    # But we want to minimize the max, so we use the soft-max
    if beta == 0:
        return float(np.max(R_vals)) * 2.0 / math.sqrt(3)
    
    # For numerical stability, shift by max
    R_shifted = R_vals - np.max(R_vals)
    log_sum_exp = np.max(R_vals) + np.log(np.sum(np.exp(beta * R_shifted))) / beta
    R_soft = log_sum_exp
    
    side = R_soft * 2.0 / math.sqrt(3)
    
    # Penalty for negative R (shouldn't happen but just in case)
    if R_soft < 0:
        side += 100.0 * (-R_soft)
    
    return side

def construct():
    global _START_TIME, _DEADLINE, _base_positions
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # 1. Generate the 11 closest points on a triangular lattice
    candidates = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            d_sq = 3 * (q*q + r*r + q*r)
            candidates.append((d_sq, q, r))
    
    candidates.sort(key=lambda x: x[0])
    selected = candidates[:N]
    
    sqrt3 = math.sqrt(3)
    scale = 1.000001
    
    positions = []
    for _, q, r in selected:
        x = 1.5 * q * scale
        y = sqrt3 * (r + q / 2.0) * scale
        positions.append((x, y))
    
    _base_positions = positions
    
    # Pre-compute base vertices for LP fallback
    all_vertices_base = []
    for px, py in positions:
        verts = _get_hex_vertices(px, py, 0.0)
        all_vertices_base.extend(verts)
    
    # First, find a good starting point using the LP approach
    best_side_lp = float('inf')
    best_center_lp = (0.0, 0.0)
    best_outer_angle_lp = 0.0
    
    # Coarse search for outer angle
    coarse_step = 1.0
    for ang_deg in np.arange(0, 60.0, coarse_step):
        if time.time() > _DEADLINE - 30.0:
            break
        res = _find_min_enclosing_hex_side_lp(all_vertices_base, ang_deg)
        if res:
            cx, cy, side = res
            if side < best_side_lp:
                best_side_lp = side
                best_center_lp = (cx, cy)
                best_outer_angle_lp = ang_deg
    
    # Fine search around best
    if best_side_lp < float('inf'):
        fine_step = 0.1
        for ang_deg in np.arange(best_outer_angle_lp - 1.0, best_outer_angle_lp + 1.1, fine_step):
            if ang_deg < 0 or ang_deg >= 60:
                continue
            if time.time() > _DEADLINE - 25.0:
                break
            res = _find_min_enclosing_hex_side_lp(all_vertices_base, ang_deg)
            if res:
                cx, cy, side = res
                if side < best_side_lp:
                    best_side_lp = side
                    best_center_lp = (cx, cy)
                    best_outer_angle_lp = ang_deg
    
    # Now use smooth optimization to refine
    # Parameters: [dx_0, dy_0, ..., dx_10, dy_10, outer_cx, outer_cy, outer_angle_deg]
    # Start from LP solution with zero inner displacements
    x0 = np.zeros(2*N + 3)
    x0[2*N] = best_center_lp[0]
    x0[2*N+1] = best_center_lp[1]
    x0[2*N+2] = best_outer_angle_lp
    
    best_result = (best_side_lp, best_center_lp, best_outer_angle_lp, [0.0]*(2*N))
    
    # Try smooth optimization with different beta values
    for beta in [20.0, 50.0, 100.0]:
        if time.time() > _DEADLINE - 15.0:
            break
        
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(all_vertices_base, beta),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if time.time() > _DEADLINE - 10.0:
                break
            
            if result.success or result.fun < best_result[0]:
                # Verify with exact LP
                params = result.x
                inner_deltas = params[:2*N]
                outer_cx = params[2*N]
                outer_cy = params[2*N+1]
                outer_angle_deg = params[2*N+2]
                
                # Compute actual vertices with displacements
                all_verts_actual = []
                for i in range(N):
                    bx, by = _base_positions[i]
                    dx, dy = inner_deltas[2*i], inner_deltas[2*i+1]
                    verts = _get_hex_vertices(bx + dx, by + dy, 0.0)
                    all_verts_actual.extend(verts)
                
                # Find exact min enclosing hex
                res = _find_min_enclosing_hex_side_lp(all_verts_actual, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_result[0]:
                        new_deltas = list(inner_deltas)
                        best_result = (side, (cx, cy), outer_angle_deg, new_deltas)
        except Exception:
            pass
    
    # Final verification and return
    final_side = best_result[0]
    final_center = best_result[1]
    final_angle = best_result[2]
    final_deltas = best_result[3]
    
    # Build inner hexagon list
    inner_hexes = []
    for i in range(N):
        bx, by = _base_positions[i]
        dx = final_deltas[2*i] if 2*i < len(final_deltas) else 0.0
        dy = final_deltas[2*i+1] if 2*i+1 < len(final_deltas) else 0.0
        inner_hexes.append([bx + dx, by + dy, 0.0])
    
    return inner_hexes, final_center, final_side, final_angle

"""Place 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12
TIME_LIMIT = 145.0  # Safety margin below 150s

def _hex_support(cx, cy, angle_deg, nx, ny, side=1.0):
    """Max of dot product of hex vertices with direction (nx,ny), offset by center."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    max_dot = 0.0
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        rx = ca * x - sa * y
        ry = sa * x + ca * y
        d = nx * rx + ny * ry
        if d > max_dot:
            max_dot = d
    return max_dot + nx * cx + ny * cy

def _hex_min_support(cx, cy, angle_deg, nx, ny, side=1.0):
    """Min of dot product of hex vertices with direction (nx,ny), offset by center."""
    return -_hex_support(cx, cy, angle_deg, -nx, -ny, side)

def _outer_hex_side_from_params(cx_arr, cy_arr, angle_arr, ocx, ocy, oangle_deg):
    """Compute minimum side length of outer hexagon enclosing all inner hexagons."""
    apothem_unit = math.sqrt(3) / 2.0
    oa = math.radians(oangle_deg)
    max_need = 0.0
    for k in range(6):
        na = oa + math.radians(90 + 60 * k)
        nx, ny = math.cos(na), math.sin(na)
        for i in range(N):
            s = _hex_support(cx_arr[i], cy_arr[i], angle_arr[i], nx, ny)
            val = (s - nx * ocx - ny * ocy) / apothem_unit
            if val > max_need:
                max_need = val
    return max_need

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    """Check that all pairs of hexagons are disjoint using SAT."""
    n = len(cx_arr)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq > (2 * math.sqrt(3)) ** 2:
                continue
            if dist_sq < 4.0 - 1e-9:
                return False
            # SAT test: check all 12 edge normals (6 from each hex)
            for ai in range(6):
                aia = math.radians(angle_arr[i] + 90 + 60 * ai)
                anx, any_ = math.cos(aia), math.sin(aia)
                minA = _hex_min_support(cx_arr[i], cy_arr[i], angle_arr[i], anx, any_)
                maxA = _hex_support(cx_arr[i], cy_arr[i], angle_arr[i], anx, any_)
                minB = _hex_min_support(cx_arr[j], cy_arr[j], angle_arr[j], anx, any_)
                maxB = _hex_support(cx_arr[j], cy_arr[j], angle_arr[j], anx, any_)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    break
            else:
                for aj in range(6):
                    aja = math.radians(angle_arr[j] + 90 + 60 * aj)
                    anx, any_ = math.cos(aja), math.sin(aja)
                    minA = _hex_min_support(cx_arr[i], cy_arr[i], angle_arr[i], anx, any_)
                    maxA = _hex_support(cx_arr[i], cy_arr[i], angle_arr[i], anx, any_)
                    minB = _hex_min_support(cx_arr[j], cy_arr[j], angle_arr[j], anx, any_)
                    maxB = _hex_support(cx_arr[j], cy_arr[j], angle_arr[j], anx, any_)
                    if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                        break
                else:
                    return False
    return True

def _smooth_collision_penalty(cx, cy, ang, sigma=0.15):
    """
    Smooth penalty using oriented-circle approximation (inradius-based).
    For each pair, compute the 'gap' between the two inscribed circles along
    the center-to-center direction, plus a rotation-aware correction.
    Returns a smooth scalar penalty.
    """
    penalty = 0.0
    r_in = math.sqrt(3) / 2.0  # inradius of unit hexagon
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = math.hypot(dx, dy)
            if dist < 1e-12:
                penalty += 1e6
                continue
            nx, ny = dx / dist, dy / dist
            # Support of hex i in direction (nx, ny) relative to its center
            sup_i = _hex_support(cx[i], cy[i], ang[i], nx, ny) - (nx * cx[i] + ny * cy[i])
            # Support of hex j in direction (-nx, -ny) relative to its center
            sup_j = _hex_support(cx[j], cy[j], ang[j], -nx, -ny) - (-nx * cx[j] - ny * cy[j])
            # The actual gap between the two hexagons along this axis
            gap = dist - sup_i - sup_j
            # Smooth penalty: penalize when gap < margin
            margin = 0.05
            if gap < margin + 2.0 * sigma:
                # Use a smooth max-like function
                z = (margin - gap) / sigma
                if z > 0:
                    penalty += sigma * (z + math.log1p(math.exp(-z)) - math.log(2))
    return penalty

def _objective_de(params):
    """
    Objective for differential evolution: outer side length + smooth collision penalty.
    params: flattened [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11]
    """
    cx = params[0::3]
    cy = params[1::3]
    ang = params[2::3]
    
    side = _outer_hex_side_from_params(cx, cy, ang, 0.0, 0.0, 0.0)
    penalty = _smooth_collision_penalty(cx, cy, ang)
    
    return side + 50.0 * penalty

def _objective_lbfgs(params):
    """
    Objective for L-BFGS-B: outer side length + smooth collision penalty.
    Includes outer hex center and angle in the parameter vector.
    params: [x0,y0,a0, ..., x11,y11,a11, ocx, ocy, oangle]
    """
    cx = params[0::3][:N]
    cy = params[1::3][:N]
    ang = params[2::3][:N]
    ocx = params[3*N]
    ocy = params[3*N+1]
    oangle = params[3*N+2]
    
    side = _outer_hex_side_from_params(cx, cy, ang, ocx, ocy, oangle)
    penalty = _smooth_collision_penalty(cx, cy, ang)
    
    return side + 80.0 * penalty

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Generate diverse initial configurations
    configs = []
    
    # Flower pattern: center + 6 around + 5 more
    pts = [[0.0, 0.0, 0.0]]
    r1 = 2 * math.sqrt(3)
    for i in range(6):
        angle = math.radians(60 * i)
        pts.append([r1 * math.cos(angle), r1 * math.sin(angle), 0.0])
    r2 = r1 * 1.732
    for deg in [30, 90, 150, 210, 270]:
        rad = math.radians(deg)
        pts.append([r2 * math.cos(rad), r2 * math.sin(rad), 0.0])
    configs.append(pts[:12])
    
    # Hexagonal lattice
    pts = []
    rh = math.sqrt(3)
    cw = 2.0
    for c in range(3):
        pts.append([c * cw - cw, -rh, 0.0])
    for c in range(4):
        pts.append([c * cw - cw - cw / 2, 0.0, 0.0])
    for c in range(3):
        pts.append([c * cw - cw, rh, 0.0])
    pts.append([-cw * 2 - cw / 2 - cw, 0.0, 0.0])
    pts.append([cw * 2 - cw / 2 + cw, 0.0, 0.0])
    configs.append(pts)
    
    # Ring pattern
    pts = [[0.0, 0.0, 0.0]]
    r = 2 * math.sqrt(3)
    for i in range(11):
        angle = math.radians(360 * i / 11 + 15)
        pts.append([r * math.cos(angle), r * math.sin(angle), 0.0])
    configs.append(pts)
    
    # Compact hex arrangement
    pts = [[0.0, 0.0, 0.0]]
    r = 2 * math.sqrt(3)
    for i in range(6):
        angle = math.radians(60 * i)
        pts.append([r * math.cos(angle), r * math.sin(angle), 0.0])
    r2 = r * 1.8
    for i in range(5):
        angle = math.radians(60 * i + 30)
        pts.append([r2 * math.cos(angle), r2 * math.sin(angle), 0.0])
    configs.append(pts)
    
    # Random configurations
    for _ in range(4):
        pts = []
        for _ in range(N):
            pts.append([random.uniform(-4, 4), random.uniform(-4, 4), random.uniform(0, 60)])
        configs.append(pts)
    
    # Check initial configs for validity
    for cfg in configs:
        if time.time() - t0 > TIME_LIMIT - 60:
            break
        if len(cfg) != N:
            continue
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid_fast(cx_arr, cy_arr, angle_arr):
            for oangle in [0, 30, 60, 90, 120, 150]:
                side = _outer_hex_side_from_params(cx_arr, cy_arr, angle_arr, 0.0, 0.0, float(oangle))
                if side < best_side:
                    best_side = side
                    best_cfg = [row[:] for row in cfg]
                    best_center = (0.0, 0.0)
                    best_angle = float(oangle)
    
    # If no valid config found, use first one as starting point
    if best_cfg is None:
        best_cfg = configs[0]
        cx_arr = np.array([p[0] for p in best_cfg])
        cy_arr = np.array([p[1] for p in best_cfg])
        angle_arr = np.array([p[2] for p in best_cfg])
        best_side = _outer_hex_side_from_params(cx_arr, cy_arr, angle_arr, 0.0, 0.0, 0.0)
        best_center = (0.0, 0.0)
        best_angle = 0.0
    
    # Phase 1: Differential Evolution for global exploration
    if time.time() - t0 < TIME_LIMIT - 30:
        bounds = [(-5, 5), (-5, 5), (0, 60)] * N
        
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3*i] = best_cfg[i][0]
            x0[3*i+1] = best_cfg[i][1]
            x0[3*i+2] = best_cfg[i][2]
        
        try:
            result = differential_evolution(
                _objective_de,
                bounds,
                seed=42,
                maxiter=40,
                popsize=12,
                tol=1e-5,
                polish=False,
                init='sobol',
                updating='deferred'
            )
            
            if result.fun < best_side + 1.0:  # Allow some slack for penalty
                x_opt = result.x
                cx_opt = x_opt[0::3]
                cy_opt = x_opt[1::3]
                ang_opt = x_opt[2::3]
                
                if _is_valid_fast(cx_opt, cy_opt, ang_opt):
                    for oangle in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]:
                        s = _outer_hex_side_from_params(cx_opt, cy_opt, ang_opt, 0.0, 0.0, float(oangle))
                        if s < best_side:
                            best_side = s
                            best_cfg = [[float(cx_opt[i]), float(cy_opt[i]), float(ang_opt[i])] for i in range(N)]
                            best_center = (0.0, 0.0)
                            best_angle = float(oangle)
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B local refinement with smooth SDF penalty
    # This is the key structural addition from the inspiration
    if time.time() - t0 < TIME_LIMIT - 15 and best_cfg is not None:
        # Build initial parameter vector for L-BFGS-B
        v0 = np.zeros(3 * N + 3)
        for i in range(N):
            v0[3*i] = best_cfg[i][0]
            v0[3*i+1] = best_cfg[i][1]
            v0[3*i+2] = best_cfg[i][2]
        v0[3*N] = best_center[0]
        v0[3*N+1] = best_center[1]
        v0[3*N+2] = best_angle
        
        # Try multiple starting points for L-BFGS-B
        start_points = [v0]
        # Add perturbed versions
        for _ in range(3):
            v_pert = v0.copy()
            for i in range(3 * N):
                v_pert[i] += random.gauss(0, 0.3)
            v_pert[2::3][:N] = np.mod(v_pert[2::3][:N], 60.0)
            start_points.append(v_pert)
        
        for v_start in start_points:
            if time.time() - t0 > TIME_LIMIT - 10:
                break
            
            try:
                res = minimize(
                    _objective_lbfgs,
                    v_start,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.fun < best_side + 0.5:
                    v_opt = res.x
                    cx_opt = v_opt[0::3][:N]
                    cy_opt = v_opt[1::3][:N]
                    ang_opt = v_opt[2::3][:N]
                    ocx_opt = v_opt[3*N]
                    ocy_opt = v_opt[3*N+1]
                    oangle_opt = v_opt[3*N+2]
                    
                    # Validate
                    if _is_valid_fast(cx_opt, cy_opt, ang_opt):
                        # Compute exact side with optimized center and angle
                        s = _outer_hex_side_from_params(cx_opt, cy_opt, ang_opt, ocx_opt, ocy_opt, oangle_opt)
                        
                        # Also try a grid of nearby angles/centers for final polish
                        best_local = s
                        best_local_c = (ocx_opt, ocy_opt)
                        best_local_a = oangle_opt
                        for oa in [oangle_opt - 15, oangle_opt - 7.5, oangle_opt, oangle_opt + 7.5, oangle_opt + 15,
                                   0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]:
                            for ocx in [ocx_opt - 0.5, ocx_opt - 0.25, ocx_opt, ocx_opt + 0.25, ocx_opt + 0.5,
                                       0, -0.5, 0.5, -1.0, 1.0]:
                                for ocy in [ocy_opt - 0.5, ocy_opt - 0.25, ocy_opt, ocy_opt + 0.25, ocy_opt + 0.5,
                                           0, -0.5, 0.5, -1.0, 1.0]:
                                    ss = _outer_hex_side_from_params(cx_opt, cy_opt, ang_opt, ocx, ocy, float(oa))
                                    if ss < best_local:
                                        best_local = ss
                                        best_local_c = (ocx, ocy)
                                        best_local_a = float(oa)
                        
                        if best_local < best_side:
                            best_side = best_local
                            best_cfg = [[float(cx_opt[i]), float(cy_opt[i]), float(ang_opt[i])] for i in range(N)]
                            best_center = best_local_c
                            best_angle = best_local_a
            except Exception:
                pass
    
    # Phase 3: Final tight polish - try to nudge each hexagon slightly
    if time.time() - t0 < TIME_LIMIT - 5 and best_cfg is not None:
        cx_cur = np.array([best_cfg[i][0] for i in range(N)])
        cy_cur = np.array([best_cfg[i][1] for i in range(N)])
        ang_cur = np.array([best_cfg[i][2] for i in range(N)])
        
        # Try small perturbations
        for trial in range(20):
            if time.time() - t0 > TIME_LIMIT - 3:
                break
            cx_try = cx_cur.copy()
            cy_try = cy_cur.copy()
            ang_try = ang_cur.copy()
            for i in range(N):
                cx_try[i] += random.gauss(0, 0.05)
                cy_try[i] += random.gauss(0, 0.05)
                ang_try[i] = (ang_try[i] + random.gauss(0, 2.0)) % 60.0
            
            if _is_valid_fast(cx_try, cy_try, ang_try):
                for oa in [best_angle - 5, best_angle, best_angle + 5, 0, 30, 60, 90, 120, 150]:
                    s = _outer_hex_side_from_params(cx_try, cy_try, ang_try, best_center[0], best_center[1], float(oa))
                    if s < best_side:
                        best_side = s
                        best_cfg = [[float(cx_try[i]), float(cy_try[i]), float(ang_try[i])] for i in range(N)]
                        best_angle = float(oa)
    
    # Final validation
    if best_cfg is None:
        best_cfg = configs[0]
        best_side = _outer_hex_side_from_params(
            np.array([p[0] for p in best_cfg]),
            np.array([p[1] for p in best_cfg]),
            np.array([p[2] for p in best_cfg]),
            0.0, 0.0, 0.0
        )
        best_center = (0.0, 0.0)
        best_angle = 0.0
    
    return best_cfg, best_center, best_side, best_angle

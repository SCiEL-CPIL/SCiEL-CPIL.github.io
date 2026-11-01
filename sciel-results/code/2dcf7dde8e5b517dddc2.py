import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
SQRT3 = math.sqrt(3)
START_TIME = time.time()
TIME_LIMIT = 145.0
SAFETY_MARGIN = 5.0

def _smooth_max(vals, beta=50.0):
    """Smooth approximation of max(vals) using log-sum-exp."""
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return 0.0
    max_val = np.max(vals)
    if max_val < -100:
        return max_val
    exp_vals = np.exp(beta * (vals - max_val))
    return max_val + math.log(np.sum(exp_vals)) / beta

def _hex_vertices(cx, cy, angle, scale=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    verts = []
    for k in range(6):
        a = angle + k * math.pi / 3.0
        verts.append((cx + scale * math.cos(a), cy + scale * math.sin(a)))
    return verts

def _sat_overlap(v1, v2, normals):
    """Check if two convex polygons overlap using SAT."""
    for nx, ny in normals:
        min1, max1 = float('inf'), float('-inf')
        min2, max2 = float('inf'), float('-inf')
        for px, py in v1:
            d = px * nx + py * ny
            min1 = min(min1, d)
            max1 = max(max1, d)
        for px, py in v2:
            d = px * nx + py * ny
            min2 = min(min2, d)
            max2 = max(max2, d)
        if max1 < min2 or max2 < min1:
            return False
    return True

def _check_disjoint(positions, angles, scale=1.0):
    """Check that all hexagons are disjoint (allowing touching)."""
    n = len(positions)
    verts = []
    for i in range(n):
        cx, cy = positions[i]
        ang = angles[i]
        verts.append(_hex_vertices(cx, cy, ang, scale))
    normals = set()
    for v in verts:
        for k in range(6):
            x1, y1 = v[k]
            x2, y2 = v[(k + 1) % 6]
            ex, ey = x2 - x1, y2 - y1
            length = math.sqrt(ex * ex + ey * ey)
            if length > 1e-12:
                normals.add((ey / length, -ex / length))
    normals = list(normals)
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_overlap(verts[i], verts[j], normals):
                return False
    return True

def _compute_required_side_exact(inner_pts, inner_angles, outer_center, outer_angle_rad, inner_scale=1.0):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner_pts: list of (x, y)
    inner_angles: list of angles in radians
    outer_center: (x, y)
    outer_angle_rad: rotation of the outer hexagon in radians
    inner_scale: side length of inner hexagons
    """
    ox, oy = outer_center
    
    # Outer hexagon normals (pointing inward)
    # For a hexagon with flat-top orientation, normals are at 30, 90, 150, 210, 270, 330 degrees
    # With rotation outer_angle_rad, normals rotate by outer_angle_rad
    normals = []
    for k in range(6):
        theta = math.pi / 6.0 + k * math.pi / 3.0 + outer_angle_rad
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    for i in range(len(inner_pts)):
        ix, iy = inner_pts[i]
        iangle = inner_angles[i]
        verts = _hex_vertices(ix, iy, iangle, inner_scale)
        for vx, vy in verts:
            dx = vx - ox
            dy = vy - oy
            for nx, ny in normals:
                proj = dx * nx + dy * ny
                if proj > max_projection:
                    max_projection = proj
    
    # The distance from center to each edge is apothem_outer = S * APOTHEM
    # We need: max_projection <= apothem_outer = S * APOTHEM
    # So: S >= max_projection / APOTHEM
    # But we also need the hexagon to contain all points, so S >= max_projection / APOTHEM
    # Actually, for a regular hexagon, the distance from center to a vertex is S, and to an edge midpoint is S*APOTHEM
    # The constraint is that all inner vertices must be inside the outer hexagon
    # For a point to be inside, its projection onto each normal must be <= apothem
    # So S * APOTHEM >= max_projection, hence S >= max_projection / APOTHEM
    
    if max_projection <= 0:
        return 0.0
    return max_projection / APOTHEM

def _compute_required_side_smooth(inner_pts, inner_angles, outer_center, outer_angle_rad, inner_scale=1.0, beta=100.0):
    """
    Smooth approximation of the required outer side length using log-sum-exp.
    """
    ox, oy = outer_center
    
    normals = []
    for k in range(6):
        theta = math.pi / 6.0 + k * math.pi / 3.0 + outer_angle_rad
        normals.append((math.cos(theta), math.sin(theta)))
    
    projections = []
    for i in range(len(inner_pts)):
        ix, iy = inner_pts[i]
        iangle = inner_angles[i]
        verts = _hex_vertices(ix, iy, iangle, inner_scale)
        for vx, vy in verts:
            dx = vx - ox
            dy = vy - oy
            for nx, ny in normals:
                proj = dx * nx + dy * ny
                projections.append(proj)
    
    if not projections:
        return 0.0
    
    max_proj = max(projections)
    if max_proj < -1000:
        return 0.0
    
    scaled = [beta * (p - max_proj) for p in projections]
    sum_exp = sum(math.exp(s) for s in scaled)
    smooth_max = max_proj + math.log(sum_exp) / beta
    
    if smooth_max <= 0:
        return 0.0
    return smooth_max / APOTHEM

def _overlap_penalty(inner_pts, inner_angles, inner_scale=1.0, weight=100.0):
    """
    Compute a penalty for overlapping inner hexagons.
    Uses distance between centers as a proxy; if centers are too close, apply penalty.
    For two unit hexagons, they overlap if center distance < ~1.0 (actually < sqrt(3) for some orientations)
    The minimum center distance for non-overlap depends on orientation.
    For aligned hexagons, min distance is 1.0 (they touch at edges)
    For worst case, it can be as low as ~0.866 (sqrt(3)/2) for certain rotations
    We'll use a soft penalty based on center distance.
    """
    n = len(inner_pts)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = inner_pts[i][0] - inner_pts[j][0]
            dy = inner_pts[i][1] - inner_pts[j][1]
            dist = math.sqrt(dx * dx + dy * dy)
            # For unit hexagons, the minimum center distance to guarantee non-overlap is 1.0
            # (when aligned, they touch at flat edges)
            # Use a threshold of 1.0 for the penalty
            threshold = 1.0
            if dist < threshold:
                penalty += weight * (threshold - dist) ** 2
    return penalty

def _lp_solve_container(pts, container_angle):
    """Solve LP to find optimal container center and D for given container angle."""
    normals = []
    for k in range(6):
        a = container_angle + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(a), math.sin(a)))
    c = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    for nx, ny in normals:
        for px, py in pts:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0, None)]
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            cx, cy, D = res.x
            S = (D + APOTHEM) / APOTHEM
            return cx, cy, S
    except Exception:
        pass
    return None, None, float('inf')

def _generate_initial_configs():
    """Generate diverse initial configurations for 11 hexagons."""
    configs = []
    
    # Config 1: Compact flower - center + 6 ring + 4 outer
    axial1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    configs.append(axial1)
    
    # Config 2: More spread
    axial2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 2), (2, -2)
    ]
    configs.append(axial2)
    
    # Config 3: Linear with branches
    axial3 = [
        (0, 0),
        (1, 0), (2, 0), (3, 0), (4, 0),
        (-1, 0), (-2, 0), (-3, 0), (-4, 0),
        (0, 1), (0, -1)
    ]
    configs.append(axial3)
    
    # Config 4: Diamond shape
    axial4 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (-2, 1), (1, 1), (-1, -1)
    ]
    configs.append(axial4)
    
    return configs

def _axial_to_cartesian(q, r, scale=1.0):
    """Convert axial hex coordinates to cartesian."""
    x = SQRT3 * (q + r / 2.0) * scale
    y = 1.5 * r * scale
    return (x, y)

def construct():
    # Check if we are already close to the time limit
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    best_result = None
    best_S = float('inf')
    
    configs = _generate_initial_configs()
    
    for config_idx, axial in enumerate(configs):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Use a small expansion to ensure strict disjointness
        SCALE = 1.0 + 1e-5
        
        pts = []
        for q, r in axial:
            x, y = _axial_to_cartesian(q, r, SCALE)
            pts.append((x, y))
        
        pts = np.array(pts)
        n_pts = len(pts)
        
        # Initial angles: all 0 (aligned)
        angles = np.zeros(n_pts)
        
        # Phase 1: Optimize container angle and center using LP for fixed inner positions
        def solve_for_beta(beta):
            """Solve LP for container given beta (container angle)."""
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return None, None, float('inf')
            
            # Get all inner hexagon vertices
            all_verts = []
            for i in range(n_pts):
                cx, cy = pts[i]
                ang = angles[i]
                verts = _hex_vertices(cx, cy, ang, 1.0)
                all_verts.extend(verts)
            
            normals = []
            for k in range(6):
                ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            
            c = np.array([0.0, 0.0, 1.0])
            A_ub = []
            b_ub = []
            
            for nx, ny in normals:
                for px, py in all_verts:
                    A_ub.append([-nx, -ny, -1.0])
                    b_ub.append(-(px * nx + py * ny))
                    A_ub.append([nx, ny, -1.0])
                    b_ub.append(px * nx + py * ny)
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            bounds = [(None, None), (None, None), (0, None)]
            
            try:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.status == 0:
                    cx = res.x[0]
                    cy = res.x[1]
                    D = res.x[2]
                    S = (D + APOTHEM) / APOTHEM
                    return cx, cy, S
                else:
                    return None, None, float('inf')
            except Exception:
                return None, None, float('inf')

        def obj_beta(beta):
            _, _, S = solve_for_beta(beta)
            return S

        # Find best container angle
        best_beta = 0.0
        best_cx, best_cy, best_S_local = solve_for_beta(best_beta)
        
        if best_S_local == float('inf'):
            continue
        
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(obj_beta, bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-8})
                if res.fun < best_S_local:
                    best_beta = res.x
                    best_cx, best_cy, best_S_local = solve_for_beta(best_beta)
                    
                    # Refine
                    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                        local_bounds = (max(0, best_beta - 0.02), min(math.pi / 6, best_beta + 0.02))
                        res2 = minimize_scalar(obj_beta, bounds=local_bounds, method='bounded',
                                              options={'maxiter': 300, 'xatol': 1e-10})
                        if res2.fun < best_S_local:
                            best_beta = res2.x
                            best_cx, best_cy, best_S_local = solve_for_beta(best_beta)
        except Exception:
            pass

        # Phase 2: Smooth optimization of inner positions, angles, and container
        # Parameters: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, ox, oy, outer_angle]
        def smooth_objective(params):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return float('inf')
            
            inner_pts_opt = []
            inner_angles_opt = []
            for i in range(n_pts):
                px = params[3*i]
                py = params[3*i+1]
                pa = params[3*i+2]
                inner_pts_opt.append((px, py))
                inner_angles_opt.append(pa)
            
            ox = params[3*n_pts]
            oy = params[3*n_pts+1]
            outer_angle = params[3*n_pts+2]
            
            # Compute smooth required side
            S_smooth = _compute_required_side_smooth(inner_pts_opt, inner_angles_opt, 
                                                      (ox, oy), outer_angle, 1.0, beta=80.0)
            
            # Overlap penalty
            pen = _overlap_penalty(inner_pts_opt, inner_angles_opt, 1.0, weight=50.0)
            
            return S_smooth + pen
        
        # Initial guess from LP solution
        x0 = np.zeros(3 * n_pts + 3)
        for i in range(n_pts):
            x0[3*i] = pts[i][0]
            x0[3*i+1] = pts[i][1]
            x0[3*i+2] = 0.0  # angle
        x0[3*n_pts] = best_cx
        x0[3*n_pts+1] = best_cy
        x0[3*n_pts+2] = best_beta
        
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res_smooth = minimize(smooth_objective, x0, method='L-BFGS-B',
                                     options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10,
                                              'maxls': 50})
                
                if res_smooth.fun < best_S_local:
                    # Extract optimized positions
                    opt_pts = []
                    opt_angles = []
                    for i in range(n_pts):
                        opt_pts.append((res_smooth.x[3*i], res_smooth.x[3*i+1]))
                        opt_angles.append(res_smooth.x[3*i+2])
                    
                    opt_ox = res_smooth.x[3*n_pts]
                    opt_oy = res_smooth.x[3*n_pts+1]
                    opt_outer_angle = res_smooth.x[3*n_pts+2]
                    
                    # Verify with exact computation
                    # First check disjointness
                    if _check_disjoint(opt_pts, opt_angles, scale=1.0):
                        # Compute exact required side
                        all_verts = []
                        for i in range(n_pts):
                            cx, cy = opt_pts[i]
                            ang = opt_angles[i]
                            verts = _hex_vertices(cx, cy, ang, 1.0)
                            all_verts.extend(verts)
                        
                        # Use LP to find optimal container for these positions
                        cx_test, cy_test, S_test = _lp_solve_container(
                            [(vx, vy) for vx, vy in all_verts], opt_outer_angle)
                        
                        if S_test < best_S_local:
                            best_cx, best_cy, best_S_local = cx_test, cy_test, S_test
                            best_beta = opt_outer_angle
                            pts = np.array(opt_pts)
                            angles = np.array(opt_angles)
        except Exception:
            pass

        # Phase 3: Final high-precision LP polish on container angle
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                # Re-solve with current positions
                all_verts = []
                for i in range(n_pts):
                    cx, cy = pts[i]
                    ang = angles[i]
                    verts = _hex_vertices(cx, cy, ang, 1.0)
                    all_verts.extend(verts)
                
                def final_obj_beta(beta):
                    _, _, S = _lp_solve_container(all_verts, beta)
                    return S
                
                local_bounds = (max(0, best_beta - 0.005), min(math.pi / 6, best_beta + 0.005))
                res3 = minimize_scalar(final_obj_beta, bounds=local_bounds, method='bounded',
                                      options={'maxiter': 1000, 'xatol': 1e-12})
                if res3.fun < best_S_local:
                    best_beta = res3.x
                    best_cx, best_cy, best_S_local = _lp_solve_container(all_verts, best_beta)
                
                # Even tighter
                if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                    local_bounds2 = (max(0, best_beta - 0.001), min(math.pi / 6, best_beta + 0.001))
                    res4 = minimize_scalar(final_obj_beta, bounds=local_bounds2, method='bounded',
                                          options={'maxiter': 2000, 'xatol': 1e-14})
                    if res4.fun < best_S_local:
                        best_beta = res4.x
                        best_cx, best_cy, best_S_local = _lp_solve_container(all_verts, best_beta)
        except Exception:
            pass

        # Update global best
        if best_S_local < best_S:
            best_S = best_S_local
            best_result = (best_cx, best_cy, best_beta, pts.copy(), angles.copy())
    
    # If no valid result found, return fallback
    if best_result is None:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    best_cx, best_cy, best_beta, best_pts, best_angles = best_result
    
    # Prepare output
    inner = []
    angle_deg = math.degrees(best_beta)
    for i in range(len(best_pts)):
        x, y = best_pts[i]
        ang_deg = math.degrees(best_angles[i])
        inner.append([float(x), float(y), float(ang_deg)])
        
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = angle_deg
    
    return inner, center, side, rotation

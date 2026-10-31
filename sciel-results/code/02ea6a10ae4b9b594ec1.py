import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0
DEADLINE = 135.0  # Safety margin before 150s kill

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    max_projection = 0.0

    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj

    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []

    # Get axes from edges of v1
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        # Normal to the edge
        axis = (-edge[1], edge[0])
        axes.append(axis)

    # Get axes from edges of v2
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        # Normal to the edge
        axis = (-edge[1], edge[0])
        axes.append(axis)

    for ax, ay in axes:
        # Project all vertices of both polygons onto this axis
        min1 = min(v[0] * ax + v[1] * ay for v in v1)
        max1 = max(v[0] * ax + v[1] * ay for v in v1)
        min2 = min(v[0] * ax + v[1] * ay for v in v2)
        max2 = max(v[0] * ax + v[1] * ay for v in v2)

        # Check for separation
        if max1 < min2 - eps or max2 < min1 - eps:
            return False

    return True

def check_all_disjoint(inner, eps=1e-9):
    """Check if all inner hexagons are disjoint."""
    n = len(inner)
    for i in range(n):
        verts_i = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            verts_j = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(verts_i, verts_j, eps):
                return False
    return True

def check_inside_outer(inner, outer_center, outer_side, outer_angle_deg, eps=1e-9):
    """Check if all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    max_allowed_proj = outer_side * SQRT3 / 2.0

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_allowed_proj + eps:
                    return False
    return True

def construct():
    start_time = time.time()
    
    # Candidate layouts (axial coordinates)
    # We use a variety of topologies to escape local minima.
    candidates = []
    
    # 1. Compact hexagonal cluster (1+6+4)
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    candidates.append(axial_1)

    # 2. Another compact variation
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_2)
    
    # 3. Rectangular-ish
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)
    
    # 4. Linear-ish
    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)
    
    # 5. Another compact
    axial_5 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_5)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            # Standard hex grid to cartesian
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        """
        Exact minimum side length S for a fixed outer rotation beta.
        Uses LP to find optimal center and minimal apothem D.
        """
        if time.time() - start_time > DEADLINE - 5:
            return float('inf'), None, None
        
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        # For each normal and each point, the projection of (p - c) onto n must be <= D
        # (p - c) . n <= D  =>  p.n - c.n <= D
        # p.n - D - c.n <= 0
        # -c.n - D + p.n <= 0  (rearranged for linprog form A_ub x <= b_ub)
        # Let x = [cx, cy, D]
        # -nx * cx - ny * cy - D + (px * nx + py * ny) <= 0
        
        for nx, ny in normals:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                # Also consider the opposite side? 
                # The hexagon is defined by |proj| <= D for 3 axes.
                # proj = (p-c).n
                # -D <= (p-c).n <= D
                # (p-c).n <= D  => -nx*cx - ny*cy - D <= -p.n
                # -(p-c).n <= D => nx*cx + ny*cy - D <= p.n
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.1):
        """
        Smooth approximation of S using log-sum-exp for the max projection.
        """
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        projs = []
        for nx, ny in normals:
            for px, py in pts:
                proj = (px - cx) * nx + (py - cy) * ny
                projs.append(proj)
                projs.append(-proj)
        
        projs = np.array(projs)
        max_val = np.max(projs)
        if max_val > 100:
            return float('inf'), (cx, cy), float('inf')
        
        # Soft max
        log_sum = np.log(np.sum(np.exp((projs - max_val) / temperature))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy), D

    def optimize_shape(pts):
        """
        Optimize the outer hexagon (beta, cx, cy) for a fixed set of inner points.
        """
        best_S = float('inf')
        best_beta = 0.0
        best_center = (0.0, 0.0)
        
        if time.time() - start_time > DEADLINE - 10:
            return float('inf'), 0.0, (0.0, 0.0)
        
        # 1. Grid search for beta
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # 2. Refine beta with scalar minimization
        try:
            if time.time() - start_time < DEADLINE - 10:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-10})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # 3. Multi-start smooth optimization for (cx, cy, beta)
        # This helps if the LP is too strict or to find better local minima in the smooth landscape
        try:
            if time.time() - start_time < DEADLINE - 15:
                temperatures = [0.5, 0.2, 0.1, 0.05]
                start_betas = [0.0, math.pi/18, math.pi/12, best_beta]
                
                for temp in temperatures:
                    if time.time() - start_time > DEADLINE - 10:
                        break
                    for sb in start_betas:
                        if time.time() - start_time > DEADLINE - 10:
                            break
                            
                        # Initialize center at centroid
                        cx0, cy0 = np.mean(pts, axis=0)
                        
                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            # Wrap beta to [0, pi/6)
                            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S, _, _ = smooth_calc_S(pts, beta, cx, cy, t)
                            return S
                        
                        try:
                            res_smooth = minimize(
                                smooth_obj,
                                [cx0, cy0, sb],
                                method='Nelder-Mead',
                                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 500}
                            )
                            if res_smooth.fun < best_S:
                                # Verify with exact LP
                                new_cx, new_cy, new_beta = res_smooth.x
                                new_beta = ((new_beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                                S_exact, center_exact, _ = calc_S_for_beta(pts, new_beta)
                                if S_exact < best_S:
                                    best_S = S_exact
                                    best_beta = new_beta
                                    best_center = center_exact
                        except Exception:
                            pass
        except Exception:
            pass

        return best_S, best_beta, best_center

    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Iterate through candidate topologies
    for axial_list in candidates:
        if time.time() - start_time > DEADLINE - 20:
            break
            
        pts = axial_to_cartesian(axial_list)
        
        # Optimize outer hexagon for this topology
        S, beta_rad, center = optimize_shape(pts)
        
        if S < best_side:
            best_side = S
            best_outer_angle = math.degrees(beta_rad)
            best_outer_center = center
            
            # Construct inner list
            new_inner = []
            for i in range(len(pts)):
                new_inner.append([float(pts[i][0]), float(pts[i][1]), 0.0])
            best_inner = new_inner

    # If no valid topology was found (should not happen), fallback
    if best_inner is None:
        # Fallback to the first candidate
        pts = axial_to_cartesian(candidates[0])
        S, beta_rad, center = optimize_shape(pts)
        best_side = S
        best_outer_angle = math.degrees(beta_rad)
        best_outer_center = center
        best_inner = [[float(pts[i][0]), float(pts[i][1]), 0.0] for i in range(len(pts))]

    # Phase 2: Joint Optimization with Penalty
    # Optimize positions of inner hexagons and outer parameters simultaneously.
    # This can squeeze the packing tighter than the lattice-based initial guess.
    
    def objective_joint(params):
        if time.time() - start_time > DEADLINE - 5:
            return 1e10
            
        # Extract inner positions
        inner_list = []
        for i in range(N):
            x = params[i * 2]
            y = params[i * 2 + 1]
            inner_list.append([x, y, 0.0])

        # Extract outer params
        cx = params[2 * N]
        cy = params[2 * N + 1]
        angle = params[2 * N + 2]

        side = compute_required_side(inner_list, (cx, cy), angle)
        
        # Penalty for overlaps: 
        # For two unit hexagons to be disjoint, their centers must be at least 
        # sqrt(3) apart if aligned.
        penalty = 0.0
        min_dist = SQRT3 - 0.01 
        for i in range(N):
            xi, yi, _ = inner_list[i]
            for j in range(i + 1, N):
                xj, yj, _ = inner_list[j]
                dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist < min_dist:
                    # Quadratic penalty
                    penalty += (min_dist - dist) ** 2 * 100.0

        return side + penalty

    # Initial parameters for joint optimization
    init_params_joint = []
    for x, y, _ in best_inner:
        init_params_joint.append(x)
        init_params_joint.append(y)
    init_params_joint.append(best_outer_center[0])
    init_params_joint.append(best_outer_center[1])
    init_params_joint.append(best_outer_angle)

    # Run joint optimization
    try:
        if time.time() - start_time < DEADLINE - 20:
            res_joint = minimize(
                objective_joint,
                init_params_joint,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7, 'maxfev': 2000}
            )
            
            if res_joint.fun < best_side + 1e-4: # Only accept if better or very close
                # Reconstruct inner from result
                new_inner = []
                for i in range(N):
                    new_inner.append([res_joint.x[i*2], res_joint.x[i*2+1], 0.0])
                
                new_cx = res_joint.x[2*N]
                new_cy = res_joint.x[2*N+1]
                new_angle = res_joint.x[2*N+2]
                
                # Check for overlaps explicitly using SAT for rigor
                valid = check_all_disjoint(new_inner)
                
                if valid:
                    new_side = compute_required_side(new_inner, (new_cx, new_cy), new_angle)
                    # Verify inside outer
                    if check_inside_outer(new_inner, (new_cx, new_cy), new_side, new_angle):
                        if new_side < best_side:
                            best_side = new_side
                            best_inner = new_inner
                            best_outer_center = (new_cx, new_cy)
                            best_outer_angle = new_angle
    except Exception:
        pass

    # Phase 3: Polish with tighter penalty
    if time.time() - start_time < DEADLINE - 10:
        def objective_sat(params):
            if time.time() - start_time > DEADLINE - 5:
                return 1e10
                
            inner_list = []
            for i in range(N):
                x = params[i * 2]
                y = params[i * 2 + 1]
                inner_list.append([x, y, 0.0])

            cx = params[2 * N]
            cy = params[2 * N + 1]
            angle = params[2 * N + 2]

            side = compute_required_side(inner_list, (cx, cy), angle)
            
            penalty = 0.0
            min_dist = SQRT3 - 0.005 
            for i in range(N):
                xi, yi, _ = inner_list[i]
                for j in range(i + 1, N):
                    xj, yj, _ = inner_list[j]
                    dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2 * 1000.0

            return side + penalty

        try:
            res_sat = minimize(
                objective_sat,
                [x for pair in best_inner for x in pair[:2]] + [best_outer_center[0], best_outer_center[1], best_outer_angle],
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 1000}
            )
            
            if res_sat.fun < best_side + 1e-5:
                new_inner = []
                for i in range(N):
                    new_inner.append([res_sat.x[i*2], res_sat.x[i*2+1], 0.0])
                
                new_cx = res_sat.x[2*N]
                new_cy = res_sat.x[2*N+1]
                new_angle = res_sat.x[2*N+2]
                
                if check_all_disjoint(new_inner):
                    new_side = compute_required_side(new_inner, (new_cx, new_cy), new_angle)
                    if check_inside_outer(new_inner, (new_cx, new_cy), new_side, new_angle):
                        if new_side < best_side:
                            best_side = new_side
                            best_inner = new_inner
                            best_outer_center = (new_cx, new_cy)
                            best_outer_angle = new_angle
        except Exception:
            pass

    # Final safety check and cleanup
    if best_side == float('inf') or best_inner is None:
        # Fallback to initial
        pts = axial_to_cartesian(candidates[0])
        best_inner = [[float(pts[i][0]), float(pts[i][1]), 0.0] for i in range(len(pts))]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)

    return best_inner, best_outer_center, best_side, best_outer_angle

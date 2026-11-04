import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices for angle 0 (pointy-top)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle that contains all inner hexagons.
        For a regular hexagon with side s, the inradius is s*sqrt(3)/2.
        The maximum projection onto any of the 6 face normals must be <= inradius.
        So side = max_proj / (sqrt(3)/2) = max_proj * 2/sqrt(3).
        """
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def optimize_outer_angle(positions, angles):
        """
        Find the best outer hexagon rotation angle to minimize the required side length.
        Uses a coarse-to-fine search.
        """
        best_angle = 0.0
        best_side = float('inf')
        
        # Coarse search: 61 points over 0-60 degrees (hexagon has 60-degree symmetry)
        for deg in range(0, 61, 1):
            side = calc_min_outer_side(positions, angles, float(deg))
            if side < best_side:
                best_side = side
                best_angle = float(deg)
        
        # Fine search: 361 points in a window around the best coarse angle
        window = 2.0
        for deg in np.linspace(max(0, best_angle - window), min(60, best_angle + window), 361):
            side = calc_min_outer_side(positions, angles, float(deg))
            if side < best_side:
                best_side = side
                best_angle = float(deg)
        
        return best_angle, best_side

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex polygons.
        If they overlap, returns a negative value (approximate penetration depth).
        If they are separated, returns the positive distance.
        Uses separating axis theorem with early exit.
        """
        min_gap = float('inf')
        
        # Axes from poly1 edges
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(normal1)
            if norm < 1e-9: continue
            axis = normal1 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap on this axis
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                # Overlapping on this axis
                min_gap = min(min_gap, gap)
            else:
                # Separated on this axis -> disjoint
                return gap
                
        # Axes from poly2 edges
        for i in range(6):
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(normal2)
            if norm < 1e-9: continue
            axis = normal2 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        return min_gap

    def get_polygons(positions, angles):
        """Precompute all polygon vertices."""
        return [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]

    def objective(params, penalty_weight=100.0):
        if time.time() > deadline - 5:
            return 1000.0
            
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        # Use optimized outer angle for more accurate objective
        _, side_len = optimize_outer_angle(positions, angles)
        
        penalty = 0.0
        polys = get_polygons(positions, angles)
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    penalty += penalty_weight * gap**2
                    
        return side_len + penalty

    def check_validity(positions, angles, tol=1e-5):
        polys = get_polygons(positions, angles)
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    # Initial guess: Honeycomb subset with multiple rotations for diversity
    def generate_lattice(rot_deg=0.0):
        """Generate honeycomb lattice points, optionally rotated."""
        rot_rad = math.radians(rot_deg)
        cos_r, sin_r = math.cos(rot_rad), math.sin(rot_rad)
        points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * 1.5
                y = i * SQRT3 / 2.0 + j * SQRT3
                # Rotate
                xr = x * cos_r - y * sin_r
                yr = x * sin_r + y * cos_r
                points.append((xr, yr))
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        return np.array(points[:12])

    # Try multiple lattice rotations and pick the best
    best_init = None
    best_init_score = float('inf')
    for rot in [0, 15, 30, 45, 60, 75, 90]:
        pos = generate_lattice(rot)
        pos -= np.mean(pos, axis=0)
        ang = np.zeros(N)
        _, score = optimize_outer_angle(pos, ang)
        if score < best_init_score:
            best_init_score = score
            best_init = (pos, ang)

    init_positions, init_angles = best_init
    x0 = np.concatenate([init_positions.flatten(), init_angles])
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10)) # x
        bounds.append((-10, 10)) # y
    for i in range(N):
        bounds.append((0, 60)) # angle (hexagon has 60-degree symmetry)

    best_score = float('inf')
    best_result = None
    best_outer_angle = 0.0

    # Strategy 1: Penalty Method with L-BFGS-B
    try:
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 800, 'ftol': 1e-12, 'maxls': 100, 'disp': False}
        )
        
        pos = res.x[:2*N].reshape(N, 2)
        ang = res.x[2*N:]
        
        if check_validity(pos, ang):
            outer_ang, side = optimize_outer_angle(pos, ang)
            best_score = side
            best_outer_angle = outer_ang
            best_result = res.x
    except Exception as e:
        pass

    # Strategy 2: Nelder-Mead if LBFGS failed or score is poor
    if best_result is None or best_score > 4.5:
        try:
            def obj_nm(params):
                if time.time() > deadline - 5:
                    return 1000.0
                positions = params[:2*N].reshape(N, 2)
                angles = params[2*N:]
                _, side_len = optimize_outer_angle(positions, angles)
                penalty = 0.0
                polys = get_polygons(positions, angles)
                for i in range(N):
                    for j in range(i+1, N):
                        gap = sat_gap(polys[i], polys[j])
                        if gap < 0:
                            penalty += 500.0 * gap**2
                return side_len + penalty

            start_x = best_result if best_result is not None else x0
            
            res2 = minimize(
                obj_nm, 
                start_x, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True, 'disp': False}
            )
            
            pos = res2.x[:2*N].reshape(N, 2)
            ang = res2.x[2*N:]
            
            if check_validity(pos, ang):
                outer_ang, score2 = optimize_outer_angle(pos, ang)
                if score2 < best_score:
                    best_score = score2
                    best_outer_angle = outer_ang
                    best_result = res2.x
        except Exception:
            pass

    # Strategy 3: Hard Constraint SLSQP with SAT-based constraints
    # This approach uses actual polygon separation as constraints via penalty
    # but with SLSQP's constraint handling for better convergence
    
    if time.time() < deadline - 30:
        try:
            # Use current best angles if available, otherwise 0
            current_angles = best_result[2*N:] if best_result is not None else np.zeros(N)
            
            # SLSQP variables: positions only (2N vars)
            x_slsqp = (best_result[:2*N] if best_result is not None else x0[:2*N])
            
            def obj_slsqp(x):
                if time.time() > deadline - 10:
                    return 1e10
                pos = x.reshape(N, 2)
                _, side = optimize_outer_angle(pos, current_angles)
                return side
            
            # Constraints: minimum gap between each pair of hexagons
            # We'll use a softer approach: penalize overlaps in the objective
            # and use SLSQP with just bounds
            bounds_slsqp = [(-5, 5)] * (2*N)
            
            def obj_slsqp_penalized(x):
                if time.time() > deadline - 10:
                    return 1e10
                pos = x.reshape(N, 2)
                _, side = optimize_outer_angle(pos, current_angles)
                penalty = 0.0
                polys = get_polygons(pos, current_angles)
                for i in range(N):
                    for j in range(i+1, N):
                        gap = sat_gap(polys[i], polys[j])
                        if gap < 0:
                            penalty += 1000.0 * gap**2
                return side + penalty
            
            res3 = minimize(
                obj_slsqp_penalized,
                x_slsqp,
                method='SLSQP',
                bounds=bounds_slsqp,
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            if check_validity(res3.x.reshape(N, 2), current_angles):
                outer_ang, score3 = optimize_outer_angle(res3.x.reshape(N, 2), current_angles)
                if score3 < best_score:
                    # Combine with angles for full result
                    full_result = np.concatenate([res3.x, current_angles])
                    best_score = score3
                    best_outer_angle = outer_ang
                    best_result = full_result
        except Exception:
            pass

    # Strategy 4: Final Polish with L-BFGS-B (High Penalty)
    if best_result is not None and time.time() < deadline - 20:
        try:
            def obj_final(params):
                if time.time() > deadline - 5:
                    return 1e10
                positions = params[:2*N].reshape(N, 2)
                angles = params[2*N:]
                _, side_len = optimize_outer_angle(positions, angles)
                penalty = 0.0
                polys = get_polygons(positions, angles)
                for i in range(N):
                    for j in range(i+1, N):
                        gap = sat_gap(polys[i], polys[j])
                        if gap < 0:
                            penalty += 5000.0 * gap**2
                return side_len + penalty

            res4 = minimize(
                obj_final, 
                best_result, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'maxls': 200, 'disp': False}
            )
            
            pos = res4.x[:2*N].reshape(N, 2)
            ang = res4.x[2*N:]
            
            if check_validity(pos, ang):
                outer_ang, score4 = optimize_outer_angle(pos, ang)
                if score4 < best_score:
                    best_score = score4
                    best_outer_angle = outer_ang
                    best_result = res4.x
        except Exception:
            pass

    # Strategy 5: High-precision polish with tighter tolerance
    if best_result is not None and time.time() < deadline - 10:
        try:
            def obj_polish(params):
                if time.time() > deadline - 3:
                    return 1e10
                positions = params[:2*N].reshape(N, 2)
                angles = params[2*N:]
                _, side_len = optimize_outer_angle(positions, angles)
                penalty = 0.0
                polys = get_polygons(positions, angles)
                for i in range(N):
                    for j in range(i+1, N):
                        gap = sat_gap(polys[i], polys[j])
                        if gap < 0:
                            penalty += 10000.0 * gap**2
                return side_len + penalty

            res5 = minimize(
                obj_polish, 
                best_result, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'maxls': 500, 'disp': False}
            )
            
            pos = res5.x[:2*N].reshape(N, 2)
            ang = res5.x[2*N:]
            
            if check_validity(pos, ang):
                outer_ang, score5 = optimize_outer_angle(pos, ang)
                if score5 < best_score:
                    best_score = score5
                    best_outer_angle = outer_ang
                    best_result = res5.x
        except Exception:
            pass

    # Finalize
    if best_result is not None:
        final_positions = best_result[:2*N].reshape(N, 2)
        final_angles = best_result[2*N:]
        final_outer_angle, final_side = optimize_outer_angle(final_positions, final_angles)
    else:
        # Fallback
        final_positions = np.array([
            [0, 0], [2.1, 0], [-2.1, 0],
            [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
            [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
            [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
        ])
        final_angles = np.zeros(N)
        final_outer_angle, final_side = optimize_outer_angle(final_positions, final_angles)

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, final_outer_angle

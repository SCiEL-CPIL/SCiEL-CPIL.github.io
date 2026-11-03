import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices for angle 0 (flat-top)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side_smooth(positions, angles, outer_angle=0.0, soft_beta=50.0):
        """
        Smooth approximation of minimum outer hexagon side length using log-sum-exp.
        """
        theta_rad = math.radians(outer_angle)
        normals = np.array([[math.cos(theta_rad + math.pi/6 + k * math.pi/3),
                             math.sin(theta_rad + math.pi/6 + k * math.pi/3)] for k in range(6)])
        
        max_projs = np.zeros(6)
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T  # shape (6, 6)
            # For each normal direction, find max projection across all vertices
            for k in range(6):
                if projs[k, 0] > max_projs[k]:
                    max_projs[k] = projs[k, 0]
        
        # Use soft-max to get smooth approximation of max over the 6 directions
        scaled = soft_beta * max_projs
        m = np.max(scaled)
        log_sum_exp = m + math.log(np.sum(np.exp(scaled - m)))
        smooth_max = log_sum_exp / soft_beta
        
        return smooth_max * 2.0 / math.sqrt(3)

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex polygons.
        If they overlap, returns a negative value (approximate penetration depth).
        If they are separated, returns the positive distance.
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

    def objective(params, penalty_weight=100.0, soft_beta=50.0):
        if time.time() > deadline - 5:
            return 1000.0
            
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        side_len = calc_min_outer_side_smooth(positions, angles, 0.0, soft_beta)
        
        penalty = 0.0
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    penalty += penalty_weight * gap**2
                    
        return side_len + penalty

    def check_validity(positions, angles, tol=1e-5):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    # Initial guess: Honeycomb subset
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * 1.5
            y = i * SQRT3 / 2.0 + j * SQRT3
            lattice_points.append((x, y))
    
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_positions = np.array(lattice_points[:12])
    init_positions -= np.mean(init_positions, axis=0)
    init_angles = np.zeros(N)
    
    x0 = np.concatenate([init_positions.flatten(), init_angles])
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10)) # x
        bounds.append((-10, 10)) # y
    for i in range(N):
        bounds.append((0, 360)) # angle

    best_score = float('inf')
    best_result = None

    # Strategy 1: Penalty Method with L-BFGS-B using smooth objective
    try:
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'maxls': 50, 'disp': False}
        )
        
        pos = res.x[:2*N].reshape(N, 2)
        ang = res.x[2*N:]
        
        if check_validity(pos, ang):
            best_score = calc_min_outer_side_smooth(pos, ang, 0.0, 100.0)
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
                side_len = calc_min_outer_side_smooth(positions, angles, 0.0, 50.0)
                penalty = 0.0
                polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
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
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True, 'disp': False}
            )
            
            pos = res2.x[:2*N].reshape(N, 2)
            ang = res2.x[2*N:]
            
            if check_validity(pos, ang):
                score2 = calc_min_outer_side_smooth(pos, ang, 0.0, 100.0)
                if score2 < best_score:
                    best_score = score2
                    best_result = res2.x
        except Exception:
            pass

    # Strategy 3: Hard Constraint SLSQP (Inspiration)
    if time.time() < deadline - 30:
        try:
            current_angles = best_result[2*N:] if best_result is not None else np.zeros(N)
            x_slsqp = (best_result[:2*N] if best_result is not None else x0[:2*N])
            
            def obj_slsqp(x):
                if time.time() > deadline - 10:
                    return 1e10
                pos = x.reshape(N, 2)
                return calc_min_outer_side_smooth(pos, current_angles, 0.0, 100.0)
            
            constraints = []
            min_dist_sq = (SQRT3 - 0.001)**2
            for i in range(N):
                for j in range(i+1, N):
                    def make_constraint(i, j):
                        def constraint(x):
                            xi = x[2*i]
                            yi = x[2*i+1]
                            xj = x[2*j]
                            yj = x[2*j+1]
                            return (xi - xj)**2 + (yi - yj)**2 - min_dist_sq
                        return constraint
                    constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
            
            bounds_slsqp = [(-5, 5)] * (2*N)
            
            res3 = minimize(
                obj_slsqp,
                x_slsqp,
                method='SLSQP',
                bounds=bounds_slsqp,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            
            if res3.success or res3.fun < best_score:
                pos = res3.x.reshape(N, 2)
                if check_validity(pos, current_angles):
                    score3 = calc_min_outer_side_smooth(pos, current_angles, 0.0, 100.0)
                    if score3 < best_score:
                        full_result = np.concatenate([pos.flatten(), current_angles])
                        best_score = score3
                        best_result = full_result
        except Exception:
            pass

    # Strategy 4: Final Polish with L-BFGS-B (High Penalty)
    if best_result is not None and time.time() < deadline - 15:
        try:
            def obj_final(params):
                positions = params[:2*N].reshape(N, 2)
                angles = params[2*N:]
                side_len = calc_min_outer_side_smooth(positions, angles, 0.0, 200.0)
                penalty = 0.0
                polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
                for i in range(N):
                    for j in range(i+1, N):
                        gap = sat_gap(polys[i], polys[j])
                        if gap < 0:
                            penalty += 1000.0 * gap**2
                return side_len + penalty

            res4 = minimize(
                obj_final, 
                best_result, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'maxls': 50, 'disp': False}
            )
            
            pos = res4.x[:2*N].reshape(N, 2)
            ang = res4.x[2*N:]
            
            if check_validity(pos, ang):
                score4 = calc_min_outer_side_smooth(pos, ang, 0.0, 200.0)
                if score4 < best_score:
                    best_score = score4
                    best_result = res4.x
        except Exception:
            pass

    # Finalize
    if best_result is not None:
        final_positions = best_result[:2*N].reshape(N, 2)
        final_angles = best_result[2*N:]
        final_side = calc_min_outer_side_smooth(final_positions, final_angles, 0.0, 200.0)
    else:
        final_positions = np.array([
            [0, 0], [2.1, 0], [-2.1, 0],
            [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
            [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
            [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
        ])
        final_angles = np.zeros(N)
        final_side = calc_min_outer_side_smooth(final_positions, final_angles, 0.0, 200.0)

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0


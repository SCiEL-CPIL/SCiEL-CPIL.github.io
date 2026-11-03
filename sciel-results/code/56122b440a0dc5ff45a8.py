import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

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

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle that contains all inner hexagons.
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
                # We return the minimum "overlap" (penetration) found so far
                # Note: This is not the true penetration depth, but a measure of violation
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

    # Objective function for SLSQP
    def objective(params):
        if time.time() > deadline - 5:
            return 1000.0
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        # We minimize the outer side length
        return calc_min_outer_side(positions, angles, 0.0)

    # Constraints: for every pair (i, j), the SAT gap must be >= 0
    # SLSQP constraints are in the form: func(x) >= 0
    def make_constraint(i, j):
        def constraint(params):
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            
            p1 = positions[i]
            p2 = positions[j]
            a1 = angles[i]
            a2 = angles[j]
            
            verts1 = get_hex_verts(p1[0], p1[1], 1.0, a1)
            verts2 = get_hex_verts(p2[0], p2[1], 1.0, a2)
            
            # We require a small positive margin to avoid numerical issues at the boundary
            return sat_gap(verts1, verts2) - 1e-6
        
        return constraint

    # Initial guess: Honeycomb subset
    sqrt3 = math.sqrt(3)
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))
    
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_positions = np.array(lattice_points[:12])
    init_positions -= np.mean(init_positions, axis=0)
    init_angles = np.zeros(N)
    
    x0 = np.concatenate([init_positions.flatten(), init_angles])
    
    # Define constraints for all pairs
    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    # Bounds: keep angles within [0, 360) and positions within a reasonable box
    # The outer hexagon side length for 12 hexes is roughly 4-5, so coordinates within [-10, 10] is safe
    bounds = []
    for i in range(N):
        bounds.append((-10, 10)) # x
        bounds.append((-10, 10)) # y
    for i in range(N):
        bounds.append((0, 360)) # angle
    
    best_score = float('inf')
    best_result = None

    # Try SLSQP
    try:
        res = minimize(
            objective, 
            x0, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
        )
        
        if res.success or res.fun < best_score:
            # Verify validity strictly
            pos = res.x[:2*N].reshape(N, 2)
            ang = res.x[2*N:]
            polys = [get_hex_verts(pos[i][0], pos[i][1], 1.0, ang[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if sat_gap(polys[i], polys[j]) < -1e-5:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                best_score = calc_min_outer_side(pos, ang, 0.0)
                best_result = res.x
            elif res.fun < best_score:
                # Even if slightly invalid, keep as candidate if much better? 
                # No, we need valid. But maybe SLSQP found a local min that is valid but reported false?
                # Let's assume valid if penalty is low.
                # For safety, only accept if valid.
                pass
    except Exception as e:
        pass

    # If SLSQP failed or didn't improve, try L-BFGS-B with penalty (backup)
    if best_result is None or best_score > 5.0:
        def obj_penalty(params):
            if time.time() > deadline - 5:
                return 1000.0
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            
            side_len = calc_min_outer_side(positions, angles, 0.0)
            penalty = 0.0
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            for i in range(N):
                for j in range(i+1, N):
                    gap = sat_gap(polys[i], polys[j])
                    if gap < 0:
                        penalty += 10.0 * gap**2
                        
            return side_len + penalty

        try:
            res2 = minimize(
                obj_penalty, 
                x0, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8, 'maxls': 50}
            )
            
            pos = res2.x[:2*N].reshape(N, 2)
            ang = res2.x[2*N:]
            polys = [get_hex_verts(pos[i][0], pos[i][1], 1.0, ang[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if sat_gap(polys[i], polys[j]) < -1e-5:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                score2 = calc_min_outer_side(pos, ang, 0.0)
                if score2 < best_score:
                    best_score = score2
                    best_result = res2.x
        except Exception:
            pass

    # Final polish with Nelder-Mead if we have a good candidate
    if best_result is not None:
        try:
            def obj_nm(params):
                positions = params[:2*N].reshape(N, 2)
                angles = params[2*N:]
                side_len = calc_min_outer_side(positions, angles, 0.0)
                penalty = 0.0
                polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
                for i in range(N):
                    for j in range(i+1, N):
                        gap = sat_gap(polys[i], polys[j])
                        if gap < 0:
                            penalty += 50.0 * gap**2
                return side_len + penalty

            res3 = minimize(
                obj_nm, 
                best_result, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            pos = res3.x[:2*N].reshape(N, 2)
            ang = res3.x[2*N:]
            polys = [get_hex_verts(pos[i][0], pos[i][1], 1.0, ang[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if sat_gap(polys[i], polys[j]) < -1e-5:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                score3 = calc_min_outer_side(pos, ang, 0.0)
                if score3 < best_score:
                    best_score = score3
                    best_result = res3.x
        except Exception:
            pass

    # Finalize
    if best_result is not None:
        final_positions = best_result[:2*N].reshape(N, 2)
        final_angles = best_result[2*N:]
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    else:
        # Fallback
        final_positions = np.array([
            [0, 0], [2.1, 0], [-2.1, 0],
            [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
            [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
            [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
        ])
        final_angles = np.zeros(N)
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0

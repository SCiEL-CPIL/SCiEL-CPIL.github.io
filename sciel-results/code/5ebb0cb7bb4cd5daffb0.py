import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices for angle 0 (flat-top)
    # Vertices at 0, 60, ..., 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    # Normals for the 6 sides of a hexagon (pointing outward)
    # For a flat-top hexagon centered at origin, side 0 is vertical right? 
    # Vertices: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    # Edge 0: (1,0) to (0.5, sqrt(3)/2). Vector: (-0.5, sqrt(3)/2). Normal: (sqrt(3)/2, 0.5) -> angle 30 deg.
    # Edge 1: (0.5, sqrt(3)/2) to (-0.5, sqrt(3)/2). Vector: (-1, 0). Normal: (0, 1) -> angle 90 deg.
    # So normals are at 30, 90, 150, 210, 270, 330 degrees.
    unit_normals_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(30, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def get_hex_normals(angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        normals = unit_normals_0 @ rot.T
        return normals

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals for the outer hexagon
        # Outer hexagon contains points.
        # Distance from center to side is apothem = side * sqrt(3) / 2.
        # So side = 2 * max_proj / sqrt(3)
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
            
            # Gap on this axis: distance between intervals
            # If disjoint, gap > 0. If overlap, gap < 0 (penetration)
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
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

    def objective(params):
        if time.time() > deadline - 5:
            return 1000.0
            
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        return calc_min_outer_side(positions, angles, 0.0)

    # Create constraints for SLSQP
    # SLSQP constraints are functions that return a value >= 0 for feasibility
    # We want gap >= 0.
    def make_constraint(i, j):
        def constraint(params):
            if time.time() > deadline - 5:
                return 1.0 # Feasible to avoid crash
            
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            
            poly_i = get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
            poly_j = get_hex_verts(positions[j][0], positions[j][1], 1.0, angles[j])
            
            return sat_gap(poly_i, poly_j)
        
        return constraint

    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})

    # Initial guess: Ring + Center topology
    # 1 in center, 6 in inner ring, 5 in outer ring?
    # Or 6 in inner ring, 6 in outer ring?
    # Let's try 1 center, 6 ring 1, 5 ring 2.
    # Actually, 12 hexagons.
    # Center (0,0)
    # 6 around it at distance 2 (touching).
    # 5 more?
    
    sqrt3 = math.sqrt(3)
    
    # Better initialization: Honeycomb lattice subset
    # Generate a larger honeycomb and pick 12 closest to origin?
    # Or specific topology:
    # Center: (0,0)
    # Ring 1: 6 hexagons at distance 2*sqrt(3)/2? No, distance between centers of touching hexagons is 2*sqrt(3)/2 * 2?
    # Side length 1. Apothem sqrt(3)/2.
    # Distance between centers of touching hexagons is 2 * apothem = sqrt(3).
    # Wait.
    # Hexagon side 1.
    # Distance from center to vertex is 1.
    # Distance from center to side is sqrt(3)/2.
    # Two touching hexagons: distance between centers = 2 * (sqrt(3)/2) = sqrt(3) ~ 1.732.
    
    # Let's place 1 at (0,0).
    # 6 at distance sqrt(3) around it.
    # That's 7.
    # Remaining 5.
    # Place them in the "gaps" of the first ring?
    # The gaps are at distance 2 from center?
    # Let's just use the lattice points.
    
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
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10)) # x
        bounds.append((-10, 10)) # y
    for i in range(N):
        bounds.append((0, 360)) # angle

    best_score = float('inf')
    best_result = None

    # Try SLSQP with constraints
    try:
        res = minimize(
            objective, 
            x0, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
        )
        
        pos = res.x[:2*N].reshape(N, 2)
        ang = res.x[2*N:]
        polys = [get_hex_verts(pos[i][0], pos[i][1], 1.0, ang[i]) for i in range(N)]
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -1e-4:
                    valid = False
                    break
            if not valid: break
        
        if valid:
            best_score = calc_min_outer_side(pos, ang, 0.0)
            best_result = res.x
        else:
            # If SLSQP failed to satisfy constraints strictly, check if it's close
            # and try to fix with penalty method
            pass
    except Exception as e:
        pass

    # If SLSQP didn't find a good valid solution, try Penalty Method as fallback
    if best_result is None or best_score > 4.5:
        try:
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
                            penalty += 500.0 * gap**2
                return side_len + penalty

            start_x = best_result if best_result is not None else x0
            
            res2 = minimize(
                obj_penalty, 
                start_x, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'maxls': 50, 'disp': False}
            )
            
            pos = res2.x[:2*N].reshape(N, 2)
            ang = res2.x[2*N:]
            polys = [get_hex_verts(pos[i][0], pos[i][1], 1.0, ang[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if sat_gap(polys[i], polys[j]) < -1e-4:
                        valid = False
                        break
            if valid:
                score2 = calc_min_outer_side(pos, ang, 0.0)
                if score2 < best_score:
                    best_score = score2
                    best_result = res2.x
        except Exception:
            pass

    # Final polish: If we have a valid result, try SLSQP again from there to refine
    if best_result is not None and time.time() < deadline - 10:
        try:
            res3 = minimize(
                objective, 
                best_result, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-10, 'disp': False}
            )
            
            pos = res3.x[:2*N].reshape(N, 2)
            ang = res3.x[2*N:]
            polys = [get_hex_verts(pos[i][0], pos[i][1], 1.0, ang[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if sat_gap(polys[i], polys[j]) < -1e-4:
                        valid = False
                        break
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

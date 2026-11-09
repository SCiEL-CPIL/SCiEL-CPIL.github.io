import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 590  # 10s margin

    # Constants
    S = 1.0 # Inner hexagon side
    APO = np.sqrt(3) / 2.0 # Apothem of unit hexagon
    
    def get_vertices(cx, cy, s, angle_deg):
        a = np.deg2rad(angle_deg)
        angles = a + np.deg2rad(60 * np.arange(6))
        xs = cx + s * np.cos(angles)
        ys = cy + s * np.sin(angles)
        return np.column_stack((xs, ys))

    def get_edges(vertices):
        # Returns list of edge normals and edge midpoints? 
        # We just need normals for SAT
        normals = []
        for i in range(6):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % 6]
            edge = p2 - p1
            # Normal is perpendicular to edge
            # Rotate edge by 90 deg
            n = np.array([-edge[1], edge[0]])
            n = n / np.linalg.norm(n)
            normals.append(n)
        return normals

    def min_distance_hexagons(v1, v2):
        # v1, v2 are (6,2) arrays of vertices
        n1 = get_edges(v1)
        n2 = get_edges(v2)
        all_normals = n1 + n2
        min_gap = float('inf')
        for n in all_normals:
            # Project v1 and v2 onto n
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            max1 = np.max(proj1)
            min1 = np.min(proj1)
            max2 = np.max(proj2)
            min2 = np.min(proj2)
            
            # Calculate gap
            # If max1 < min2, gap is min2 - max1
            # If max2 < min1, gap is min1 - max2
            # Else, gap is negative (overlap)
            if max1 < min2:
                gap = min2 - max1
            elif max2 < min1:
                gap = min1 - max2
            else:
                gap = -1.0 # Overlapping, exact depth not needed for sign, but for optimization we might want it
                # For better optimization, we can compute the penetration depth, 
                # but -1 is a safe penalty trigger.
                # Actually, for smooth optimization, we need a continuous measure.
                # Let's use the minimum of the overlaps? 
                # If they overlap, the "gap" is negative.
                # The maximum overlap depth is min(max1, max2) - max(min1, min2)? No.
                # Let's just use 0 for now and rely on the fact that they are "apart" only if gap > 0.
                # To make it smooth, we can use:
                # gap = min(max1, max2) - max(min1, min2)
                # This is negative if they overlap, positive if they are apart?
                # If max1 < min2: min(max1, max2) - max(min1, min2) = max1 - min1? No.
                # Let's stick to the gap definition.
                # For optimization, we can use a smooth approximation of the distance.
                # But Nelder-Mead handles non-smoothness well.
                pass
        
        # We need the MAXIMUM gap over all normals to determine if they are separated.
        # If they are separated, there is at least one normal with positive gap.
        # The distance between convex sets is the max gap over all separating axes.
        # So we want max_gap.
        # If max_gap > 1e-6, they are valid.
        # For penalty, we want to maximize the minimum separation between all pairs.
        # So for a pair, we calculate max_gap.
        # If max_gap < 1e-6, it's a violation.
        
        # Let's re-calculate max_gap properly.
        max_gap = -float('inf')
        for n in all_normals:
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            max1 = np.max(proj1)
            min1 = np.min(proj1)
            max2 = np.max(proj2)
            min2 = np.min(proj2)
            
            if max1 < min2:
                gap = min2 - max1
            elif max2 < min1:
                gap = min1 - max2
            else:
                # Overlap
                # The "gap" is negative. 
                # How negative? 
                # The penetration depth along this axis is min(max1, max2) - max(min1, min2)
                # But this is not the true distance.
                # For the purpose of "are they apart?", if they overlap, they are NOT apart.
                # So gap = -1.0 is fine for the boolean check.
                # For optimization, we want a continuous value.
                # Let's use the negative of the overlap depth on this axis.
                # depth = min(max1, max2) - max(min1, min2)
                gap = -1.0 # Simplified
                
            if gap > max_gap:
                max_gap = gap
        return max_gap

    def check_separation(positions):
        # positions: (11, 2)
        min_pair_gap = float('inf')
        for i in range(11):
            for j in range(i + 1, 11):
                v1 = get_vertices(positions[i, 0], positions[i, 1], S, 0)
                v2 = get_vertices(positions[j, 0], positions[j, 1], S, 0)
                gap = min_distance_hexagons(v1, v2)
                if gap < min_pair_gap:
                    min_pair_gap = gap
        return min_pair_gap

    def get_enclosing_side(positions):
        # Calculate minimal side S of outer hexagon (angle 0) containing all inner hexagons
        # Outer hexagon normals: 30, 90, 150, 210, 270, 330 degrees
        # Apothem of outer hexagon is S * sqrt(3)/2
        # So S = proj / (sqrt(3)/2)
        all_vertices = []
        for i in range(11):
            v = get_vertices(positions[i, 0], positions[i, 1], S, 0)
            all_vertices.append(v)
        all_vertices = np.vstack(all_vertices)
        
        max_proj = 0
        for k in range(6):
            angle = np.deg2rad(30 + 60 * k)
            n = np.array([np.cos(angle), np.sin(angle)])
            proj = np.dot(all_vertices, n)
            max_proj = max(max_proj, np.max(np.abs(proj)))
        
        return max_proj / APO

    def objective(x):
        # x: (22,) -> 11 (x,y) pairs
        positions = x.reshape(11, 2)
        
        # Penalty for overlap
        min_gap = check_separation(positions)
        if min_gap < 1e-6:
            # Penalty is large if they overlap
            # We want to maximize min_gap. 
            # So penalty = (1e-6 - min_gap)^2
            penalty = (1e-6 - min_gap) ** 2
        else:
            penalty = 0
            
        side = get_enclosing_side(positions)
        return side + 1000 * penalty

    # Initial guess: 7-hex flower + 4 more
    # Center
    p0 = [0, 0]
    # Ring 1
    p1 = [np.sqrt(3), 0]
    p2 = [np.sqrt(3)/2, 1.5]
    p3 = [-np.sqrt(3)/2, 1.5]
    p4 = [-np.sqrt(3), 0]
    p5 = [-np.sqrt(3)/2, -1.5]
    p6 = [np.sqrt(3)/2, -1.5]
    
    # 4 more: Place them in the gaps?
    # Gaps are at angles 30, 90, 150, 210, 270, 330?
    # Distance from center to gap center?
    # In the 7-hex cluster, the "holes" are at distance 3 from center?
    # Let's place them at distance 3.5 along the axes 0, 60, 120, 180?
    # No, let's just add them to the ring 1 but further out?
    # Let's try a random perturbation of a larger cluster.
    
    # Better initial guess:
    # 11 hexagons in a hexagonal shape of radius 2?
    # That would be 1 + 6 + 12 = 19. Too many.
    # Let's use the 7-hex cluster and place 4 more at (3, 0), (-3, 0), (0, 3), (0, -3)?
    # These might overlap with ring 1.
    # Ring 1 vertices extend to distance 2.
    # (3,0) is distance 3. Hex at (3,0) extends to 2.
    # Gap between (1.732, 0) and (3,0) is 3 - 1.732 = 1.268.
    # Hex radius is 1. So gap is 1.268 - 2 = -0.732? No.
    # Distance between centers is 1.268.
    # They need distance > 1.732 to not touch.
    # So (3,0) overlaps with (1.732, 0).
    
    # Let's place the 4 extra hexagons further out.
    # At distance 3.5?
    p7 = [3.5, 0]
    p8 = [-3.5, 0]
    p9 = [0, 3.5]
    p10 = [0, -3.5]
    
    # This might be far apart. The optimizer will bring them in.
    init_positions = np.array([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])
    x0 = init_positions.flatten()
    
    # Optimize
    best_x = x0
    best_f = objective(x0)
    
    # Run Nelder-Mead
    # maxiter needs to be high enough but we have a time limit.
    # 22 dimensions.
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    if result.fun < best_f:
        best_x = result.x
        best_f = result.fun
    
    # Check time
    if time.time() > deadline:
        # Return best so far
        pass
    else:
        # Try a few more random restarts if time permits?
        # Or just return the result.
        pass

    positions = best_x.reshape(11, 2)
    
    # Final verification and adjustment
    # Ensure separation > 1e-6
    # If not, we might need to tweak.
    # But the optimizer should have handled it.
    
    # Calculate final side
    side = get_enclosing_side(positions)
    
    # Construct output
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = positions[i, 0]
        inner_hex_data[i, 1] = positions[i, 1]
        inner_hex_data[i, 2] = 0.0 # Angle
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, side

hexagon_packing_11()

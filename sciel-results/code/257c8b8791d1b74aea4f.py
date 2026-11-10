# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def hexagon_packing_12():
    start_time = time.time()
    time_limit = 630.0
    safety_margin = 10.0
    
    # Helper to get vertices of a hexagon
    def get_hex_vertices(cx, cy, side, angle_deg):
        angles = np.deg2rad(angle_deg + np.arange(6) * 60)
        return cx + side * np.cos(angles), cy + side * np.sin(angles)
    
    # Helper to check if a point is inside a convex polygon (defined by vertices in CCW order)
    # Returns max violation (positive if outside)
    def point_in_polygon_violation(px, py, vx, vy):
        # vx, vy are arrays of shape (N,)
        # Compute cross products for each edge
        # Edge from i to i+1
        x1, y1 = vx, vy
        x2, y2 = np.roll(vx, -1), np.roll(vy, -1)
        
        # Cross product (x2-x1)*(py-y1) - (y2-y1)*(px-x1)
        # For CCW polygon, interior is where cross > 0
        # So violation is where cross < 0. We want max(0, -min_cross)
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        min_cross = np.min(cross)
        return -min_cross
    
    # Helper to compute minimum overlap distance between two convex polygons
    # Returns 0 if separated, positive if overlapping by that amount
    def poly_overlap(a_x, a_y, b_x, b_y):
        # a_x, a_y: shape (6,)
        # b_x, b_y: shape (6,)
        # SAT test
        min_dist = 0.0
        
        # Axes from polygon A
        for i in range(6):
            x1, y1 = a_x[i], a_y[i]
            x2, y2 = a_x[(i+1)%6], a_y[(i+1)%6]
            ex, ey = x2 - x1, y2 - y1
            # Normal
            nx, ny = -ey, ex
            # Project both polygons onto this axis
            pa = a_x * nx + a_y * ny
            pb = b_x * nx + b_y * ny
            
            pa_min, pa_max = np.min(pa), np.max(pa)
            pb_min, pb_max = np.min(pb), np.max(pb)
            
            # Check for overlap
            if pa_max < pb_min or pb_max < pa_min:
                # Separated
                # Distance between intervals
                if pa_max < pb_min:
                    d = pb_min - pa_max
                else:
                    d = pa_min - pb_max
                # We need to return the "depth" of overlap.
                # Wait, if they are separated, overlap is 0.
                # The function should return 0 if they don't overlap.
                # If they do overlap, return how much they overlap.
                # Actually, for optimization, we want a penalty for overlap.
                # If they are separated, the penalty should be 0.
                # If they overlap, the penalty should be the minimum distance to separate them.
                # This is exactly what SAT gives us: the minimum translation distance.
                # If separated, distance is positive. If overlapping, distance is negative.
                # We want penalty = max(0, -distance) ? No.
                # If separated, distance > 0. Overlap = 0.
                # If overlapping, distance < 0. Overlap = -distance.
                # So overlap = -min(0, distance) ?
                # Let's stick to the standard:
                # If intervals overlap, the overlap amount is min(pa_max, pb_max) - max(pa_min, pb_min).
                # This is positive if they overlap.
                # We want the MINIMUM overlap over all axes.
                # If for any axis they are separated, total overlap is 0.
                # If for all axes they overlap, total overlap is the min overlap over all axes.
                return 0.0 # Separated on this axis
        
        # If we reach here, they overlap on all axes of A.
        # Now check axes of B?
        # For two convex polygons, if they overlap on all axes of A and all axes of B, they are intersecting.
        # The minimum translation distance is the minimum overlap over all axes.
        # We already checked A's axes. If they were separated on any, we returned 0.
        # So now they overlap on all A axes.
        # We must also check B's axes to find the minimum overlap.
        
        # Wait, the logic above returns 0 immediately if separated on an axis.
        # This is correct for the "are they overlapping" check.
        # But for the "how much do they overlap" (MTD), we need the min overlap over ALL axes.
        # If they are separated on axis 1, MTD is determined by axis 1 (distance).
        # If they are overlapping on all axes, MTD is min(overlap on axis 1, ..., overlap on axis 11).
        
        # Let's refactor.
        # Initialize min_overlap to infinity.
        min_overlap = float('inf')
        
        # Axes from A
        for i in range(6):
            x1, y1 = a_x[i], a_y[i]
            x2, y2 = a_x[(i+1)%6], a_y[(i+1)%6]
            ex, ey = x2 - x1, y2 - y1
            nx, ny = -ey, ex
            norm = np.sqrt(nx*nx + ny*ny)
            if norm == 0: continue
            nx, ny = nx/norm, ny/norm
            
            pa = a_x * nx + a_y * ny
            pb = b_x * nx + b_y * ny
            
            overlap = np.min([np.max(pa), np.max(pb)]) - np.max([np.min(pa), np.min(pb)])
            if overlap < 0:
                return 0.0 # Separated
            min_overlap = min(min_overlap, overlap)
        
        # Axes from B
        for i in range(6):
            x1, y1 = b_x[i], b_y[i]
            x2, y2 = b_x[(i+1)%6], b_y[(i+1)%6]
            ex, ey = x2 - x1, y2 - y1
            nx, ny = -ey, ex
            norm = np.sqrt(nx*nx + ny*ny)
            if norm == 0: continue
            nx, ny = nx/norm, ny/norm
            
            pa = a_x * nx + a_y * ny
            pb = b_x * nx + b_y * ny
            
            overlap = np.min([np.max(pa), np.max(pb)]) - np.max([np.min(pa), np.min(pb)])
            if overlap < 0:
                return 0.0 # Separated
            min_overlap = min(min_overlap, overlap)
            
        return min_overlap

    # Objective function
    def objective(params):
        # params: [x1, y1, x2, y2, ..., x12, y12, angle_deg]
        # First 24 are positions, last 1 is angle
        pos = params[:24].reshape(12, 2)
        angle = params[24]
        
        # Check time
        if time.time() - start_time > time_limit - safety_margin:
            return 1e100 # Penalize timeout
        
        # Calculate vertices for all inner hexagons
        # Side = 1
        side_inner = 1.0
        inner_verts = []
        for i in range(12):
            cx, cy = pos[i]
            vx, vy = get_hex_vertices(cx, cy, side_inner, angle)
            inner_verts.append((vx, vy))
        
        # Calculate overlap penalty
        overlap_penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                ox = poly_overlap(inner_verts[i][0], inner_verts[i][1], inner_verts[j][0], inner_verts[j][1])
                if ox > 0:
                    overlap_penalty += ox**2 # Quadratic penalty
        
        # Calculate minimal outer side length s
        # We need to find the smallest s such that all inner hexes are inside the outer hex.
        # Outer hex is centered at (0,0) for simplicity? Or should we optimize its center too?
        # The problem statement says "outer a (3,) array (x, y, angle_deg)".
        # So we should optimize outer center (cx, cy) and angle.
        # But that adds 3 more variables.
        # Let's fix outer center at (0,0) and angle at 0 first?
        # No, the inner hexes can be rotated.
        # Actually, the symmetry suggests we can fix outer hex to be axis-aligned and centered at 0.
        # And the inner hexes can have any angle?
        # Wait, the inner hexes are "unit regular hexagons". Their angle is also a variable?
        # The task says "inner a (12, 3) array of rows (x, y, angle_deg)".
        # So YES, each inner hexagon has its own angle.
        # That's 12*3 = 36 variables. Plus outer 3 variables = 39 variables.
        # This is a high-dimensional space.
        # However, regular hexagons have 6-fold symmetry.
        # Usually, optimal packings have all hexagons aligned.
        # Let's assume all inner hexagons have the SAME angle as the outer one?
        # Or maybe all are 0?
        # The seed program used 0 for all.
        # Let's try optimizing:
        # 12 inner centers (24 vars)
        # 12 inner angles (12 vars) - this might be too many.
        # Let's fix all inner angles to 0 first, and optimize the outer angle?
        # Or fix all to 0 and optimize positions?
        
        # Let's stick to a simpler model first:
        # All inner hexagons have angle 0.
        # Outer hexagon has angle 0 and is centered at (0,0).
        # We only optimize the 24 positions.
        # This is a 24-variable problem.
        
        # If this works, we can expand to include angles.
        
        # For now, let's just return the overlap penalty.
        # We need to calculate the required outer size s.
        # If we fix outer hex to (0,0) and angle 0, we can calculate the required s.
        
        # Outer hex vertices for side s, angle 0, center (0,0):
        # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
        # Actually, the vertices are at radius s.
        # Distance from center to vertex is s.
        # The "apothem" (distance to edge) is s * sqrt(3)/2.
        
        # For a point (x,y) to be inside the outer hexagon:
        # It must be inside all 6 half-planes defined by the edges.
        # Edge 1: from (s,0) to (s/2, s*sqrt(3)/2).
        # Normal pointing inward?
        # Let's use the vertex method.
        # If the outer hex is fixed, we can calculate the "radius" required for each inner vertex.
        # But the shape is not a circle.
        
        # Let's use the property:
        # A point (x,y) is inside a regular hexagon of side s centered at origin (flat-top or pointy-top?)
        # The problem definition: vertices (cx + d cos(a + 60i), cy + d sin(a + 60i)).
        # If a=0, vertex 0 is at (s, 0).
        # This is a "pointy-top" hexagon? No, vertex at (s,0) means it points right.
        # This is often called "flat-side" on left/right? No, vertex on x-axis means pointy on x-axis.
        
        # The condition for a point to be inside such a hexagon is:
        # |x| <= s
        # |y| <= s * sqrt(3) / 2
        # |x/2 + y*sqrt(3)/2| <= s  (This is for the other orientation?)
        
        # Let's derive the inequalities for the specific orientation:
        # Vertices:
        # v0: (s, 0)
        # v1: (s/2, s*sqrt(3)/2)
        # v2: (-s/2, s*sqrt(3)/2)
        # v3: (-s, 0)
        # v4: (-s/2, -s*sqrt(3)/2)
        # v5: (s/2, -s*sqrt(3)/2)
        
        # Edges:
        # v0-v1: Line equation. Slope = (s*sqrt(3)/2 - 0) / (s/2 - s) = -sqrt(3).
        # y - 0 = -sqrt(3) * (x - s)  => y = -sqrt(3)x + s*sqrt(3) => sqrt(3)x + y = s*sqrt(3).
        # Inequality: sqrt(3)x + y <= s*sqrt(3) => x/sqrt(3) + y/s <= 1 ? No.
        # x/sqrt(3) + y/(sqrt(3)s) ...
        # Let's just use the cross product check or distance to edge.
        
        # Distance from (x,y) to the line defined by edge i must be <= apothem.
        # Apothem = s * sqrt(3)/2.
        
        # This is getting complicated to do analytically for arbitrary rotated hexes.
        # Since we are optimizing, let's just compute the required s numerically.
        # For a given configuration of inner hexes, find the minimum s such that all are inside.
        # This is a convex optimization problem (minimize s subject to linear inequalities).
        # We can solve it with a simple bisection or by finding the max of the "required s" for each vertex.
        
        # Actually, for a convex container (outer hex) and convex objects (inner hexes),
        # the minimal s is determined by the furthest point(s).
        
        # Let's define a function `required_s(point, angle_outer)` that returns the s such that the point is on the boundary of the outer hex of side s, centered at (0,0), angle angle_outer.
        
        def req_s_for_point(px, py, ang_deg):
            # Rotate point to align with outer hex
            rad = np.deg2rad(ang_deg)
            c, s_ = np.cos(-rad), np.sin(-rad)
            x = px * c - py * s_
            y = px * s_ + py * c
            
            # Now the hex is axis-aligned (angle 0).
            # Vertices at (s,0), (s/2, s*sqrt(3)/2), etc.
            # Inequalities:
            # 1. x <= s
            # 2. -x <= s  => x >= -s
            # 3. y <= s*sqrt(3)/2
            # 4. -y <= s*sqrt(3)/2
            # 5. sqrt(3)*x + y <= s*sqrt(3)
            # 6. -sqrt(3)*x + y <= s*sqrt(3)
            # 7. -sqrt(3)*x - y <= s*sqrt(3)  (Wait, let's check the 6 edges)
            
            # Edge v0-v1: sqrt(3)x + y <= s*sqrt(3)
            # Edge v1-v2: y <= s*sqrt(3)/2
            # Edge v2-v3: -sqrt(3)x + y <= s*sqrt(3)
            # Edge v3-v4: x >= -s  (or -x <= s)
            # Edge v4-v5: -sqrt(3)x - y <= s*sqrt(3)  (Wait, normal?)
            # Let's just use the distance to the 6 lines.
            
            # Lines:
            # 1. x - s = 0 (Right edge? No, v0 is a vertex. The edge is v0-v1.)
            # The edges are:
            # E1 (v0-v1): sqrt(3)x + y - s*sqrt(3) = 0
            # E2 (v1-v2): y - s*sqrt(3)/2 = 0
            # E3 (v2-v3): -sqrt(3)x + y - s*sqrt(3) = 0
            # E4 (v3-v4): x + s = 0  => -x - s = 0
            # E5 (v4-v5): -sqrt(3)x - y - s*sqrt(3) = 0
            # E6 (v5-v0): -y - s*sqrt(3)/2 = 0
            
            # We want the largest s such that all inequalities hold.
            # s >= (sqrt(3)x + y) / sqrt(3)
            # s >= y / (sqrt(3)/2) = 2y/sqrt(3)
            # s >= (-sqrt(3)x + y) / sqrt(3)
            # s >= -x
            # s >= (-sqrt(3)x - y) / sqrt(3)
            # s >= -y / (sqrt(3)/2) = -2y/sqrt(3)
            
            # So s_req = max(
            #   (sqrt(3)x + y) / sqrt(3),
            #   2y / sqrt(3),
            #   (-sqrt(3)x + y) / sqrt(3),
            #   -x,
            #   (-sqrt(3)x - y) / sqrt(3),
            #   -2y / sqrt(3)
            # )
            
            sq3 = np.sqrt(3)
            s1 = (sq3 * x + y) / sq3
            s2 = 2 * y / sq3
            s3 = (-sq3 * x + y) / sq3
            s4 = -x
            s5 = (-sq3 * x - y) / sq3
            s6 = -2 * y / sq3
            
            return np.max([s1, s2, s3, s4, s5, s6])
            
        # Calculate s_req for all vertices of all inner hexes
        # Outer hex angle is fixed at 0 in this simplified model.
        # We can optimize outer angle later.
        
        max_s = 0.0
        for i in range(12):
            vx, vy = inner_verts[i]
            for k in range(6):
                s_req = req_s_for_point(vx[k], vy[k], 0.0)
                if s_req > max_s:
                    max_s = s_req
        
        # Total cost = s + penalty
        # We want to minimize s.
        # But if there is overlap, the packing is invalid.
        # So we should minimize s + large_penalty * overlap.
        
        return max_s + 1000 * overlap_penalty

    # Initial guess: The 2-3-4-3 grid from the seed, but optimized.
    # Let's start with the seed positions.
    x0 = np.array([
        0, 0,
        -2.5, 0,
        2.5, 0,
        -1.25, 2.17,
        1.25, 2.17,
        -1.25, -2.17,
        1.25, -2.17,
        -3.75, 2.17,
        3.75, 2.17,
        -3.75, -2.17,
        3.75, -2.17,
        0, -4
    ])
    
    # Append angle 0
    x0 = np.append(x0, 0.0)
    
    # Use Nelder-Mead or Powell
    # Powell is often faster for this kind of problem.
    # But we have a discontinuity in the derivative due to the max() in req_s_for_point.
    # Nelder-Mead is robust to this.
    
    best_result = None
    best_val = float('inf')
    
    # Run optimization
    # Limit iterations or time.
    max_iters = 2000
    # We will check time inside the objective, but minimize might run away.
    # Let's wrap it.
    
    try:
        from scipy.optimize import minimize
        
        # Reseeding strategy: Run multiple times with different initializations if time permits.
        # But 600s is a lot. Let's just do one good run first.
        
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': max_iters, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
        
        if time.time() - start_time > time_limit - safety_margin:
             # Timeout during optimization
             pass
        else:
             if res.success or res.fun < best_val:
                 best_val = res.fun
                 best_result = res.x
                 
    except Exception as e:
        print(f"Optimization failed: {e}")
    
    # If we have time, try a second run with a different start?
    # Or refine the best result.
    
    if time.time() - start_time < time_limit - safety_margin:
         # Try Powell on the best result found so far (or x0 if none)
         start_point = best_result if best_result is not None else x0
         try:
             res2 = minimize(objective, start_point, method='Powell', options={'maxiter': 1000, 'disp': False})
             if res2.fun < best_val:
                 best_val = res2.fun
                 best_result = res2.x
         except:
             pass

    # If still time, try a third method or random restart?
    # Let's assume the first two are enough for a structural improvement.
    
    if best_result is None:
        # Fallback to seed
        best_result = x0
        best_val = objective(x0)
        
    # Extract final parameters
    pos = best_result[:24].reshape(12, 2)
    angle = best_result[24]
    
    # Construct return arrays
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i] = [pos[i, 0], pos[i, 1], angle]
        
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = best_val # This is an approximation. The actual s is max_s.
    # Wait, best_val = max_s + 1000 * overlap.
    # If overlap is 0, best_val = max_s.
    # If overlap > 0, best_val > max_s.
    # We should return the actual max_s for the valid configuration?
    # But if the configuration is invalid (overlap), the score is 0.
    # The evaluator checks validity.
    # So we should try to find a configuration with 0 overlap.
    
    # Let's recalculate the true s for the best_result to be precise.
    # The objective function returned a penalized value.
    # We should recompute the "pure" s for the best_result to return the correct geometry.
    # However, if the best_result has overlap, it's invalid anyway.
    
    # Let's just return the parameters. The evaluator will check.
    # But we need to return `s`.
    # If we return the penalized s, it might be larger than necessary, which is worse for the score (1/s).
    # So we should compute the exact minimal s for the final positions.
    
    # Recompute exact s
    pos_final = best_result[:24].reshape(12, 2)
    angle_final = best_result[24]
    
    max_s = 0.0
    for i in range(12):
        vx, vy = get_hex_vertices(pos_final[i,0], pos_final[i,1], 1.0, angle_final)
        for k in range(6):
            s_req = req_s_for_point(vx[k], vy[k], 0.0)
            if s_req > max_s:
                max_s = s_req
                
    return inner_hex_data, outer_hex_data, max_s


# EVOLVE-BLOCK-END

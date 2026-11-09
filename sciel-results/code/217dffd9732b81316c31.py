import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 10s buffer
    
    # Seed points
    P1 = (0, 0)
    P2 = (-56, 33)
    P3 = (-16, 63)
    P4 = (0, 33)
    P5 = (16, 63)
    
    base_points = [P1, P2, P3, P4, P5]
    
    # Helper functions
    def is_square(n):
        if n < 0: return False
        r = math.isqrt(n)
        return r*r == n
        
    def dist_sq(p1, p2):
        return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
        
    def is_int_dist(p1, p2):
        d2 = dist_sq(p1, p2)
        return is_square(d2)
        
    def are_collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) == 0
        
    def are_concyclic(p1, p2, p3, p4):
        # Check if 4 points are concyclic
        # Det of matrix:
        # x^2+y^2  x  y  1
        # ...
        # If det is 0, they are concyclic or collinear.
        # We assume collinearity is checked separately.
        
        def det4(m):
            # m is 4x4 list of lists
            return (
                m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
                - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
                + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
                - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
            )

        M = [[p[0]**2 + p[1]**2, p[0], p[1], 1] for p in [p1, p2, p3, p4]]
        return det4(M) == 0

    def check_validity(points):
        n = len(points)
        # Check all pairwise distances
        for i in range(n):
            for j in range(i+1, n):
                if not is_int_dist(points[i], points[j]):
                    return False
        # Check collinearity
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        return False
        # Check concyclicity
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if are_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    # Search for additional points using grid search
    # We will try to add points one by one.
    current_points = base_points[:]
    
    # Search range
    # The base points are in [-56, 56] x [0, 63].
    # Let's search in a box around them.
    # If we find a 6th point, we can try to find a 7th.
    
    search_range = 200 # Initial range
    
    # We'll iterate through potential new points (x, y)
    # and check if they can be added to current_points.
    
    # To make it faster, we can pre-filter based on distance to the first point P1(0,0).
    # If (x,y) is a candidate, x^2 + y^2 must be a square.
    # This reduces candidates significantly.
    
    # Let's generate a list of candidate points (x,y) such that dist to P1 is integer.
    # And dist to P4(0,33) is integer.
    # This is the same constraint as the original code, but we can iterate over x,y directly or use the distance logic.
    # The original code iterated k1, k4. Let's stick to that but maybe expand range or change anchor points.
    
    # Actually, the original code's logic was:
    # x^2 + y^2 = k1^2
    # x^2 + (y-33)^2 = k4^2
    # This implies 66y - 1089 = k1^2 - k4^2.
    # y = (k1^2 - k4^2 + 1089) / 66.
    # This is a very efficient way to generate candidates that satisfy distance constraints to P1 and P4.
    # The issue might be that max_k=300 is too small, or no solution exists in that range.
    
    # Let's increase max_k to 1000 and see if it finds a point.
    # 1000 * 1000 = 1,000,000 iterations. Very fast.
    
    found_point = None
    max_k = 1000 # Increased search range
    
    for k1 in range(1, max_k):
        for k4 in range(1, max_k):
            if time.time() > deadline:
                break
                
            num_y = k1*k1 - k4*k4 + 1089
            if num_y % 66 != 0:
                continue
            
            y = num_y // 66
            
            # x^2 = k1^2 - y^2
            x2 = k1*k1 - y*y
            if x2 < 0:
                continue
            
            if not is_square(x2):
                continue
            
            x = math.isqrt(x2)
            
            # We have two candidates: (x, y) and (-x, y)
            for cx in [x, -x]:
                P6 = (cx, y)
                
                # Check distinctness
                if P6 in current_points:
                    continue
                
                # Check integer distances to all existing points
                valid = True
                for p in current_points:
                    if not is_int_dist(P6, p):
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check collinearity with all pairs of existing points
                n_curr = len(current_points)
                for i in range(n_curr):
                    for j in range(i+1, n_curr):
                        if are_collinear(P6, current_points[i], current_points[j]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Check concyclicity with all triples of existing points
                for i in range(n_curr):
                    for j in range(i+1, n_curr):
                        for k in range(j+1, n_curr):
                            if are_concyclic(P6, current_points[i], current_points[j], current_points[k]):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                    if not valid:
                        break
                
                if valid:
                    # Final check of the whole set
                    new_set = current_points + [P6]
                    if check_validity(new_set):
                        found_point = P6
                        break
            if found_point:
                break
        if found_point:
            break
            
    if found_point:
        current_points.append(found_point)
        
    # Try to find a 7th point?
    # The constraints become much tighter. 
    # We can try the same method with the new set of 6 points.
    # We need to pick 2 anchor points from the 6 to constrain the new point.
    # Let's pick the first two points (0,0) and (0,33) again, as they are fixed.
    # But the new point must also be at integer distance from the other 4 points.
    
    if len(current_points) == 6:
        found_point_7 = None
        # Use P1(0,0) and P4(0,33) as anchors again
        for k1 in range(1, max_k):
            for k4 in range(1, max_k):
                if time.time() > deadline:
                    break
                    
                num_y = k1*k1 - k4*k4 + 1089
                if num_y % 66 != 0:
                    continue
                
                y = num_y // 66
                
                x2 = k1*k1 - y*y
                if x2 < 0:
                    continue
                
                if not is_square(x2):
                    continue
                
                x = math.isqrt(x2)
                
                for cx in [x, -x]:
                    P7 = (cx, y)
                    
                    if P7 in current_points:
                        continue
                    
                    valid = True
                    for p in current_points:
                        if not is_int_dist(P7, p):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    n_curr = len(current_points)
                    for i in range(n_curr):
                        for j in range(i+1, n_curr):
                            if are_collinear(P7, current_points[i], current_points[j]):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    for i in range(n_curr):
                        for j in range(i+1, n_curr):
                            for k in range(j+1, n_curr):
                                if are_concyclic(P7, current_points[i], current_points[j], current_points[k]):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                        if not valid:
                            break
                    
                    if valid:
                        new_set = current_points + [P7]
                        if check_validity(new_set):
                            found_point_7 = P7
                            break
                if found_point_7:
                    break
            if found_point_7:
                break
                
        if found_point_7:
            current_points.append(found_point_7)

    final_points = current_points
        
    # Convert to required format
    D = 1
    points_out = []
    for p in final_points:
        points_out.append([
            [p[0], 1],
            [p[1], 1]
        ])
        
    return {'D': D, 'points': points_out}

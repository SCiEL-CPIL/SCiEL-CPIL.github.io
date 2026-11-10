import time
import math
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit to be safe
    
    # Seed points from the current program
    # These are known to be valid: 5 points, D=1, integer distances, no 3 collinear, no 4 concyclic
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def dist_sq(p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    def are_collinear(p1, p2, p3):
        # (x2-x1)(y3-y1) - (y2-y1)(x3-x1) == 0
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

    def are_concyclic(p1, p2, p3, p4):
        # Determinant of the matrix:
        # x^2+y^2, x, y, 1
        # For 4 points, det must be 0
        # Using expansion or direct calculation
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Matrix rows:
        # r1^2, x1, y1, 1
        # r2^2, x2, y2, 1
        # r3^2, x3, y3, 1
        # r4^2, x4, y4, 1
        
        # Subtract row 1 from others to simplify? 
        # Or just compute determinant directly.
        # Let's use the property that 4 points are concyclic iff:
        # det([[x1^2+y1^2, x1, y1, 1], ...]) == 0
        
        # Optimization: translate p1 to origin
        # Let q1 = (0,0)
        # q2 = (x2-x1, y2-y1)
        # q3 = (x3-x1, y3-y1)
        # q4 = (x4-x1, y4-y1)
        # Points are concyclic iff det([[0,0,0,1], [q2x^2+q2y^2, q2x, q2y, 1], ...]) == 0
        # This is equivalent to checking if the circle through q2, q3, q4 passes through q1.
        # Or simpler: 4 points are concyclic iff the circumcircle of any 3 passes through the 4th.
        
        # Let's compute the circumcircle of p1, p2, p3 and check if p4 is on it.
        # Equation: x^2 + y^2 + Dx + Ey + F = 0
        # p1: x1^2+y1^2 + D x1 + E y1 + F = 0
        # p2: x2^2+y2^2 + D x2 + E y2 + F = 0
        # p3: x3^2+y3^2 + D x3 + E y3 + F = 0
        
        # Subtract p1 from p2 and p3:
        # (x2^2-x1^2) + (y2^2-y1^2) + D(x2-x1) + E(y2-y1) = 0
        # (x3^2-x1^2) + (y3^2-y1^2) + D(x3-x1) + E(y3-y1) = 0
        
        # This is a linear system for D and E.
        # A D + B E = C
        # D D + E E = F
        
        # A = x2-x1, B = y2-y1, C = -( (x2^2-x1^2) + (y2^2-y1^2) )
        # D = x3-x1, E = y3-y1, F = -( (x3^2-x1^2) + (y3^2-y1^2) )
        
        # Det = A*E - B*D
        # If Det == 0, points are collinear (handled separately).
        # D_val = (C*E - B*F) / Det
        # E_val = (A*F - C*D) / Det
        
        # Then check p4: x4^2+y4^2 + D_val*x4 + E_val*y4 + F_val == 0
        # F_val = -(x1^2+y1^2 + D_val*x1 + E_val*y1)
        
        # To avoid fractions, we can check:
        # (x4^2+y4^2)*Det + D_val_num*(A*x4 - B*y4) + E_val_num*(D*x4 - E*y4) + F_val_num*Det == 0 ?
        # Actually, easier:
        # Let M = [[x1^2+y1^2, x1, y1, 1], ...]
        # Just compute the 4x4 determinant.
        
        def det4(m):
            # m is 4x4 list of lists
            # Expansion along first row
            return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1])) -
                    m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0])) +
                    m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])) -
                    m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])))
        
        r1sq = x1*x1 + y1*y1
        r2sq = x2*x2 + y2*y2
        r3sq = x3*x3 + y3*y3
        r4sq = x4*x4 + y4*y4
        
        mat = [
            [r1sq, x1, y1, 1],
            [r2sq, x2, y2, 1],
            [r3sq, x3, y3, 1],
            [r4sq, x4, y4, 1]
        ]
        return det4(mat) == 0

    def is_valid_set(points):
        n = len(points)
        # Check no 3 collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        return False
        # Check no 4 concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if are_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        # Check distances are integers (for D=1, integer coords)
        for i in range(n):
            for j in range(i+1, n):
                if not is_square(dist_sq(points[i], points[j])):
                    return False
        return True

    # Verify base points
    if not is_valid_set(base_points):
        # Should not happen, but fallback to a trivial valid set of 3
        # e.g., (0,0), (3,0), (0,4) -> dist 3,4,5. No 3 collinear (only 3 pts). No 4 concyclic.
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[3, 1], [0, 1]], [[0, 1], [4, 1]]]}

    # Try to find a 6th point
    # Search range
    search_range = 300
    found = False
    best_points = base_points
    
    # Optimization: Pre-filter candidates by distance to base points
    # For a candidate P, dist_sq(P, Pi) must be a perfect square for all i in 0..4
    
    for x in range(-search_range, search_range + 1):
        if time.time() > deadline:
            break
        for y in range(-search_range, search_range + 1):
            if time.time() > deadline:
                break
            
            cand = (x, y)
            
            # Check distinctness
            if cand in base_points:
                continue
                
            # Check distances to base points
            valid_dist = True
            for p in base_points:
                if not is_square(dist_sq(cand, p)):
                    valid_dist = False
                    break
            
            if not valid_dist:
                continue
            
            # Check collinearity with base points
            valid_coll = True
            for i in range(len(base_points)):
                for j in range(i+1, len(base_points)):
                    if are_collinear(base_points[i], base_points[j], cand):
                        valid_coll = False
                        break
                if not valid_coll:
                    break
            if not valid_coll:
                continue
            
            # Check concyclicity with base points
            # We need to check if cand lies on any circle defined by 3 base points
            valid_conc = True
            for i in range(len(base_points)):
                for j in range(i+1, len(base_points)):
                    for k in range(j+1, len(base_points)):
                        # Circle through Pi, Pj, Pk. Does P_cand lie on it?
                        # This is equivalent to are_concyclic(Pi, Pj, Pk, cand)
                        if are_concyclic(base_points[i], base_points[j], base_points[k], cand):
                            valid_conc = False
                            break
                    if not valid_conc:
                        break
                if not valid_conc:
                    break
            if not valid_conc:
                continue
            
            # If we get here, it's a valid 6th point
            new_points = base_points + [cand]
            # Final full verification just in case
            if is_valid_set(new_points):
                best_points = new_points
                found = True
                break
        if found:
            break
            
    if not found:
        # Try a slightly larger range or different base if time permits?
        # For now, return the 5 points if 6 not found quickly.
        # But we can try to search for a 6th point in a different configuration?
        # The seed's 5 points are symmetric. Maybe a different 5-point set allows a 6th?
        # Let's stick with the found 5 points.
        
        # If we have time, we could try random 5-point sets and extend them.
        # But the current 5 points are a good start.
        
        pass

    # Format output
    # Points are (x, y) with D=1
    # Format: [[[x_num, x_den], [y_num, y_den]], ...]
    res_points = []
    for p in best_points:
        res_points.append([ [p[0], 1], [p[1], 1] ])
        
    return {'D': 1, 'points': res_points}

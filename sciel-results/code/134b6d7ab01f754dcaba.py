import math
import time

def construct():
    # Start with the known 5-point set
    points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify the 5-point set is valid (it is, per seed)
    # Check for 6th point
    # We search for integer (x, y) in a box
    # To avoid timeout, limit search space or time
    
    start_time = time.time()
    timeout = 110.0 # 10s margin
    
    # Search range
    # The points are roughly in [-60, 60] x [0, 65]
    # Let's search [-100, 100] x [-100, 100]
    
    # Optimization: Precompute squares up to a limit
    # Max dist squared approx (200)^2 + (200)^2 = 80000
    # sqrt(80000) ~ 283
    # We need to check if d2 is a square.
    
    def is_square(n):
        if n < 0: return False
        r = int(math.isqrt(n))
        return r * r == n

    found_6th = None
    # Iterate x, y
    # To make it efficient, we can iterate and break early if time runs out
    # But 200x200 = 40,000 candidates. 5 points each. 200,000 checks.
    # Each check is 5 is_square calls. Very fast.
    
    for x in range(-100, 101):
        for y in range(-100, 101):
            if time.time() - start_time > timeout:
                break
            
            # Check if (x,y) is distinct from existing points
            if (x, y) in points:
                continue
                
            valid = True
            for px, py in points:
                dx = x - px
                dy = y - py
                d2 = dx*dx + dy*dy
                if not is_square(d2):
                    valid = False
                    break
            
            if valid:
                # Check collinearity and concyclicity for the new set
                new_points = points + [(x, y)]
                if is_valid_set(new_points):
                    found_6th = (x, y)
                    break
        if found_6th:
            break
            
    if found_6th:
        points.append(found_6th)
        
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}

def is_valid_set(points):
    n = len(points)
    # Check no three collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
                area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                if area == 0:
                    return False
                    
    # Check no four concyclic
    # Equation of circle: x^2 + y^2 + Dx + Ey + F = 0
    # For 4 points, the determinant of the matrix must be non-zero
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    x4, y4 = points[l]
                    
                    # Determinant of:
                    # [x1^2+y1^2, x1, y1, 1]
                    # [x2^2+y2^2, x2, y2, 1]
                    # [x3^2+y3^2, x3, y3, 1]
                    # [x4^2+y4^2, x4, y4, 1]
                    
                    # Calculate determinant manually or use a helper
                    # For small n, this is fine
                    if is_concyclic(points[i], points[j], points[k], points[l]):
                        return False
                        
    return True

def is_concyclic(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Use the property that 4 points are concyclic iff the determinant is 0
    # det = | x1^2+y1^2  x1  y1  1 |
    #       | x2^2+y2^2  x2  y2  1 |
    #       | x3^2+y3^2  x3  y3  1 |
    #       | x4^2+y4^2  x4  y4  1 |
    
    # Expand along the last column? Or just compute.
    # Given small numbers, exact integer arithmetic is safe.
    
    a1 = x1*x1 + y1*y1
    a2 = x2*x2 + y2*y2
    a3 = x3*x3 + y3*y3
    a4 = x4*x4 + y4*y4
    
    # Determinant of 4x4
    # M = [[a1, x1, y1, 1],
    #      [a2, x2, y2, 1],
    #      [a3, x3, y3, 1],
    #      [a4, x4, y4, 1]]
    
    # We can compute the determinant using row operations to simplify
    # Subtract row 4 from rows 1, 2, 3
    # R1 = [a1-a4, x1-x4, y1-y4, 0]
    # R2 = [a2-a4, x2-x4, y2-y4, 0]
    # R3 = [a3-a4, x3-x4, y3-y4, 0]
    # R4 = [a4, x4, y4, 1]
    
    # Det = 1 * det([[a1-a4, x1-x4, y1-y4],
    #                [a2-a4, x2-x4, y2-y4],
    #                [a3-a4, x3-x4, y3-y4]])
    
    b1 = a1 - a4
    b2 = a2 - a4
    b3 = a3 - a4
    
    c1 = x1 - x4
    c2 = x2 - x4
    c3 = x3 - x4
    
    d1 = y1 - y4
    d2 = y2 - y4
    d3 = y3 - y4
    
    det = b1*(c2*d3 - c3*d2) - c1*(b2*d3 - b3*d2) + d1*(b2*c3 - b3*c2)
    
    return det == 0

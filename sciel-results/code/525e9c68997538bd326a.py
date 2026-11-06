import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Seed 5-point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Function to check if a set of points is valid
    def is_valid(points):
        n = len(points)
        if n < 3:
            return False
        
        # Check distinctness
        for i in range(n):
            for j in range(i+1, n):
                if points[i] == points[j]:
                    return False
        
        # Check distances are integers (squared distances are perfect squares)
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx*dx + dy*dy
                s = int(math.isqrt(d2))
                if s*s != d2:
                    return False
        
        # Check no three collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    # Area of triangle
                    area2 = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
                    if area2 == 0:
                        return False
        
        # Check no four concyclic
        # Four points are concyclic if the determinant of the circle equation is 0
        # x^2 + y^2 + Dx + Ey + F = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        p1 = points[i]
                        p2 = points[j]
                        p3 = points[k]
                        p4 = points[l]
                        
                        # Determinant of:
                        # x1^2+y1^2  x1  y1  1
                        # x2^2+y2^2  x2  y2  1
                        # x3^2+y3^2  x3  y3  1
                        # x4^2+y4^2  x4  y4  1
                        # Using expansion for 4x4 determinant
                        a1 = p1[0]**2 + p1[1]**2
                        a2 = p2[0]**2 + p2[1]**2
                        a3 = p3[0]**2 + p3[1]**2
                        a4 = p4[0]**2 + p4[1]**2
                        
                        x1, y1 = p1
                        x2, y2 = p2
                        x3, y3 = p3
                        x4, y4 = p4
                        
                        # Det = a1*(x2*y3*1 - x2*y4*1 - y2*x3*1 + y2*x4*1 + 1*x3*y4 - 1*x4*y3)
                        #     - x1*(a2*y3*1 - a2*y4*1 - y2*a3*1 + y2*a4*1 + 1*a3*y4 - 1*a4*y3)
                        #     + y1*(a2*x3*1 - a2*x4*1 - x2*a3*1 + x2*a4*1 + 1*a3*x4 - 1*a4*x3)
                        #     - 1*(a2*x3*y4 - a2*x4*y3 - x2*a3*y4 + x2*a4*y3 + y2*a3*x4 - y2*a4*x3)
                        
                        term1 = a1 * (x2*y3 - x2*y4 - y2*x3 + y2*x4 + x3*y4 - x4*y3)
                        term2 = -x1 * (a2*y3 - a2*y4 - y2*a3 + y2*a4 + a3*y4 - a4*y3)
                        term3 = y1 * (a2*x3 - a2*x4 - x2*a3 + x2*a4 + a3*x4 - a4*x3)
                        term4 = -1 * (a2*x3*y4 - a2*x4*y3 - x2*a3*y4 + x2*a4*y3 + y2*a3*x4 - y2*a4*x3)
                        
                        if term1 + term2 + term3 + term4 == 0:
                            return False
        
        return True

    # Try to find a 6th point to add to base_points
    # Search in a box around the base points
    # Base points range: x in [-56, 16], y in [0, 63]
    # Expand search range
    x_min, x_max = -100, 100
    y_min, y_max = -100, 100
    
    # To speed up, we can check distances to the first 5 points
    def check_candidate(x, y):
        for px, py in base_points:
            dx = x - px
            dy = y - py
            d2 = dx*dx + dy*dy
            s = int(math.isqrt(d2))
            if s*s != d2:
                return False
        return True

    found_6 = None
    # Iterate over a subset of the grid to avoid timeout
    # Step size 1 might be too slow for 200x200 = 40000 points * 5 checks = 200k ops. This is fine.
    # But we also need to check collinearity/concyclicity for the full 6-point set.
    # That is O(6^4) = 1296 determinants. Also fine.
    
    # To be safe, limit iterations
    max_checks = 50000
    checks = 0
    
    for x in range(x_min, x_max+1):
        if time.time() > deadline:
            break
        for y in range(y_min, y_max+1):
            if time.time() > deadline:
                break
            checks += 1
            if checks > max_checks:
                break
            
            # Skip if already in base_points
            if (x, y) in base_points:
                continue
                
            if check_candidate(x, y):
                candidate_set = base_points + [(x, y)]
                if is_valid(candidate_set):
                    found_6 = candidate_set
                    break
        if found_6:
            break
    
    if found_6:
        points = found_6
    else:
        points = base_points
    
    # Format output
    # D=1, so y*sqrt(1) = y
    # Points are (x, y)
    # Return format: {'D': 1, 'points': [[[x_num, x_den], [y_num, y_den]], ...]}
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}

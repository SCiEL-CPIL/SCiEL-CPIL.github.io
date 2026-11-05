import time
import math
from fractions import Fraction

def _is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def _dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def _check_rational_distances(points):
    # points are tuples of (x, y) where x,y are Fractions or ints
    # For now, assume ints for speed in grid search
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            d2 = _dist_sq(points[i], points[j])
            if not _is_square(d2):
                return False
    return True

def _check_general_position(points):
    # No 3 collinear
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Collinearity check: (x2-x1)(y3-y1) - (y2-y1)(x3-x1) == 0
                if (p2[0]-p1[0])*(p3[1]-p1[1]) == (p2[1]-p1[1])*(p3[0]-p1[0]):
                    return False
    # No 4 concyclic
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    # Circle through p1, p2, p3
                    # x^2 + y^2 + Dx + Ey + F = 0
                    # Subtract to eliminate x^2+y^2
                    # (x2-x1)x + (y2-y1)y = (x2^2-y2^2 - x1^2+y1^2)/2 ... this is messy with fractions
                    # Use determinant of matrix:
                    # | x^2+y^2, x, y, 1 |
                    # | x1^2+y1^2, x1, y1, 1 |
                    # | x2^2+y2^2, x2, y2, 1 |
                    # | x3^2+y3^2, x3, y3, 1 |
                    # If det == 0, then p1,p2,p3,p4 concyclic? No, this checks if p1,p2,p3 define a circle passing through origin?
                    # Standard check:
                    # A = x1^2+y1^2, B = x2^2+y2^2, C = x3^2+y3^2
                    # Matrix M:
                    # [x1, y1, A, 1]
                    # [x2, y2, B, 1]
                    # [x3, y3, C, 1]
                    # [x4, y4, D, 1]
                    # Det(M) == 0 implies concyclic.
                    x1, y1 = p1
                    x2, y2 = p2
                    x3, y3 = p3
                    x4, y4 = p4
                    A = x1*x1 + y1*y1
                    B = x2*x2 + y2*y2
                    C = x3*x3 + y3*y3
                    D = x4*x4 + y4*y4
                    
                    # Calculate determinant of 4x4 matrix
                    # | x1 y1 A 1 |
                    # | x2 y2 B 1 |
                    # | x3 y3 C 1 |
                    # | x4 y4 D 1 |
                    # Use expansion or specific formula
                    # For speed, we can use the property that 4 points are concyclic iff the cross ratio is real? No.
                    # Let's use the determinant.
                    # To avoid huge numbers, we can check if p4 lies on the circle defined by p1,p2,p3.
                    # If p1,p2,p3 are collinear, they don't define a circle. But we already checked no 3 collinear.
                    # Equation of circle: x^2+y^2+Dx+Ey+F=0
                    # D, E, F can be found by solving linear system.
                    # (x1-x4)^2 + (y1-y4)^2 ... no.
                    # Let's solve for D, E, F using p1, p2, p3.
                    # x1^2+y1^2 + D x1 + E y1 + F = 0
                    # x2^2+y2^2 + D x2 + E y2 + F = 0
                    # x3^2+y3^2 + D x3 + E y3 + F = 0
                    # Subtract 1 from 2: (x2^2-x1^2) + (y2^2-y1^2) + D(x2-x1) + E(y2-y1) = 0
                    # (x2-x1)(x2+x1) + (y2-y1)(y2+y1) + D(x2-x1) + E(y2-y1) = 0
                    # Let dx12 = x2-x1, dy12 = y2-y1
                    # D dx12 + E dy12 = - (dx12(x1+x2) + dy12(y1+y2))
                    # Similarly for 1,3.
                    # Solve 2x2 system for D, E.
                    dx12 = x2 - x1
                    dy12 = y2 - y1
                    dx13 = x3 - x1
                    dy13 = y3 - y1
                    
                    rhs1 = - (dx12 * (x1 + x2) + dy12 * (y1 + y2))
                    rhs2 = - (dx13 * (x1 + x3) + dy13 * (y1 + y3))
                    
                    det = dx12 * dy13 - dx13 * dy12
                    if det == 0:
                        return False # Collinear, should have been caught, but just in case
                    D_circ = (rhs1 * dy13 - rhs2 * dy12) / det
                    E_circ = (dx12 * rhs2 - dx13 * rhs1) / det
                    F_circ = - (x1*x1 + y1*y1 + D_circ * x1 + E_circ * y1)
                    
                    # Check if p4 satisfies
                    val = x4*x4 + y4*y4 + D_circ * x4 + E_circ * y4 + F_circ
                    if val == 0:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Seed 5 points
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_set = seed_points
    
    # Try to find a 6th point
    # Grid search around the centroid
    cx = sum(p[0] for p in seed_points) // 5
    cy = sum(p[1] for p in seed_points) // 5
    
    found_6 = False
    for radius in range(10, 200, 10):
        if time.time() > deadline:
            break
        for dx in range(-radius, radius+1):
            if time.time() > deadline:
                break
            for dy in range(-radius, radius+1):
                p6 = (cx + dx, cy + dy)
                if p6 in seed_points:
                    continue
                candidate = seed_points + [p6]
                if _check_rational_distances(candidate):
                    if _check_general_position(candidate):
                        best_set = candidate
                        found_6 = True
                        break
            if found_6:
                break
        if found_6:
            break
            
    # If found 6, try to find 7?
    # Given time constraints, finding 6 is a good improvement.
    # Let's try a small search for 7 if 6 was found quickly.
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}

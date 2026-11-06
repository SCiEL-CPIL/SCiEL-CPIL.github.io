import time
import math
from fractions import Fraction

def is_square(n):
    if n < 0:
        return False
    if n == 0:
        return True
    r = int(math.isqrt(n))
    return r*r == n

def dist_sq_rational(p1, p2):
    # p1, p2 are (x_num, x_den, y_num, y_den) for x, y*sqrt(D)
    # Here we assume D=1 for simplicity in the search, so points are rational.
    # If D=1, points are (x, y) with rational x, y.
    # Distance squared is (x1-x2)^2 + (y1-y2)^2.
    # We need this to be a rational square.
    pass

# Since the seed uses D=1, we can search for integer points first.
# Integer points with integer distances are a subset of rational points with rational distances.

def check_validity(points):
    # points: list of (x, y) integers
    n = len(points)
    # Check pairwise distances
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d2 = dx*dx + dy*dy
            if not is_square(d2):
                return False
    # Check no 3 collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                if (x2-x1)*(y3-y1) == (x3-x1)*(y2-y1):
                    return False
    # Check no 4 concyclic
    # Equation of circle: x^2 + y^2 + Dx + Ey + F = 0
    # For 4 points, the determinant of the matrix formed by [x^2+y^2, x, y, 1] must be non-zero.
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    # Using determinant of 4x4 matrix
                    # | x1^2+y1^2, x1, y1, 1 |
                    # | x2^2+y2^2, x2, y2, 1 |
                    # | x3^2+y3^2, x3, y3, 1 |
                    # | x4^2+y4^2, x4, y4, 1 |
                    # We can use the property that 4 points are concyclic iff the determinant is 0.
                    # To avoid large numbers, we can use floating point with tolerance, or exact rational arithmetic.
                    # Given small coordinates, exact rational arithmetic is feasible.
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    # Translate to make p1 origin? No, just compute det.
                    # Det = 0
                    # Using Sarrus rule or expansion.
                    # For speed, we can check if the circle through p1,p2,p3 passes through p4.
                    # Circle through p1,p2,p3:
                    # If p1,p2,p3 are collinear, they don't define a circle, but we already checked collinearity.
                    # So circle exists.
                    # Equation: x^2+y^2+Dx+Ey+F=0
                    # D, E, F can be found by solving linear system.
                    # Alternatively, use the determinant formula.
                    x1, y1 = p1
                    x2, y2 = p2
                    x3, y3 = p3
                    x4, y4 = p4
                    # Matrix M
                    # M[i][0] = xi^2+yi^2
                    # M[i][1] = xi
                    # M[i][2] = yi
                    # M[i][3] = 1
                    # Det(M) != 0
                    # We can compute Det(M) using exact arithmetic.
                    # Since coordinates are small, this is fast.
                    m = [
                        [x1*x1+y1*y1, x1, y1, 1],
                        [x2*x2+y2*y2, x2, y2, 1],
                        [x3*x3+y3*y3, x3, y3, 1],
                        [x4*x4+y4*y4, x4, y4, 1]
                    ]
                    # Calculate determinant
                    # 4x4 determinant
                    det = 0
                    # Expansion along first row
                    # a11 * det(M11) - a12 * det(M12) + a13 * det(M13) - a14 * det(M14)
                    # M11 is 3x3
                    def det3(m3):
                        return (m3[0][0]*(m3[1][1]*m3[2][2]-m3[1][2]*m3[2][1]) -
                                m3[0][1]*(m3[1][0]*m3[2][2]-m3[1][2]*m3[2][0]) +
                                m3[0][2]*(m3[1][0]*m3[2][1]-m3[1][1]*m3[2][0]))
                    
                    # Submatrices
                    m11 = [[m[1][1], m[1][2], m[1][3]], [m[2][1], m[2][2], m[2][3]], [m[3][1], m[3][2], m[3][3]]]
                    m12 = [[m[1][0], m[1][2], m[1][3]], [m[2][0], m[2][2], m[2][3]], [m[3][0], m[3][2], m[3][3]]]
                    m13 = [[m[1][0], m[1][1], m[1][3]], [m[2][0], m[2][1], m[2][3]], [m[3][0], m[3][1], m[3][3]]]
                    m14 = [[m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]], [m[3][0], m[3][1], m[3][2]]]
                    
                    det = m[0][0]*det3(m11) - m[0][1]*det3(m12) + m[0][2]*det3(m13) - m[0][3]*det3(m14)
                    if det == 0:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Start with a known 5-point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to find a 6th point
    # Search space: x, y in [-200, 200]
    # To speed up, we can filter candidates that are at integer distance from all 5 base points.
    
    # Precompute squares
    max_coord = 200
    squares = set()
    for i in range(max_coord*2):
        squares.add(i*i)
        
    best_points = base_points
    
    # Heuristic: Randomly perturb or search nearby
    # Since 6 points are rare, we might need to search a larger space or use a different base.
    # Let's try to find a 6-point set from scratch using a greedy approach with backtracking.
    # Or simply try to add a point to the base set.
    
    # Candidate generation
    # For a point (x,y) to be at integer distance from (0,0), x^2+y^2 must be a square.
    # We can iterate through possible distances and find (x,y).
    
    # Let's try a specific known 6-point set if we can find one.
    # Actually, let's just run a search.
    
    # Search for a 6th point for the base set
    found_6 = False
    for x in range(-100, 101):
        if time.time() > deadline:
            break
        for y in range(-100, 101):
            if time.time() > deadline:
                break
            # Check distance to all base points
            valid = True
            for px, py in base_points:
                dx = x - px
                dy = y - py
                d2 = dx*dx + dy*dy
                if d2 not in squares:
                    valid = False
                    break
            if valid:
                # Check collinearity and concyclicity with new point
                new_points = base_points + [(x, y)]
                if check_validity(new_points):
                    best_points = new_points
                    found_6 = True
                    break
        if found_6:
            break
            
    if found_6:
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
    else:
        # If 6 not found, return the 5-point set
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}

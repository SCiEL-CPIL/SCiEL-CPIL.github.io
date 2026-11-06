import time
import math
from itertools import combinations

def is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def check_set(points, D):
    n = len(points)
    # Check distinct
    for i in range(n):
        for j in range(i + 1, n):
            if points[i] == points[j]:
                return False
            
    # Check distances and collinearity
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d2 = dx * dx + D * dy * dy
            if not is_square(d2):
                return False
            for k in range(n):
                if k != i and k != j:
                    # Check collinear i, j, k
                    dx1 = points[j][0] - points[i][0]
                    dy1 = points[j][1] - points[i][1]
                    dx2 = points[k][0] - points[i][0]
                    dy2 = points[k][1] - points[i][1]
                    if dx1 * dy2 - dy1 * dx2 == 0:
                        return False
    # Check concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    x1, y1 = p1
                    x2, y2 = p2
                    x3, y3 = p3
                    x4, y4 = p4
                    # Circle in (x, y*sqrt(D)) plane:
                    # X^2 + Y^2 + aX + bY + c = 0
                    # X = x, Y = y*sqrt(D)
                    # x^2 + D*y^2 + a*x + b*sqrt(D)*y + c = 0
                    # This form is tricky for exact arithmetic if b is irrational.
                    # However, 4 points are concyclic iff the determinant of
                    # | x_i^2 + D*y_i^2, x_i, y_i, 1 | is 0?
                    # No, the circle equation in terms of x,y (where Y=y*sqrt(D)) is:
                    # x^2 + D*y^2 + A*x + B*(y*sqrt(D)) + C = 0
                    # If we multiply by sqrt(D) or something?
                    # Actually, the condition for 4 points (x_i, y_i) to be concyclic in the metric $x^2+Dy^2$
                    # is that the determinant of the matrix:
                    # | x_i^2 + D*y_i^2, x_i, y_i, 1 | is NOT sufficient because the circle is not "standard".
                    # Wait, a circle in the plane with metric $dx^2 + D dy^2$ is an ellipse in the Euclidean plane?
                    # No, the points are in the Euclidean plane, but the distance is defined by $D$.
                    # The "circle" is the set of points at distance R from center.
                    # $(x-h)^2 + D(y-k)^2 = R^2$.
                    # This is an ellipse in the (x,y) coordinates.
                    # 4 points lie on such an ellipse if:
                    # $x^2 + D y^2 + A x + B y + C = 0$ has a solution for A,B,C.
                    # This is a linear system in A,B,C for each point.
                    # For 4 points, the system is overdetermined.
                    # The condition is that the determinant of:
                    # | x_i^2 + D*y_i^2, x_i, y_i, 1 | is 0.
                    # Let's verify.
                    # $x^2 + D y^2 + A x + B y + C = 0$
                    # This is a linear equation in parameters A, B, C for fixed x, y.
                    # If 4 points satisfy this, the rank of the coefficient matrix must be < 3?
                    # No, the matrix is:
                    # [ x1^2+Dy1^2, x1, y1, 1 ]
                    # [ x2^2+Dy2^2, x2, y2, 1 ]
                    # ...
                    # If the determinant is 0, there is a non-trivial solution for (A,B,C,1)?
                    # No, the last column is 1.
                    # The condition is that the 4x4 determinant is 0.
                    det = ( (x1*x1 + D*y1*y1) * (x2*(y3-y4) - y2*(x3-x4) + (x3*y4 - y3*x4))
                         - x1 * ( (x2*x2 + D*y2*y2)*(y3-y4) - y2*((x3*x3 + D*y3*y3)-(x4*x4 + D*y4*y4)) + ((x3*x3 + D*y3*y3)*y4 - (x4*x4 + D*y4*y4)*y3) )
                         + y1 * ( (x2*x2 + D*y2*y2)*(x3-x4) - x2*((x3*x3 + D*y3*y3)-(x4*x4 + D*y4*y4)) + ((x3*x3 + D*y3*y3)*x4 - (x4*x4 + D*y4*y4)*x3) )
                         - 1 * ( (x2*x2 + D*y2*y2)*(x3*y4 - y3*x4) - x2*((x3*x3 + D*y3*y3)*y4 - (x4*x4 + D*y4*y4)*y3) + y2*((x3*x3 + D*y3*y3)*x4 - (x4*x4 + D*y4*y4)*x3) ) )
                    if det == 0:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Fallback: 5-point set for D=1
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_points = base_points
    best_D = 1
    
    # Try to find 6 points for D=3
    D = 3
    candidates = []
    # Generate points (u^2-3v^2, 2uv)
    max_uv = 20
    for u in range(1, max_uv + 1):
        for v in range(0, max_uv + 1):
            x = u*u - 3*v*v
            y = 2*u*v
            candidates.append((x, y))
    
    # Add origin
    candidates.append((0, 0))
    
    # Remove duplicates
    candidates = list(set(candidates))
    
    # Filter: keep only points with small coordinates to speed up search
    # And ensure distance from origin is rational (it is by construction)
    
    # Search for 6-cliques
    # This is expensive. We will check subsets of candidates.
    # To make it feasible, we will only check a limited number of candidates.
    # We prioritize candidates with small coordinates.
    candidates.sort(key=lambda p: abs(p[0]) + abs(p[1]))
    candidates = candidates[:50]
    
    found_6 = None
    for combo in combinations(candidates, 6):
        if time.time() > deadline:
            break
        if check_set(combo, D):
            found_6 = combo
            break
    
    if found_6:
        best_points = found_6
        best_D = D
    else:
        # Try 5 points for D=3?
        # The 5-point set for D=1 is already good.
        # We can try to find a 5-point set for D=3 to see if it's better?
        # No, 5 is 5.
        pass

    return {'D': best_D, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}

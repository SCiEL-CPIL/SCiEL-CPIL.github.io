import math
from fractions import Fraction

def construct():
    # Five integer points with integer pairwise distances, no three collinear and no four concyclic
    # (found by a small local search over lattice points).
    points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verification logic (optional but good for safety)
    # Convert to Fractions for exact arithmetic
    pts = [[Fraction(x), Fraction(y)] for x, y in points]
    
    # Check distinctness
    if len(set(tuple(p) for p in pts)) != len(pts):
        return {'D': 1, 'points': [[[0, 1], [0, 1]]]} # Fallback, shouldn't happen
    
    # Check rational distances
    def dist_sq(p1, p2):
        return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    
    def is_rational_square(val):
        # val is Fraction
        num = val.numerator
        den = val.denominator
        # Check if num and den are perfect squares
        if num < 0: return False
        r_num = math.isqrt(num)
        if r_num * r_num != num: return False
        r_den = math.isqrt(den)
        if r_den * r_den != den: return False
        return True

    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            if not is_rational_square(dist_sq(pts[i], pts[j])):
                # Fallback to a smaller valid set if verification fails
                return {'D': 1, 'points': [[[0, 1], [0, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 1]]]}

    # Check no three collinear
    def are_collinear(p1, p2, p3):
        # Area of triangle is 0
        return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]) == 0

    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            for k in range(j+1, len(pts)):
                if are_collinear(pts[i], pts[j], pts[k]):
                    return {'D': 1, 'points': [[[0, 1], [0, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 1]]]}

    # Check no four concyclic
    def are_concyclic(p1, p2, p3, p4):
        # Determinant of circle equation
        # x^2+y^2+Dx+Ey+F=0
        # Matrix:
        # x y x^2+y^2 1
        # ...
        # If det is 0, they are concyclic
        def det4(m):
            # m is 4x4
            return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
                 - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
                 + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
                 - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])))

        m = []
        for p in [p1, p2, p3, p4]:
            m.append([p[0], p[1], p[0]**2 + p[1]**2, 1])
        return det4(m) == 0

    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            for k in range(j+1, len(pts)):
                for l in range(k+1, len(pts)):
                    if are_concyclic(pts[i], pts[j], pts[k], pts[l]):
                        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 1]]]}

    # Format output
    return {'D': 1, 'points': [[[int(p[0].numerator), int(p[0].denominator)], [int(p[1].numerator), int(p[1].denominator)]] for p in pts]}

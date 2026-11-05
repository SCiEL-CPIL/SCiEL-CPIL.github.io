import math
import time

def is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def collinear(p1, p2, p3):
    return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) == 0

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    # Seed 5 points
    points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Precompute circles for all triplets
    # Circle: x^2 + y^2 + Ax + By + C = 0
    # We store (A, B, C) for each triplet
    circles = []
    for i in range(5):
        for j in range(i+1, 5):
            for k in range(j+1, 5):
                p1, p2, p3 = points[i], points[j], points[k]
                # Solve for A, B, C
                # x1^2 + y1^2 + A x1 + B y1 + C = 0
                # x2^2 + y2^2 + A x2 + B y2 + C = 0
                # x3^2 + y3^2 + A x3 + B y3 + C = 0
                
                # Subtracting equations to eliminate C
                # A(x1-x2) + B(y1-y2) = x2^2 + y2^2 - x1^2 - y1^2
                # A(x1-x3) + B(y1-y3) = x3^2 + y3^2 - x1^2 - y1^2
                
                a11 = p1[0] - p2[0]
                a12 = p1[1] - p2[1]
                b1 = p2[0]**2 + p2[1]**2 - p1[0]**2 - p1[1]**2
                
                a21 = p1[0] - p3[0]
                a22 = p1[1] - p3[1]
                b2 = p3[0]**2 + p3[1]**2 - p1[0]**2 - p1[1]**2
                
                det = a11*a22 - a12*a21
                if det == 0:
                    # Collinear, no circle (or line)
                    # If collinear, they are not concyclic with any 4th point unless the 4th is also on the line?
                    # No, 3 collinear points define a line, not a circle.
                    # The condition "no four concyclic" implies we check if 4 points lie on a circle.
                    # If 3 are collinear, they don't define a unique circle.
                    # However, if a 4th point is added, and the 4 points are concyclic, then the 3 collinear points must be on the circle, which is impossible unless the circle is a line (infinite radius).
                    # So if 3 are collinear, they cannot be part of a concyclic 4-set (unless the 4th is also on the line, but then they are collinear, not concyclic in the strict sense? No, a line is a circle of infinite radius. But the problem says "no four concyclic", usually implying a proper circle).
                    # Actually, if 3 points are collinear, no circle passes through them. So they can never be part of a concyclic 4-set.
                    # So we can skip storing a circle for collinear triplets.
                    continue
                
                A = (b1*a22 - b2*a12) / det
                B = (a11*b2 - a21*b1) / det
                C = -(p1[0]**2 + p1[1]**2 + A*p1[0] + B*p1[1])
                
                # A, B, C are rational. We can store them as Fractions or just use the equation with cross-multiplication.
                # To avoid floating point, we can store the equation as:
                # det * (x^2 + y^2) + (b1*a22 - b2*a12) x + (a11*b2 - a21*b1) y - det * (p1[0]**2 + p1[1]**2 + A*p1[0] + B*p1[1]) = 0
                # Let's just store A, B, C as Fractions for clarity, or use the determinant form.
                # Using Fractions is safe and fast enough for 10 circles.
                from fractions import Fraction
                A_frac = Fraction(b1*a22 - b2*a12, det)
                B_frac = Fraction(a11*b2 - a21*b1, det)
                C_frac = Fraction(-(p1[0]**2 + p1[1]**2 + A_frac*p1[0] + B_frac*p1[1]), 1)
                
                circles.append((A_frac, B_frac, C_frac))
                
    # Search for 6th point
    # Expand range
    found = False
    # We can try a few different ranges or a single large range
    # Let's try a range that covers likely candidates
    # The seed points are within [-56, 63].
    # Let's search [-200, 200] first, then expand if time permits.
    
    ranges = [
        (-100, 100),
        (-200, 200),
        (-500, 500),
        (-1000, 1000)
    ]
    
    for r_low, r_high in ranges:
        if time.time() > deadline:
            break
        for x in range(r_low, r_high + 1):
            if time.time() > deadline:
                break
            for y in range(r_low, r_high + 1):
                p_new = (x, y)
                if p_new in points:
                    continue
                
                # Check distances
                ok = True
                for p in points:
                    if not is_square(dist_sq(p_new, p)):
                        ok = False
                        break
                if not ok:
                    continue
                
                # Check collinear
                for i in range(len(points)):
                    for j in range(i+1, len(points)):
                        if collinear(points[i], points[j], p_new):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                
                # Check concyclic
                # Check if p_new lies on any of the precomputed circles
                for A, B, C in circles:
                    # x^2 + y^2 + Ax + By + C = 0
                    val = x*x + y*y + A*x + B*y + C
                    if val == 0:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Valid 6th point
                points.append(p_new)
                found = True
                break
            if found:
                break
        
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}

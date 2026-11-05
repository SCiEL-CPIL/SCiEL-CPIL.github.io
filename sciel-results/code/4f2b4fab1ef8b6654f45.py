import random
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 100.0  # Leave 20s margin for 120s limit
    
    def is_square(n):
        if n < 0:
            return False
        r = int(n**0.5)
        return r*r == n or (r+1)*(r+1) == n

    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def is_rational_dist(p1, p2):
        return is_square(dist_sq(p1, p2))

    def are_collinear(p1, p2, p3):
        # (x2-x1)(y3-y1) - (y2-y1)(x3-x1) == 0
        return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) == 0

    def are_concyclic(p1, p2, p3, p4):
        # Determinant check
        # | x1^2+y1^2, x1, y1, 1 |
        # | x2^2+y2^2, x2, y2, 1 |
        # | x3^2+y3^2, x3, y3, 1 |
        # | x4^2+y4^2, x4, y4, 1 |
        # Using fractions to be safe, but integers are fine for D=1
        # We can use the property that 4 points are concyclic iff the power of a point is consistent
        # Or just calculate the circle through 3 and check the 4th.
        
        # Circle through p1, p2, p3
        # x^2 + y^2 + Dx + Ey + F = 0
        # System:
        # x1^2+y1^2 + D x1 + E y1 + F = 0
        # x2^2+y2^2 + D x2 + E y2 + F = 0
        # x3^2+y3^2 + D x3 + E y3 + F = 0
        
        # Let's use the determinant method with integer arithmetic to avoid floating point errors
        # D = 0 iff concyclic
        # We can compute the determinant of a 4x4 matrix.
        # For performance, we can check if p4 lies on the circle defined by p1,p2,p3.
        
        # Equation of circle:
        # A x^2 + A y^2 + B x + C y + D = 0
        # Since p1, p2, p3 are on it:
        # A(x1^2+y1^2) + B x1 + C y1 + D = 0
        # ...
        
        # Let's solve for A, B, C, D (homogeneous)
        # This is a bit heavy.
        # Alternative:
        # The points are concyclic iff the cross ratio is real? No.
        # Let's use the determinant.
        
        M = [
            [p1[0]*p1[0] + p1[1]*p1[1], p1[0], p1[1], 1],
            [p2[0]*p2[0] + p2[1]*p2[1], p2[0], p2[1], 1],
            [p3[0]*p3[0] + p3[1]*p3[1], p3[0], p3[1], 1],
            [p4[0]*p4[0] + p4[1]*p4[1], p4[0], p4[1], 1]
        ]
        
        # Calculate determinant of 4x4 matrix
        def det4(m):
            # Sarrus-like or cofactor expansion
            # Expansion along first row
            d = 0
            for j in range(4):
                sub = [[row[i] for i in range(4) if i != j] for row in m[1:]]
                # det3
                det3_val = (sub[0][0]*(sub[1][1]*sub[2][2]-sub[1][2]*sub[2][1]) -
                             sub[0][1]*(sub[1][0]*sub[2][2]-sub[1][2]*sub[2][0]) +
                             sub[0][2]*(sub[1][0]*sub[2][1]-sub[1][1]*sub[2][0]))
                d += ((-1)**j) * m[0][j] * det3_val
            return d

        return det4(M) == 0

    def is_valid_set(points):
        n = len(points)
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

    best_points = None
    best_size = 0
    
    # Generate Pythagorean triples for candidate distances
    # We want integer points.
    # Candidates for a point P are P + (dx, dy) where dx^2+dy^2 is a square.
    # Precompute a list of (dx, dy) vectors for small distances.
    max_coord = 50
    vectors = []
    for dx in range(-max_coord, max_coord+1):
        for dy in range(-max_coord, max_coord+1):
            if dx == 0 and dy == 0:
                continue
            if is_square(dx*dx + dy*dy):
                vectors.append((dx, dy))
    
    # Sort vectors by distance to prefer closer points? Maybe.
    # vectors.sort(key=lambda v: v[0]*v[0]+v[1]*v[1])
    
    def try_add_point(current_points, max_iter=1000):
        # Try to add a new point
        for _ in range(max_iter):
            # Pick a random existing point
            p_base = random.choice(current_points)
            # Pick a random vector
            dx, dy = random.choice(vectors)
            p_new = (p_base[0] + dx, p_base[1] + dy)
            
            # Check if p_new is distinct
            if p_new in current_points:
                continue
            
            # Check distances to all existing points
            valid_dist = True
            for p in current_points:
                if not is_rational_dist(p, p_new):
                    valid_dist = False
                    break
            if not valid_dist:
                continue
            
            # Check collinearity with all pairs
            valid_collinear = True
            for i in range(len(current_points)):
                for j in range(i+1, len(current_points)):
                    if are_collinear(current_points[i], current_points[j], p_new):
                        valid_collinear = False
                        break
                if not valid_collinear:
                    break
            if not valid_collinear:
                continue
            
            # Check concyclicity with all triples
            valid_concyclic = True
            n = len(current_points)
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if are_concyclic(current_points[i], current_points[j], current_points[k], p_new):
                            valid_concyclic = False
                            break
                    if not valid_concyclic:
                        break
                if not valid_concyclic:
                    break
            if not valid_concyclic:
                continue
            
            return p_new
        return None

    # Main search loop
    while time.time() < deadline:
        # Start with 3 random points
        # Ensure they form a valid triangle (non-collinear, rational distances)
        pts = []
        # Point 1
        p1 = (0, 0)
        pts.append(p1)
        
        # Point 2: Must be at integer distance from p1
        while True:
            dx, dy = random.choice(vectors)
            p2 = (p1[0] + dx, p1[1] + dy)
            if p2 != p1:
                break
        pts.append(p2)
        
        # Point 3: Must be at integer distance from p1 and p2, and not collinear
        while True:
            # Pick base
            base = random.choice(pts)
            dx, dy = random.choice(vectors)
            p3 = (base[0] + dx, base[1] + dy)
            if p3 in pts:
                continue
            valid = True
            for p in pts:
                if not is_rational_dist(p, p3):
                    valid = False
                    break
            if valid and not are_collinear(pts[0], pts[1], p3):
                pts.append(p3)
                break
        
        # Now try to extend
        current = pts
        while len(current) < 12:
            p_new = try_add_point(current, max_iter=500)
            if p_new is None:
                break
            current.append(p_new)
        
        if len(current) > best_size:
            best_size = len(current)
            best_points = current
            if best_size == 12:
                break

    if best_points is None:
        # Fallback: return the seed's 5 points
        points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}

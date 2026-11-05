import random
import time
import math
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for 150s limit
    
    def is_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy

    def is_rational_dist(p1, p2):
        return is_square(dist_sq(p1, p2))

    def are_collinear(p1, p2, p3):
        # (x2-x1)(y3-y1) - (y2-y1)(x3-x1) == 0
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

    def are_concyclic(p1, p2, p3, p4):
        # Four points are concyclic iff the determinant of this 4x4 matrix is zero:
        # | x1^2+y1^2, x1, y1, 1 |
        # | x2^2+y2^2, x2, y2, 1 |
        # | x3^2+y3^2, x3, y3, 1 |
        # | x4^2+y4^2, x4, y4, 1 |
        
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        s1 = x1*x1 + y1*y1
        s2 = x2*x2 + y2*y2
        s3 = x3*x3 + y3*y3
        s4 = x4*x4 + y4*y4
        
        # Compute determinant by cofactor expansion along first row
        # det = s1 * det3(x2,y2,1; x3,y3,1; x4,y4,1) 
        #       - x1 * det3(s2,y2,1; s3,y3,1; s4,y4,1)
        #       + y1 * det3(s2,x2,1; s3,x3,1; s4,x4,1)
        #       - 1 * det3(s2,x2,y2; s3,x3,y3; s4,x4,y4)
        
        # det3(a,b,c; d,e,f; g,h,i) = a(ei-fh) - b(di-fg) + c(dh-eg)
        
        d1 = x2*(y3 - y4) - y2*(x3 - x4) + (x3*y4 - x4*y3)
        d2 = s2*(y3 - y4) - y2*(s3 - s4) + (s3*y4 - s4*y3)
        d3 = s2*(x3 - x4) - x2*(s3 - s4) + (s3*x4 - s4*x3)
        d4 = s2*(x3*y4 - x4*y3) - x2*(s3*y4 - s4*y3) + y2*(s3*x4 - s4*x3)
        
        return s1 * d1 - x1 * d2 + y1 * d3 - d4 == 0

    # Precompute all valid displacement vectors
    max_coord = 120
    vectors = []
    for dx in range(-max_coord, max_coord + 1):
        for dy in range(-max_coord, max_coord + 1):
            if dx == 0 and dy == 0:
                continue
            if is_square(dx * dx + dy * dy):
                vectors.append((dx, dy))
    
    # Sort vectors by distance squared, prefer closer points
    vectors.sort(key=lambda v: v[0]*v[0] + v[1]*v[1])
    
    # Precompute distance lookup for speed
    dist_lookup = {}
    for v in vectors:
        dist_lookup[(v[0], v[1])] = v[0]*v[0] + v[1]*v[1]
        dist_lookup[(-v[0], -v[1])] = v[0]*v[0] + v[1]*v[1]
    
    def try_add_point(current_points, max_iter=3000):
        n = len(current_points)
        point_set = set(current_points)
        
        for _ in range(max_iter):
            # Pick a random existing point as base
            idx = random.randint(0, n - 1)
            p_base = current_points[idx]
            # Pick a random vector
            dx, dy = random.choice(vectors)
            p_new = (p_base[0] + dx, p_base[1] + dy)
            
            # Check if p_new is distinct
            if p_new in point_set:
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
            for i in range(n):
                for j in range(i + 1, n):
                    if are_collinear(current_points[i], current_points[j], p_new):
                        valid_collinear = False
                        break
                if not valid_collinear:
                    break
            if not valid_collinear:
                continue
            
            # Check concyclicity with all triples
            valid_concyclic = True
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
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

    def try_add_point_smart(current_points, max_iter=5000):
        """Try to add a point using a smarter strategy: try vectors from multiple base points."""
        n = len(current_points)
        point_set = set(current_points)
        
        # Try each base point multiple times
        for base_idx in range(n):
            if time.time() > deadline - 5:
                break
            p_base = current_points[base_idx]
            for _ in range(max_iter // n):
                if time.time() > deadline - 5:
                    break
                dx, dy = random.choice(vectors)
                p_new = (p_base[0] + dx, p_base[1] + dy)
                
                if p_new in point_set:
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
                for i in range(n):
                    for j in range(i + 1, n):
                        if are_collinear(current_points[i], current_points[j], p_new):
                            valid_collinear = False
                            break
                    if not valid_collinear:
                        break
                if not valid_collinear:
                    continue
                
                # Check concyclicity with all triples
                valid_concyclic = True
                for i in range(n):
                    for j in range(i + 1, n):
                        for k in range(j + 1, n):
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

    best_points = None
    best_size = 0
    
    # Known valid configuration with 5 points
    known_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_points = known_5[:]
    best_size = 5
    
    # Try to extend the known 5-point configuration first
    current = known_5[:]
    while len(current) < 12 and time.time() < deadline - 30:
        p_new = try_add_point_smart(current, max_iter=2000)
        if p_new is None:
            break
        current.append(p_new)
    
    if len(current) > best_size:
        best_size = len(current)
        best_points = current[:]
    
    # Main search loop with multiple strategies
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Strategy 1: Start from known 5-point config and extend
        if attempts % 3 == 1:
            current = known_5[:]
            while len(current) < 12 and time.time() < deadline - 10:
                p_new = try_add_point_smart(current, max_iter=1000)
                if p_new is None:
                    break
                current.append(p_new)
        
        # Strategy 2: Start with 3 random points
        elif attempts % 3 == 2:
            pts = []
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
            found_p3 = False
            for _ in range(5000):
                if time.time() > deadline - 10:
                    break
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
                    found_p3 = True
                    break
            
            if not found_p3:
                continue
            current = pts[:]
        
        # Strategy 3: Start with a different known configuration
        else:
            # Try a different starting configuration
            alt_configs = [
                [(0, 0), (3, 4), (0, 5), (3, 9), (6, 4)],
                [(0, 0), (5, 12), (0, 13), (5, 25), (10, 12)],
                [(0, 0), (8, 15), (0, 17), (8, 32), (16, 15)],
            ]
            config = random.choice(alt_configs)
            # Verify the config is valid
            valid_config = True
            for i in range(len(config)):
                for j in range(i+1, len(config)):
                    if not is_rational_dist(config[i], config[j]):
                        valid_config = False
                        break
                if not valid_config:
                    break
            if not valid_config:
                current = known_5[:]
            else:
                current = config[:]
        
        # Now try to extend
        while len(current) < 12 and time.time() < deadline - 5:
            p_new = try_add_point_smart(current, max_iter=800)
            if p_new is None:
                # Try the original method as fallback
                p_new = try_add_point(current, max_iter=500)
                if p_new is None:
                    break
            current.append(p_new)
        
        if len(current) > best_size:
            best_size = len(current)
            best_points = current[:]
            if best_size >= 10:
                break

    if best_points is None:
        points = known_5
    else:
        points = best_points
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}

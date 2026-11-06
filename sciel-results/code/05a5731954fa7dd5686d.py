import math
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 110  # seconds, safe margin from 120s limit (evaluator kills at 150s)
    
    def check_time():
        return time.time() - start_time < deadline

    def gcd(a, b):
        a, b = abs(a), abs(b)
        while b:
            a, b = b, a % b
        return a

    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    # D = 3, points are (x, y*sqrt(3))
    # Distance squared between (x1, y1*sqrt(3)) and (x2, y2*sqrt(3)):
    # dx^2 + 3*dy^2
    # This must be a perfect square for rational distance.
    
    def dist_sq_rational(x1, y1, x2, y2):
        """Check if distance squared is a perfect square (integer coords in Q(sqrt(3)))"""
        dx = x1 - x2
        dy = y1 - y2
        d2 = dx*dx + 3*dy*dy
        return is_square(d2)

    def collinear(x1, y1, x2, y2, x3, y3):
        """Check if three points (xi, yi*sqrt(3)) are collinear"""
        # (x2-x1)(y3-y1)*sqrt(3) - (x3-x1)(y2-y1)*sqrt(3) == 0
        # => (x2-x1)(y3-y1) - (x3-x1)(y2-y1) == 0
        return (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) == 0

    def concyclic(x1, y1, x2, y2, x3, y3, x4, y4):
        """Check if four points are concyclic in Q(sqrt(3))"""
        # Circle equation: x^2 + 3y^2 + Ax + By + C = 0
        # For point (xi, yi): xi^2 + 3yi^2 + Axi + Byi + C = 0
        # Solve for A, B, C using first 3 points, check 4th.
        
        # System:
        # x1*A + y1*B = -(x1^2 + 3*y1^2) - C
        # x2*A + y2*B = -(x2^2 + 3*y2^2) - C
        # x3*A + y3*B = -(x3^2 + 3*y3^2) - C
        
        # Subtract eq1 from eq2 and eq3:
        # (x2-x1)*A + (y2-y1)*B = -(x2^2+3*y2^2) + (x1^2+3*y1^2)
        # (x3-x1)*A + (y3-y1)*B = -(x3^2+3*y3^2) + (x1^2+3*y1^2)
        
        a1 = x2 - x1
        b1 = y2 - y1
        c1 = (x1*x1 + 3*y1*y1) - (x2*x2 + 3*y2*y2)
        
        a2 = x3 - x1
        b2 = y3 - y1
        c2 = (x1*x1 + 3*y1*y1) - (x3*x3 + 3*y3*y3)
        
        det = a1*b2 - a2*b1
        if det == 0:
            # Collinear, which shouldn't happen if we check collinearity first
            # But if they are collinear, any 4th point on same line would be "concyclic"
            # in extended sense. Return True to be safe (we want to AVOID this).
            return True
        
        A_num = c1*b2 - c2*b1
        B_num = a1*c2 - a2*c1
        
        # A = A_num / det, B = B_num / det
        # C = -(x1^2 + 3*y1^2) - A*x1 - B*y1
        # Check point 4:
        # x4^2 + 3*y4^2 + A*x4 + B*y4 + C = 0
        # Multiply by det:
        # det*(x4^2 + 3*y4^2) + A_num*x4 + B_num*y4 + det*C = 0
        # C = -(x1^2 + 3*y1^2)*det - A_num*x1 - B_num*y1 all over det
        # So: det*(x4^2 + 3*y4^2) + A_num*x4 + B_num*y4 - (x1^2 + 3*y1^2)*det - A_num*x1 - B_num*y1 = 0
        
        val = (det*(x4*x4 + 3*y4*y4) 
               + A_num*x4 
               + B_num*y4 
               - (x1*x1 + 3*y1*y1)*det 
               - A_num*x1 
               - B_num*y1)
        
        return val == 0

    def validate_set(points):
        """points: list of (x, y) integers, representing (x, y*sqrt(3))"""
        n = len(points)
        # Check distinct
        if len(set(points)) != n:
            return False
        # Check all pairwise distances rational (perfect square)
        for i in range(n):
            for j in range(i+1, n):
                if not dist_sq_rational(points[i][0], points[i][1], 
                                        points[j][0], points[j][1]):
                    return False
        # Check no 3 collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if collinear(points[i][0], points[i][1],
                                 points[j][0], points[j][1],
                                 points[k][0], points[k][1]):
                        return False
        # Check no 4 concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if concyclic(points[i][0], points[i][1],
                                     points[j][0], points[j][1],
                                     points[k][0], points[k][1],
                                     points[l][0], points[l][1]):
                            return False
        return True

    # Strategy: Try regular hexagon in Q(sqrt(3))
    # Regular hexagon with side length s centered at origin:
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # In Q(sqrt(3)), we can place vertices at:
    # (s, 0), (s/2, s*sqrt(3)/2), ... but we need rational coords in Q(sqrt(3))
    # Point form: (x, y*sqrt(3))
    # So (s/2, s/2) represents (s/2, (s/2)*sqrt(3))
    
    # Let's try side length 2 for simplicity:
    # (2, 0), (1, 1), (-1, 1), (-2, 0), (-1, -1), (1, -1)
    # Check distances:
    # (2,0) to (1,1): dx=1, dy=1, d2 = 1 + 3*1 = 4, sqrt=2 ✓
    # (1,1) to (-1,1): dx=2, dy=0, d2 = 4, sqrt=2 ✓
    # (1,1) to (-1,-1): dx=2, dy=2, d2 = 4 + 3*4 = 16, sqrt=4 ✓
    # (2,0) to (-1,-1): dx=3, dy=1, d2 = 9 + 3 = 12, not square ✗
    
    # Try side length 4:
    # (4, 0), (2, 2), (-2, 2), (-4, 0), (-2, -2), (2, -2)
    # (4,0) to (2,2): dx=2, dy=2, d2 = 4 + 12 = 16, sqrt=4 ✓
    # (2,2) to (-2,2): dx=4, dy=0, d2 = 16, sqrt=4 ✓
    # (2,2) to (-2,-2): dx=4, dy=4, d2 = 16 + 48 = 64, sqrt=8 ✓
    # (4,0) to (-2,-2): dx=6, dy=2, d2 = 36 + 12 = 48, not square ✗
    
    # The issue is that in a regular hexagon, not all pairwise distances are equal.
    # Distances are: side, 2*side, side*sqrt(3), etc.
    # In Q(sqrt(3)), side*sqrt(3) needs to be rational, which it is if side is rational.
    # But we need d2 to be a perfect square.
    
    # Let me reconsider. The distance squared is dx^2 + 3*dy^2.
    # For a regular hexagon with "radius" r (distance from center to vertex):
    # Vertices: (r, 0), (r/2, r*sqrt(3)/2), (-r/2, r*sqrt(3)/2), (-r, 0), (-r/2, -r*sqrt(3)/2), (r/2, -r*sqrt(3)/2)
    # In our representation (x, y*sqrt(3)):
    # (r, 0), (r/2, r/2), (-r/2, r/2), (-r, 0), (-r/2, -r/2), (r/2, -r/2)
    
    # For r=2: (2,0), (1,1), (-1,1), (-2,0), (-1,-1), (1,-1)
    # Pairwise d2 values:
    # Adjacent: dx=1, dy=1 => 1+3=4 ✓ (dist=2)
    # Skip one: dx=2, dy=0 => 4 ✓ (dist=2)
    # Opposite: dx=2, dy=2 => 4+12=16 ✓ (dist=4)
    # (2,0) to (-1,-1): dx=3, dy=1 => 9+3=12 ✗
    
    # So regular hexagon doesn't work directly because some distances have d2=12.
    
    # Alternative: Use a known valid configuration. Let's search computationally.
    
    # Grid search for 6 points with D=3
    # Points (x, y) where x, y are integers, and all pairwise dx^2+3*dy^2 is a perfect square.
    
    # Precompute valid distance pairs
    # For a point at origin, valid (dx, dy) such that dx^2 + 3*dy^2 is a perfect square.
    
    def find_valid_displacements(limit):
        """Find all (dx, dy) with |dx|,|dy| <= limit such that dx^2+3*dy^2 is a perfect square"""
        valid = []
        for dy in range(-limit, limit+1):
            for dx in range(-limit, limit+1):
                if dx == 0 and dy == 0:
                    continue
                d2 = dx*dx + 3*dy*dy
                if is_square(d2):
                    valid.append((dx, dy))
        return valid

    # Search for 6-point configuration
    # Start with (0,0) and greedily add points
    
    def search_6_points(limit=20):
        """Search for 6 points with D=3 in grid [-limit, limit]"""
        if not check_time():
            return None
        
        # Anchor at (0,0)
        points = [(0, 0)]
        
        # Find all valid displacements from origin
        valid_from_origin = []
        for dy in range(-limit, limit+1):
            for dx in range(-limit, limit+1):
                if dx == 0 and dy == 0:
                    continue
                d2 = dx*dx + 3*dy*dy
                if is_square(d2):
                    valid_from_origin.append((dx, dy))
        
        # Sort by some heuristic - prefer smaller coordinates
        valid_from_origin.sort(key=lambda p: (abs(p[0]) + abs(p[1]), p[0], p[1]))
        
        # Try to build a set of 6
        # Use backtracking with pruning
        
        def backtrack(current_points, remaining_needed, candidates):
            if not check_time():
                return None
            
            if remaining_needed == 0:
                return current_points[:]
            
            for idx, (dx, dy) in enumerate(candidates):
                new_point = (current_points[0][0] + dx, current_points[0][1] + dy)
                
                # Check bounds
                if abs(new_point[0]) > limit or abs(new_point[1]) > limit:
                    continue
                
                # Check not already in set
                if new_point in current_points:
                    continue
                
                # Check distance to all existing points
                valid = True
                for p in current_points:
                    if not dist_sq_rational(p[0], p[1], new_point[0], new_point[1]):
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check collinearity with all pairs
                valid = True
                n = len(current_points)
                for i in range(n):
                    for j in range(i+1, n):
                        if collinear(current_points[i][0], current_points[i][1],
                                     current_points[j][0], current_points[j][1],
                                     new_point[0], new_point[1]):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Check concyclicity with all triples
                valid = True
                n = len(current_points)
                for i in range(n):
                    for j in range(i+1, n):
                        for k in range(j+1, n):
                            if concyclic(current_points[i][0], current_points[i][1],
                                         current_points[j][0], current_points[j][1],
                                         current_points[k][0], current_points[k][1],
                                         new_point[0], new_point[1]):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Add and recurse
                new_candidates = [c for c in candidates[idx+1:] if c != (dx, dy)]
                # Actually, we should allow reusing displacements relative to different points
                # This is getting complex. Let's simplify.
                
                result = backtrack(current_points + [new_point], remaining_needed - 1, candidates[idx+1:])
                if result:
                    return result
            
            return None
        
        # Simplified approach: just try adding points one by one
        # Use a different strategy - enumerate small sets
        
        # Try specific known patterns
        # Pattern 1: Two points on x-axis, others symmetric
        
        # Let's just do a direct search for 6 points
        # Start with (0,0), then try to find 5 more
        
        best = None
        
        # Try different anchor pairs
        for x1 in range(-5, 6):
            if not check_time():
                break
            for y1 in range(-5, 6):
                if not check_time():
                    break
                p1 = (x1, y1)
                for x2 in range(-5, 6):
                    if not check_time():
                        break
                    for y2 in range(-5, 6):
                        if not check_time():
                            break
                        p2 = (x2, y2)
                        if p1 == p2:
                            continue
                        if not dist_sq_rational(x1, y1, x2, y2):
                            continue
                        
                        # Try to add 4 more points
                        current = [p1, p2]
                        
                        # Find candidate points that work with both p1 and p2
                        candidates = []
                        for x3 in range(-15, 16):
                            if not check_time():
                                break
                            for y3 in range(-15, 16):
                                if not check_time():
                                    break
                                p3 = (x3, y3)
                                if p3 in current:
                                    continue
                                if not dist_sq_rational(x1, y1, x3, y3):
                                    continue
                                if not dist_sq_rational(x2, y2, x3, y3):
                                    continue
                                if collinear(x1, y1, x2, y2, x3, y3):
                                    continue
                                candidates.append(p3)
                        
                        # Now try to pick 3 more from candidates
                        # This is still expensive. Let's limit.
                        
                        # For now, just return if we find a valid 3-point set
                        # and let the main function handle the rest
                        if len(candidates) >= 3:
                            # Try to complete to 6
                            for i in range(min(3, len(candidates))):
                                if not check_time():
                                    break
                                p3 = candidates[i]
                                current3 = [p1, p2, p3]
                                
                                # Find candidates for p4
                                cand4 = []
                                for x4 in range(-15, 16):
                                    if not check_time():
                                        break
                                    for y4 in range(-15, 16):
                                        if not check_time():
                                            break
                                        p4 = (x4, y4)
                                        if p4 in current3:
                                            continue
                                        valid = True
                                        for p in current3:
                                            if not dist_sq_rational(p[0], p[1], x4, y4):
                                                valid = False
                                                break
                                        if not valid:
                                            continue
                                        # Check collinearity
                                        for a in range(3):
                                            for b in range(a+1, 3):
                                                if collinear(current3[a][0], current3[a][1],
                                                             current3[b][0], current3[b][1],
                                                             x4, y4):
                                                    valid = False
                                                    break
                                            if not valid:
                                                break
                                        if not valid:
                                            continue
                                        # Check concyclicity with the 3 points
                                        if concyclic(current3[0][0], current3[0][1],
                                                     current3[1][0], current3[1][1],
                                                     current3[2][0], current3[2][1],
                                                     x4, y4):
                                            continue
                                        cand4.append(p4)
                                
                                if len(cand4) >= 2:
                                    # Try to complete to 6
                                    for j in range(min(2, len(cand4))):
                                        if not check_time():
                                            break
                                        p4 = cand4[j]
                                        current4 = current3 + [p4]
                                        
                                        # Find p5
                                        cand5 = []
                                        for x5 in range(-15, 16):
                                            if not check_time():
                                                break
                                            for y5 in range(-15, 16):
                                                if not check_time():
                                                    break
                                                p5 = (x5, y5)
                                                if p5 in current4:
                                                    continue
                                                valid = True
                                                for p in current4:
                                                    if not dist_sq_rational(p[0], p[1], x5, y5):
                                                        valid = False
                                                        break
                                                if not valid:
                                                    continue
                                                for a in range(4):
                                                    for b in range(a+1, 4):
                                                        if collinear(current4[a][0], current4[a][1],
                                                                     current4[b][0], current4[b][1],
                                                                     x5, y5):
                                                            valid = False
                                                            break
                                                    if not valid:
                                                        break
                                                if not valid:
                                                    continue
                                                for a in range(4):
                                                    for b in range(a+1, 4):
                                                        for c in range(b+1, 4):
                                                            if concyclic(current4[a][0], current4[a][1],
                                                                         current4[b][0], current4[b][1],
                                                                         current4[c][0], current4[c][1],
                                                                         x5, y5):
                                                                valid = False
                                                                break
                                                        if not valid:
                                                            break
                                                    if not valid:
                                                        break
                                                if not valid:
                                                    continue
                                                cand5.append(p5)
                                        
                                        if len(cand5) >= 1:
                                            if not check_time():
                                                break
                                            p5 = cand5[0]
                                            current5 = current4 + [p5]
                                            
                                            # Find p6
                                            for x6 in range(-15, 16):
                                                if not check_time():
                                                    break
                                                for y6 in range(-15, 16):
                                                    if not check_time():
                                                        break
                                                    p6 = (x6, y6)
                                                    if p6 in current5:
                                                        continue
                                                    valid = True
                                                    for p in current5:
                                                        if not dist_sq_rational(p[0], p[1], x6, y6):
                                                            valid = False
                                                            break
                                                    if not valid:
                                                        continue
                                                    for a in range(5):
                                                        for b in range(a+1, 5):
                                                            if collinear(current5[a][0], current5[a][1],
                                                                         current5[b][0], current5[b][1],
                                                                         x6, y6):
                                                                valid = False
                                                                break
                                                        if not valid:
                                                            break
                                                    if not valid:
                                                        continue
                                                    for a in range(5):
                                                        for b in range(a+1, 5):
                                                            for c in range(b+1, 5):
                                                                if concyclic(current5[a][0], current5[a][1],
                                                                             current5[b][0], current5[b][1],
                                                                             current5[c][0], current5[c][1],
                                                                             x6, y6):
                                                                    valid = False
                                                                    break
                                                            if not valid:
                                                                break
                                                        if not valid:
                                                            break
                                                    if not valid:
                                                        continue
                                                    
                                                    # Full validation
                                                    full_set = current5 + [p6]
                                                    if validate_set(full_set):
                                                        return full_set
                                            break
                                    break
                                break
                        break
                break
            break
        
        return best

    # Try the search
    result_6 = search_6_points(limit=12)
    
    if result_6 and validate_set(result_6):
        return {'D': 3, 'points': [[[x, 1], [y, 1]] for x, y in result_6]}
    
    # Fallback: return a known valid 5-point set
    # Let's verify a simple 5-point set with D=1 (integer coordinates)
    # Points: (0,0), (3,4), (0,8), (6,0), (9,4)
    # Check distances:
    # (0,0)-(3,4): 9+16=25 ✓
    # (0,0)-(0,8): 64 ✓
    # (0,0)-(6,0): 36 ✓
    # (0,0)-(9,4): 81+16=97 ✗
    
    # Try: (0,0), (3,4), (6,0), (9,4), (6,8)
    # (0,0)-(3,4): 25 ✓
    # (0,0)-(6,0): 36 ✓
    # (0,0)-(9,4): 97 ✗
    
    # Use the seed's 5-point set with D=1
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify with D=1
    def dist_sq_d1(x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        d2 = dx*dx + dy*dy
        return is_square(d2)
    
    def collinear_d1(x1, y1, x2, y2, x3, y3):
        return (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) == 0
    
    def concyclic_d1(x1, y1, x2, y2, x3, y3, x4, y4):
        # x^2 + y^2 + Ax + By + C = 0
        a1 = x2 - x1
        b1 = y2 - y1
        c1 = (x1*x1 + y1*y1) - (x2*x2 + y2*y2)
        
        a2 = x3 - x1
        b2 = y3 - y1
        c2 = (x1*x1 + y1*y1) - (x3*x3 + y3*y3)
        
        det = a1*b2 - a2*b1
        if det == 0:
            return True
        
        A_num = c1*b2 - c2*b1
        B_num = a1*c2 - a2*c1
        
        val = (det*(x4*x4 + y4*y4) 
               + A_num*x4 
               + B_num*y4 
               - (x1*x1 + y1*y1)*det 
               - A_num*x1 
               - B_num*y1)
        
        return val == 0
    
    def validate_d1(points):
        n = len(points)
        if len(set(points)) != n:
            return False
        for i in range(n):
            for j in range(i+1, n):
                if not dist_sq_d1(points[i][0], points[i][1], points[j][0], points[j][1]):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if collinear_d1(points[i][0], points[i][1], points[j][0], points[j][1], points[k][0], points[k][1]):
                        return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if concyclic_d1(points[i][0], points[i][1], points[j][0], points[j][1],
                                        points[k][0], points[k][1], points[l][0], points[l][1]):
                            return False
        return True
    
    if validate_d1(seed_points):
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in seed_points]}
    
    # Last resort: return 3 known valid points
    # (0,0), (3,4), (0,8) with D=1
    # (0,0)-(3,4): 25 ✓
    # (0,0)-(0,8): 64 ✓
    # (3,4)-(0,8): 9+16=25 ✓
    # Not collinear: (3)(8) - (0)(4) = 24 ≠ 0 ✓
    return {'D': 1, 'points': [[[0, 1], [0, 1]], [[3, 1], [4, 1]], [[0, 1], [8, 1]]]}

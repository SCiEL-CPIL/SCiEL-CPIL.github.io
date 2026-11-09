import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Helper: Check if 4 points are collinear
    def has_four_collinear(points):
        n = len(points)
        # Map line to count of points
        # Line represented by (A, B, C) where Ax + By + C = 0, normalized
        lines = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                # A = y1 - y2, B = x2 - x1, C = x1*y2 - x2*y1
                A = y1 - y2
                B = x2 - x1
                C = x1 * y2 - x2 * y1
                # Normalize
                if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
                    A, B, C = -A, -B, -C
                # GCD reduction
                def gcd(a, b):
                    while b:
                        a, b = b, a % b
                    return abs(a)
                g = gcd(A, B)
                if g == 0: g = 1 # Point, but i!=j so not both 0
                g = gcd(g, C)
                if g > 1:
                    A //= g
                    B //= g
                    C //= g
                key = (A, B, C)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
        
        for pts in lines.values():
            if len(pts) >= 4:
                return True
        return False

    def get_collinear_triples(points):
        n = len(points)
        triples = set()
        # Map line to points
        lines = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                A = y1 - y2
                B = x2 - x1
                C = x1 * y2 - x2 * y1
                if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
                    A, B, C = -A, -B, -C
                def gcd(a, b):
                    while b:
                        a, b = b, a % b
                    return abs(a)
                g = gcd(A, B)
                if g == 0: g = 1
                g = gcd(g, C)
                if g > 1:
                    A //= g
                    B //= g
                    C //= g
                key = (A, B, C)
                if key not in lines:
                    lines[key] = []
                lines[key].append(i)
                lines[key].append(j)
        
        # For each line, if it has k points, it contributes C(k,3) triples
        for pts in lines.values():
            if len(pts) >= 3:
                for comb in combinations(pts, 3):
                    triples.add(comb)
        return list(triples)

    def min_hitting_set_size(triples):
        if not triples:
            return 0
        
        # Convert to list of tuples for speed
        triples = [tuple(t) for t in triples]
        
        import sys
        sys.setrecursionlimit(10000)
        
        best = len(triples) # Upper bound: one point per triple
        
        # Sort triples by length? All are 3.
        # Heuristic: Pick the triple that appears in the most other triples?
        # Or just pick the first one.
        
        def backtrack(remaining_triples, current_size):
            nonlocal best
            if not remaining_triples:
                if current_size < best:
                    best = current_size
                return
            if current_size >= best:
                return
            
            # Pick a triple with the highest "constraint"
            # Simple: pick the first one
            t = remaining_triples[0]
            rest = remaining_triples[1:]
            
            for v in t:
                # New remaining triples: those that do NOT contain v
                new_rest = [rt for rt in rest if v not in rt]
                # Optimization: if new_rest is same as rest, we might be looping?
                # No, we are removing t, and filtering rest.
                backtrack(new_rest, current_size + 1)
                
        backtrack(triples, 0)
        return best

    def score(points):
        if has_four_collinear(points):
            return 100 # Invalid
        triples = get_collinear_triples(points)
        h = min_hitting_set_size(triples)
        return 20 - h

    # Initial Configuration: 6 disjoint lines with 3 points each
    # Lines: y = 0, 10, 20, -10, -20, 30
    # Points on y=0: (-30, 0), (0, 0), (30, 0)
    # Points on y=10: (-30, 10), (0, 10), (30, 10) -- Wait, vertical lines x=-30, 0, 30 would create 4 collinear?
    # If I use horizontal lines, vertical alignment creates 4 collinear if I use same x.
    # So I must jitter x.
    
    def generate_initial():
        pts = []
        # 6 horizontal lines
        y_coords = [0, 10, 20, -10, -20, 30]
        for y in y_coords:
            # Pick 3 distinct x's, avoiding vertical collisions with other lines
            # Use random x in [-50, 50]
            x1, x2, x3 = random.sample(range(-50, 51), 3)
            pts.append([x1, y])
            pts.append([x2, y])
            pts.append([x3, y])
        # Add 2 more random points
        for _ in range(2):
            pts.append([random.randint(-100, 100), random.randint(-100, 100)])
        return pts

    best_points = generate_initial()
    best_score_val = score(best_points)
    
    # Local Search
    while time.time() < deadline - 5.0:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_x, old_y = best_points[idx]
        
        # Try a few random moves
        found_improvement = False
        for _ in range(20):
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            candidate = [p[:] for p in best_points]
            candidate[idx] = [new_x, new_y]
            
            # Check distinctness
            if len(set(map(tuple, candidate))) != 20:
                continue
            
            s = score(candidate)
            if s < best_score_val:
                best_points = candidate
                best_score_val = s
                found_improvement = True
                break
            elif s == best_score_val and random.random() < 0.1:
                # Accept equal to escape local minima?
                # Risky if it leads to invalid, but score() handles invalid.
                # Actually, if s is 100 (invalid), we skip.
                pass

        if not found_improvement and random.random() < 0.05:
            # Restart with new random config occasionally
            new_pts = generate_initial()
            s = score(new_pts)
            if s < best_score_val:
                best_points = new_pts
                best_score_val = s

    return best_points

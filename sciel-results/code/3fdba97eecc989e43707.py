import time
import random
import itertools
from math import comb

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for safety
    
    # Helper to check collinearity of 3 points
    def is_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    # Helper to calculate the number of collinear triples
    def count_triples(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        count += 1
        return count

    # Helper to check if any 4 points are collinear
    def has_four_collinear(points):
        n = len(points)
        # Efficient check: for each pair, count points on the line
        # But for n=20, O(n^4) is ~ 10k ops, which is fine.
        # We can optimize by just checking if any line has >= 4 points.
        lines = set()
        for i in range(n):
            for j in range(i + 1, n):
                # Canonical line representation: (A, B, C, D) where Ax^2+Bxy+By^2+Cx+Dy+E=0
                # To avoid floating point, use integer coefficients normalized.
                # Line through (x1,y1) and (x2,y2):
                # (y2-y1)x - (x2-x1)y + (x2*y1 - y2*x1) = 0
                a = points[j][1] - points[i][1]
                b = points[i][0] - points[j][0]
                c = points[j][0]*points[i][1] - points[i][0]*points[j][1]
                
                # Normalize
                # Find gcd of a, b, c
                def gcd(a, b):
                    while b:
                        a, b = b, a % b
                    return a
                g = gcd(gcd(abs(a), abs(b)), abs(c))
                if g == 0:
                    continue
                a //= g
                b //= g
                c //= g
                # Ensure canonical sign
                if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                    a = -a
                    b = -b
                    c = -c
                lines.add((a, b, c))
        
        # Now check if any line has >= 4 points
        for line in lines:
            a, b, c = line
            cnt = 0
            for p in points:
                if a*p[0] + b*p[1] + c == 0:
                    cnt += 1
            if cnt >= 4:
                return True
        return False

    # Generate initial 20 distinct points with no 4 collinear
    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    points = []
    attempts = 0
    while len(points) < 20:
        p = random_point()
        if p in points:
            continue
        
        valid = True
        # Check no 4 collinear with existing points
        # A new point p creates a 4-collinear set if there exist 3 existing points
        # that are collinear with p? No, 4 collinear means 4 points on a line.
        # So if p is added, we check if there is a line containing p and 3 other points from 'points'.
        # This is equivalent to: for every pair (i, j) in points, if p is on line(i,j),
        # we must ensure no other point k is on line(i,j).
        # Actually simpler: just check all combinations of 3 existing points.
        # If any 3 existing points are collinear with p, then those 3 + p are 4 collinear.
        # Wait, if 3 existing points are NOT collinear, they don't form a line.
        # So we only need to check: for every pair (i, j) in points, if p is on line(i,j),
        # is there a third point k on line(i,j)?
        
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if is_collinear(points[i], points[j], p):
                    # Check if any other point is on this line
                    for k in range(n):
                        if k != i and k != j and is_collinear(points[i], points[j], points[k]):
                            valid = False
                            break
            if not valid:
                break
        
        if valid:
            points.append(p)
        attempts += 1
        if attempts > 10000:
            points = []
            attempts = 0

    current_points = points
    current_triples = count_triples(current_points)
    best_points = [p[:] for p in current_points]
    best_triples = current_triples

    # Greedy Line Balancing Strategy
    # We identify all lines with exactly 3 points.
    # For each such line, we try to "break" it by moving one of its points.
    # We choose the point that, when moved, reduces the total triple count the most.
    # To make it deterministic and fast, we try a set of candidate moves for the "worst" point.
    
    def get_lines_with_3(points):
        lines = {}
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                a = points[j][1] - points[i][1]
                b = points[i][0] - points[j][0]
                c = points[j][0]*points[i][1] - points[i][0]*points[j][1]
                
                def gcd(a, b):
                    while b:
                        a, b = b, a % b
                    return a
                g = gcd(gcd(abs(a), abs(b)), abs(c))
                if g == 0:
                    continue
                a //= g
                b //= g
                c //= g
                if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                    a = -a
                    b = -b
                    c = -c
                key = (a, b, c)
                if key not in lines:
                    lines[key] = []
                lines[key].append(i)
        
        # Filter to lines with exactly 3 points
        return [pts for pts in lines.values() if len(pts) == 3]

    # Main loop
    while time.time() < deadline:
        # Find all lines with 3 points
        triples_lines = get_lines_with_3(current_points)
        
        if not triples_lines:
            break # No triples, perfect! (Though unlikely to reach 0, maybe 1 or 2)
            
        # Pick a line to break. 
        # Strategy: Pick the line that is part of the most other triples? 
        # Or just iterate through them.
        # Let's pick the first one, but try to break it optimally.
        # Actually, let's find the point that is in the most 3-collinear lines.
        point_freq = {}
        for line_pts in triples_lines:
            for idx in line_pts:
                point_freq[idx] = point_freq.get(idx, 0) + 1
        
        if not point_freq:
            break
            
        # Pick the point with highest frequency
        target_idx = max(point_freq, key=point_freq.get)
        
        # Try to move this point to a new location that reduces triples
        old_p = current_points[target_idx]
        
        # Generate candidate moves
        # 1. Random moves
        # 2. Moves that are "near" the old point but not on existing lines
        candidates = []
        for _ in range(50):
            new_p = [random.randint(-100, 100), random.randint(-100, 100)]
            if new_p != old_p and new_p not in current_points:
                candidates.append(new_p)
        
        # Also try some structured moves: shift x or y by small amounts
        for dx in [-2, -1, 1, 2]:
            for dy in [-2, -1, 1, 2]:
                new_p = [old_p[0] + dx, old_p[1] + dy]
                if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                    if new_p != old_p and new_p not in current_points:
                        candidates.append(new_p)
        
        best_new_p = None
        best_new_triples = current_triples
        
        for new_p in candidates:
            test_points = current_points.copy()
            test_points[target_idx] = new_p
            
            # Quick validity check: no 4 collinear
            if has_four_collinear(test_points):
                continue
            
            new_triples = count_triples(test_points)
            if new_triples < best_new_triples:
                best_new_triples = new_triples
                best_new_p = new_p
            
            # Early exit if we found a perfect move (reduced triples)
            # We don't necessarily need the absolute best, just an improvement
            # But taking the best among candidates is better.
        
        if best_new_p is not None:
            current_points[target_idx] = best_new_p
            current_triples = best_new_triples
            if current_triples < best_triples:
                best_triples = current_triples
                best_points = [p[:] for p in current_points]
        else:
            # If no candidate improved the score, we might be stuck.
            # Try a random restart of the whole configuration? 
            # Or just accept a random move to escape?
            # Let's try a random move that is valid, even if it doesn't improve triples immediately.
            # This is a form of simulated annealing but triggered only on stall.
            valid_move_found = False
            for _ in range(100):
                new_p = [random.randint(-100, 100), random.randint(-100, 100)]
                if new_p != old_p and new_p not in current_points:
                    test_points = current_points.copy()
                    test_points[target_idx] = new_p
                    if not has_four_collinear(test_points):
                        # Accept if it doesn't make it worse than current?
                        # Or just accept to explore?
                        # Let's accept if it's not significantly worse.
                        nt = count_triples(test_points)
                        if nt <= current_triples + 2: # Allow small increase
                            current_points[target_idx] = new_p
                            current_triples = nt
                            valid_move_found = True
                            break
            
            if not valid_move_found:
                # If we can't even find a valid move, restart the whole thing?
                # Or just break and return best.
                # Given time constraints, let's just break out of this specific line-breaking loop
                # and try a different point or line.
                # To avoid infinite loop on the same bad point, we can remove it from consideration
                # but our logic picks max freq each time.
                # Let's just force a random perturbation of a different point.
                other_idx = random.randint(0, 19)
                if other_idx != target_idx:
                    old_other = current_points[other_idx]
                    for _ in range(50):
                        new_other = [random.randint(-100, 100), random.randint(-100, 100)]
                        if new_other != old_other and new_other not in current_points:
                            test_points = current_points.copy()
                            test_points[other_idx] = new_other
                            if not has_four_collinear(test_points):
                                current_points[other_idx] = new_other
                                current_triples = count_triples(current_points)
                                break

    return best_points
# EVOLVE-BLOCK-END

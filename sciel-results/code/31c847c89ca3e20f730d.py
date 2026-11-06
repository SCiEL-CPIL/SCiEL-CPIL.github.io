import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = time.time() + 145  # 5s margin for safety
    
    # Helper: Check if 3 points are collinear
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    # Helper: Check if any 4 points are collinear in a list of points - O(n^3) using line grouping
    def has_4_collinear(points):
        n = len(points)
        # For each point, count how many other points are on each line through it
        for i in range(n):
            line_counts = defaultdict(int)
            for j in range(n):
                if i == j:
                    continue
                # Compute normalized direction from i to j
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                # Normalize direction
                if dx == 0 and dy == 0:
                    continue
                if dx < 0 or (dx == 0 and dy < 0):
                    dx, dy = -dx, -dy
                g = abs(dx)
                h = abs(dy)
                while g != 0 and h != 0:
                    g, h = h, g % h
                g = g if g != 0 else 1
                dx, dy = dx // g, dy // g
                line_counts[(dx, dy)] += 1
            if any(c >= 3 for c in line_counts.values()):
                return True
        return False

    # Helper: Exact branch and bound for largest subset with no 3 collinear
    # Uses bitmasks for speed with better pruning
    def max_no_3_collinear(points):
        n = len(points)
        
        # Precompute for each i, the set of pairs (j, k) that are collinear with i
        bad_pairs_for = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        bad_pairs_for[i].add((j, k))
                        bad_pairs_for[j].add((i, k))
                        bad_pairs_for[k].add((i, j))
        
        # Also precompute: for each i, a bitmask of points that form a bad pair with some other point
        # Actually, let's use a different approach: for each i, precompute which subsets of {0..i-1} 
        # are "compatible" with i. But that's too complex.
        
        # Better pruning: use upper bound based on remaining points minus forced exclusions
        best = 0
        
        # Precompute for each i, the number of bad pairs involving i with indices < i
        # This helps with tighter bounds
        def backtrack(S_mask, start_idx):
            nonlocal best
            current_len = bin(S_mask).count('1')
            remaining = n - start_idx
            # Simple upper bound
            if current_len + remaining <= best:
                return
            
            if start_idx == n:
                if current_len > best:
                    best = current_len
                return
            
            # Try including start_idx first (often finds good solutions faster)
            # Check if adding start_idx to S is valid
            valid = True
            for (a, b) in bad_pairs_for[start_idx]:
                if a < start_idx and b < start_idx:
                    if (S_mask & (1 << a)) and (S_mask & (1 << b)):
                        valid = False
                        break
            
            if valid:
                backtrack(S_mask | (1 << start_idx), start_idx + 1)
            
            # Option: Skip point start_idx
            backtrack(S_mask, start_idx + 1)
                
        backtrack(0, 0)
        return best

    # Strategy: Greedy Construction
    # We want to maximize the number of collinear triples to minimize the max independent set (no-3-collinear).
    # Greedy: Place points one by one.
    # At each step, consider all candidate points (x, y) in [-100, 100] x [-100, 100] that are distinct and don't create 4-collinear.
    # Score a candidate by how many new collinear triples it creates.
    # Pick the candidate with the highest score.
    
    # To make this efficient, we don't check all 200x200 points at each step.
    # We can restrict candidates to points that are collinear with at least 2 existing points (to form a triple) 
    # or just random sampling if none exist.
    # Actually, a good heuristic is to try to extend existing lines.
    
    def greedy_construct():
        pts = []
        # Start with a few random points to break symmetry
        for _ in range(3):
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if [x, y] not in pts:
                    pts.append([x, y])
                    break
        
        while len(pts) < 20:
            if time.time() > deadline - 5:
                break
                
            best_candidate = None
            best_score = -1
            
            # Generate candidates: points that create collinear triples with existing points
            candidates = set()
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    p1 = pts[i]
                    p2 = pts[j]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    
                    if dx == 0:
                        for y in range(-100, 101):
                            p = [p1[0], y]
                            if p not in pts:
                                candidates.add((p[0], p[1]))
                    else:
                        for x in range(-100, 101):
                            num = dy * (x - p1[0])
                            if num % dx == 0:
                                y = p1[1] + num // dx
                                if -100 <= y <= 100:
                                    p = [x, y]
                                    if p not in pts:
                                        candidates.add((p[0], p[1]))
            
            # Add some random points to escape local minima
            for _ in range(30):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if [x, y] not in pts:
                    candidates.add((x, y))
            
            # Evaluate candidates
            for cx, cy in candidates:
                p = [cx, cy]
                # Quick check: does this create 4-collinear?
                # Only check lines that contain p and at least 2 existing points
                creates_4 = False
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        if are_collinear(p, pts[i], pts[j]):
                            # Check if there's already a 3rd point on this line (excluding p, i, j)
                            for k in range(len(pts)):
                                if k != i and k != j:
                                    if are_collinear(p, pts[i], pts[k]):
                                        creates_4 = True
                                        break
                        if creates_4:
                            break
                    if creates_4:
                        break
                if creates_4:
                    continue
                
                # Count new triples
                triples = 0
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        if are_collinear(p, pts[i], pts[j]):
                            triples += 1
                
                if triples > best_score:
                    best_score = triples
                    best_candidate = p
            
            if best_candidate is None:
                # Fallback: pick a random valid point
                for _ in range(100):
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    p = [x, y]
                    if p not in pts and not has_4_collinear(pts + [p]):
                        best_candidate = p
                        break
                if best_candidate is None:
                    best_candidate = [random.randint(-100, 100), random.randint(-100, 100)]
            
            pts.append(best_candidate)
            
        return pts

    # Main loop
    best_config = None
    best_score = 21  # Max possible score is 20
    
    # Run greedy construction multiple times
    num_greedy = 15
    for _ in range(num_greedy):
        if time.time() > deadline - 15:
            break
            
        pts = greedy_construct()
        
        # Validate
        if len(set(map(tuple, pts))) != 20:
            continue
        if has_4_collinear(pts):
            continue
        if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in pts):
            continue
            
        score = max_no_3_collinear(pts)
        
        if score < best_score:
            best_score = score
            best_config = [list(p) for p in pts]
            
        # Local search: try to improve by moving individual points
        if best_config is not None and time.time() < deadline - 10:
            improved = [list(p) for p in best_config]
            for step in range(100):
                if time.time() > deadline - 3:
                    break
                idx = random.randint(0, 19)
                old_p = improved[idx][:]
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                improved[idx] = [new_x, new_y]
                
                if len(set(map(tuple, improved))) != 20 or has_4_collinear(improved):
                    improved[idx] = old_p
                    continue
                    
                new_score = max_no_3_collinear(improved)
                if new_score < best_score:
                    best_score = new_score
                    best_config = [list(p) for p in improved]
                else:
                    improved[idx] = old_p
                    
    if best_config is None:
        # Fallback: use a known good configuration
        # Points on a parabola y = x^2 have no 3 collinear, so score = 20 (bad)
        # Instead, use a configuration with more collinearities
        # Try a grid-like pattern with perturbations
        fallback = []
        for i in range(20):
            x = i % 5
            y = (i // 5) * 7 + (i % 3)
            fallback.append([x - 2, y])
        return fallback
        
    return best_config

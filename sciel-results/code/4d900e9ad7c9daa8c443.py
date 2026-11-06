import time
import math
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0
    
    def elapsed():
        return time.time() - start_time

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = math.gcd(A, math.gcd(B, C))
        if g == 0:
            return None
        A //= g
        B //= g
        C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def has_4_collinear(pts):
        n = len(pts)
        if n < 4:
            return False
        line_map = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is not None:
                    line_map[key].append((i, j))
        for key, pairs in line_map.items():
            A, B, C = key
            count = 0
            for k in range(n):
                x3, y3 = pts[k]
                if A * x3 + B * y3 + C == 0:
                    count += 1
            if count >= 4:
                return True
        return False

    def count_triples(pts):
        """Count number of collinear triples in the point set."""
        n = len(pts)
        triples = 0
        line_map = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is not None:
                    line_map[key].add(i)
                    line_map[key].add(j)
        for key, indices in line_map.items():
            if len(indices) >= 3:
                # Count triples: C(k,3)
                k = len(indices)
                triples += k * (k - 1) * (k - 2) // 6
        return triples

    def calculate_score(pts):
        """Exact branch and bound for largest subset with no 3 collinear."""
        n = len(pts)
        if n < 3:
            return n
        
        # Build conflict graph: edge between i,j if there exists k such that i,j,k collinear
        # Actually, we need: triple (i,j,k) collinear means we can't have all three in subset
        triples_with_idx = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        triples_with_idx[i].append((j, k))
                        triples_with_idx[j].append((i, k))
                        triples_with_idx[k].append((i, j))
        
        best_score = 0
        
        # Order by number of conflicts (most constrained first)
        order = sorted(range(n), key=lambda i: len(triples_with_idx[i]), reverse=True)
        
        def bbb(step, current_mask, current_size):
            nonlocal best_score
            if elapsed() > deadline - 5:
                return
            if current_size > best_score:
                best_score = current_size
            if step == n:
                return
            if current_size + (n - step) <= best_score:
                return
            idx = order[step]
            # Try excluding
            bbb(step + 1, current_mask, current_size)
            # Try including if valid
            valid = True
            for a, b in triples_with_idx[idx]:
                if (current_mask & (1 << a)) and (current_mask & (1 << b)):
                    valid = False
                    break
            if valid:
                bbb(step + 1, current_mask | (1 << idx), current_size + 1)
                
        bbb(0, 0, 0)
        return best_score

    def fast_score_estimate(pts):
        """Fast heuristic estimate of the score using greedy."""
        n = len(pts)
        if n < 3:
            return n
        
        # Build line map
        line_map = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is not None:
                    line_map[key].add(i)
                    line_map[key].add(j)
        
        # Greedy: repeatedly pick the point that is in the fewest "active" lines
        remaining = set(range(n))
        score = 0
        while remaining:
            # For each point, count how many lines it's on with other remaining points
            best_pt = None
            best_count = float('inf')
            for pt in remaining:
                line_count = 0
                for key, indices in line_map.items():
                    if pt in indices:
                        other_in_remaining = [x for x in indices if x in remaining and x != pt]
                        if len(other_in_remaining) >= 2:
                            line_count += 1
                if line_count < best_count:
                    best_count = line_count
                    best_pt = pt
            score += 1
            remaining.discard(best_pt)
        return score

    # Strategy: Line-based construction
    # We want to create many lines with exactly 3 points (triples)
    # This forces any no-3-collinear subset to pick at most 2 from each such line
    
    def construct_from_lines():
        """Construct points by choosing lines and placing points on them."""
        # Use a grid-based approach with controlled perturbations
        # Start with a base grid and add noise to break 4-collinear while keeping triples
        
        best_pts = None
        best_triples = -1
        
        for attempt in range(30):
            if elapsed() > deadline - 50:
                break
            
            # Create a perturbed grid
            base_size = random.randint(5, 7)
            step = random.randint(10, 20)
            ox = random.randint(-50, 50)
            oy = random.randint(-50, 50)
            
            # Generate candidate points from grid with perturbations
            candidates = []
            for i in range(base_size):
                for j in range(base_size):
                    x = ox + i * step + random.randint(-3, 3)
                    y = oy + j * step + random.randint(-3, 3)
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        candidates.append((x, y))
            
            # Remove duplicates
            seen = set()
            unique_candidates = []
            for p in candidates:
                if p not in seen:
                    seen.add(p)
                    unique_candidates.append(p)
            
            if len(unique_candidates) < 20:
                continue
            
            # Greedily select 20 points maximizing triples
            selected = []
            for _ in range(20):
                if elapsed() > deadline - 50:
                    break
                best_cand = None
                best_triple_gain = -1
                
                for cand in unique_candidates:
                    if cand in selected:
                        continue
                    
                    # Check if adding this creates 4-collinear
                    valid = True
                    triple_gain = 0
                    for i in range(len(selected)):
                        a = selected[i]
                        for j in range(i + 1, len(selected)):
                            b = selected[j]
                            x1, y1 = a
                            x2, y2 = b
                            xp, yp = cand
                            if (x2 - x1) * (yp - y1) == (y2 - y1) * (xp - x1):
                                # Check for 4th collinear
                                for k in range(len(selected)):
                                    if k != i and k != j:
                                        c = selected[k]
                                        xc, yc = c
                                        if (x2 - x1) * (yc - y1) == (y2 - y1) * (xc - x1):
                                            valid = False
                                            break
                                if not valid:
                                    break
                                else:
                                    triple_gain += 1
                        if not valid:
                            break
                    
                    if valid and triple_gain > best_triple_gain:
                        best_triple_gain = triple_gain
                        best_cand = cand
                
                if best_cand is None:
                    break
                selected.append(best_cand)
            
            if len(selected) == 20 and not has_4_collinear(selected):
                t = count_triples(selected)
                if t > best_triples:
                    best_triples = t
                    best_pts = selected
        
        return best_pts

    def local_search(pts, max_iters=200):
        """Local search to improve triple count while maintaining validity."""
        if pts is None:
            return None
        
        current = list(pts)
        current_triples = count_triples(current)
        best = list(current)
        best_triples = current_triples
        
        for it in range(max_iters):
            if elapsed() > deadline - 20:
                break
            
            idx = random.randint(0, 19)
            old_p = current[idx]
            
            # Generate neighbor
            r = random.random()
            if r < 0.4:
                new_p = (old_p[0] + random.randint(-4, 4), old_p[1] + random.randint(-4, 4))
            elif r < 0.7:
                new_p = (old_p[0] + random.randint(-10, 10), old_p[1] + random.randint(-10, 10))
            else:
                new_p = (random.randint(-100, 100), random.randint(-100, 100))
            
            if new_p in current:
                continue
            if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                continue
            
            # Check validity
            valid = True
            for i in range(20):
                if i == idx:
                    continue
                x1, y1 = current[i]
                for j in range(i + 1, 20):
                    if j == idx:
                        continue
                    x2, y2 = current[j]
                    xp, yp = new_p
                    if (x2 - x1) * (yp - y1) == (y2 - y1) * (xp - x1):
                        for k in range(20):
                            if k == idx or k == i or k == j:
                                continue
                            x3, y3 = current[k]
                            if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            new_pts = current[:]
            new_pts[idx] = new_p
            new_triples = count_triples(new_pts)
            
            if new_triples > current_triples:
                current = new_pts
                current_triples = new_triples
                if current_triples > best_triples:
                    best_triples = current_triples
                    best = list(current)
            elif new_triples == current_triples and random.random() < 0.1:
                current = new_pts
        
        return best

    # Main search
    best_points = None
    best_score = float('inf')
    
    # Phase 1: Construct good initial configurations
    for attempt in range(5):
        if elapsed() > deadline - 60:
            break
        
        pts = construct_from_lines()
        if pts is not None:
            # Local search to improve
            improved = local_search(pts, max_iters=100)
            if improved is not None:
                s = calculate_score(improved)
                if s < best_score:
                    best_score = s
                    best_points = improved
    
    # Phase 2: Try more diverse constructions
    if elapsed() < deadline - 40:
        for attempt in range(8):
            if elapsed() > deadline - 30:
                break
            
            # Different grid parameters
            base_size = random.randint(4, 8)
            step_x = random.randint(8, 25)
            step_y = random.randint(8, 25)
            ox = random.randint(-60, 60)
            oy = random.randint(-60, 60)
            
            candidates = []
            for i in range(base_size):
                for j in range(base_size):
                    x = ox + i * step_x + random.randint(-2, 2)
                    y = oy + j * step_y + random.randint(-2, 2)
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        candidates.append((x, y))
            
            seen = set()
            unique_candidates = []
            for p in candidates:
                if p not in seen:
                    seen.add(p)
                    unique_candidates.append(p)
            
            if len(unique_candidates) < 20:
                continue
            
            # Greedy selection
            selected = []
            for _ in range(20):
                if elapsed() > deadline - 30:
                    break
                best_cand = None
                best_gain = -1
                for cand in unique_candidates:
                    if cand in selected:
                        continue
                    valid = True
                    gain = 0
                    for i in range(len(selected)):
                        a = selected[i]
                        for j in range(i + 1, len(selected)):
                            b = selected[j]
                            x1, y1 = a
                            x2, y2 = b
                            xp, yp = cand
                            if (x2 - x1) * (yp - y1) == (y2 - y1) * (xp - x1):
                                for k in range(len(selected)):
                                    if k != i and k != j:
                                        c = selected[k]
                                        xc, yc = c
                                        if (x2 - x1) * (yc - y1) == (y2 - y1) * (xc - x1):
                                            valid = False
                                            break
                                if not valid:
                                    break
                                gain += 1
                        if not valid:
                            break
                    if valid and gain > best_gain:
                        best_gain = gain
                        best_cand = cand
                if best_cand is None:
                    break
                selected.append(best_cand)
            
            if len(selected) == 20 and not has_4_collinear(selected):
                improved = local_search(selected, max_iters=80)
                if improved is not None:
                    s = calculate_score(improved)
                    if s < best_score:
                        best_score = s
                        best_points = improved
    
    # Phase 3: Intensive local search on best found
    if best_points is not None and elapsed() < deadline - 15:
        improved = local_search(best_points, max_iters=300)
        if improved is not None:
            s = calculate_score(improved)
            if s < best_score:
                best_score = s
                best_points = improved
    
    # Fallback
    if best_points is None:
        while True:
            pts = []
            while len(pts) < 20:
                p = (random.randint(-100, 100), random.randint(-100, 100))
                if p not in pts:
                    pts.append(p)
            if not has_4_collinear(pts):
                best_points = pts
                break
    
    return [[x, y] for x, y in best_points]

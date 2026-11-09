import random
import itertools
import time
import math

def is_collinear(p1, p2, p3):
    return (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]) == 0)

def has_four_collinear(points):
    # Check all combinations of 4
    for quad in itertools.combinations(points, 4):
        if is_collinear(quad[0], quad[1], quad[2]) and is_collinear(quad[0], quad[1], quad[3]):
            return True
    return False

def count_three_collinear(points):
    count = 0
    for tri in itertools.combinations(points, 3):
        if is_collinear(tri[0], tri[1], tri[2]):
            count += 1
    return count

def max_3_free_subset(points):
    n = len(points)
    coll = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if is_collinear(points[i], points[j], points[k]):
                    coll[i][j] = coll[i][k] = coll[j][k] = True
                    
    best = 0
    def backtrack(start, current, size):
        nonlocal best
        if size > best:
            best = size
        if start >= n:
            return
        if size + (n - start) <= best:
            return
        valid = True
        for i in current:
            if coll[i][start]:
                valid = False
                break
        if valid:
            backtrack(start + 1, current + [start], size + 1)
        backtrack(start + 1, current, size)
        
    backtrack(0, [], 0)
    return best

def approx_max_3_free_subset(points):
    """Fast heuristic: Greedy removal of points involved in collinear triples."""
    n = len(points)
    active = [True] * n
    # Precompute collinearity pairs for speed? No, just check on the fly for the greedy step.
    # We want to estimate the size of the largest 3-free subset.
    # Greedy: While there exists a collinear triple among active points, remove one.
    # To maximize the remaining set, remove the point that is in the most triples?
    # Or simply remove any point from the first triple found.
    
    # A better heuristic: 
    # 1. Count how many collinear triples each point is part of.
    # 2. Repeatedly remove the point with the highest count, update counts.
    # 3. The number of remaining points is a lower bound on the max 3-free size.
    
    counts = [0] * n
    # Precompute triples
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if is_collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
                    counts[i] += 1
                    counts[j] += 1
                    counts[k] += 1
    
    active = [True] * n
    remaining = n
    
    while True:
        # Find a triple where all are active
        found_triple = None
        for tri in triples:
            if active[tri[0]] and active[tri[1]] and active[tri[2]]:
                found_triple = tri
                break
        if not found_triple:
            break
        
        # Remove the point in the triple with the highest count
        i, j, k = found_triple
        max_count = -1
        idx_to_remove = i
        for idx in [i, j, k]:
            if counts[idx] > max_count:
                max_count = counts[idx]
                idx_to_remove = idx
        
        # Remove idx_to_remove
        active[idx_to_remove] = False
        remaining -= 1
        
        # Update counts for all triples involving idx_to_remove
        for tri in triples:
            if idx_to_remove in tri:
                for p in tri:
                    if p != idx_to_remove:
                        counts[p] -= 1
                        # Note: counts[p] is now the number of active triples p is in?
                        # Actually, we just decremented it. We need to ensure we only count active triples.
                        # The above loop decrements counts for p if the triple was active?
                        # No, the triple might have been inactive before?
                        # If the triple had an inactive point, it wasn't the one we found (found_triple was active).
                        # But other triples involving idx_to_remove might have been inactive.
                        # This heuristic is approximate. It's good enough for a cost function.
                        
    return remaining

def construct():
    random.seed(42)
    
    # Strategy: Generate multiple candidate sets, run local search on each, return best.
    
    def generate_random_set():
        pts = []
        while len(pts) < 20:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        return pts

    def generate_algebraic_set():
        # Inspired by PG(2,4) affine part
        # Map F4 to {0, 1, 2, 3}
        # Points (x, y) in {0,1,2,3}x{0,1,2,3}
        # This creates 4-collinearities. We perturb them.
        pts = []
        for x in range(4):
            for y in range(4):
                # Add jitter to break collinearities
                jx = random.randint(-2, 2)
                jy = random.randint(-2, 2)
                pts.append([x*10 + jx, y*10 + jy])
        
        # Add 4 more points from "infinity" or random to make 20
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        
        # Ensure no duplicates
        unique_pts = []
        for p in pts:
            if p not in unique_pts:
                unique_pts.append(p)
        return unique_pts[:20]

    best_pts = None
    best_score = 20
    
    candidates = [generate_random_set(), generate_algebraic_set(), generate_random_set()]
    
    for pts in candidates:
        # Phase 1: Break 4-collinearities
        for _ in range(1000):
            if not has_four_collinear(pts):
                break
            # Find a 4-collinear quad
            quad_found = None
            for quad in itertools.combinations(pts, 4):
                if is_collinear(quad[0], quad[1], quad[2]) and is_collinear(quad[0], quad[1], quad[3]):
                    quad_found = quad
                    break
            if not quad_found:
                break
            
            # Move one point randomly
            idx = random.choice([0, 1, 2, 3])
            p_idx = pts.index(quad_found[idx])
            old_p = pts[p_idx]
            for _ in range(20):
                nx = random.randint(-100, 100)
                ny = random.randint(-100, 100)
                if [nx, ny] not in pts:
                    pts[p_idx] = [nx, ny]
                    if not has_four_collinear(pts):
                        break
            else:
                pts[p_idx] = old_p
        
        if has_four_collinear(pts):
            continue
            
        # Phase 2: Simulated Annealing to minimize 3-free subset
        # Cost = approx_3_free_size * 100 - triple_count
        current_triples = count_three_collinear(pts)
        current_approx = approx_max_3_free_subset(pts)
        current_cost = current_approx * 100 - current_triples
        
        temperature = 10.0
        cooling = 0.95
        
        for step in range(5000):
            if temperature < 0.1:
                break
            
            i = random.randint(0, 19)
            old_p = pts[i]
            nx = random.randint(-100, 100)
            ny = random.randint(-100, 100)
            
            if [nx, ny] in pts:
                continue
            
            pts[i] = [nx, ny]
            
            if has_four_collinear(pts):
                pts[i] = old_p
                continue
            
            new_triples = count_three_collinear(pts)
            new_approx = approx_max_3_free_subset(pts)
            new_cost = new_approx * 100 - new_triples
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_triples = new_triples
                current_approx = new_approx
                current_cost = new_cost
            else:
                pts[i] = old_p
            
            temperature *= cooling
        
        # Final exact evaluation
        if not has_four_collinear(pts):
            exact_score = max_3_free_subset(pts)
            if exact_score < best_score:
                best_score = exact_score
                best_pts = pts[:]
    
    if best_pts is None:
        # Fallback
        best_pts = generate_random_set()
        while has_four_collinear(best_pts):
             # Fix
             for quad in itertools.combinations(best_pts, 4):
                if is_collinear(quad[0], quad[1], quad[2]) and is_collinear(quad[0], quad[1], quad[3]):
                    idx = random.choice([0,1,2,3])
                    p_idx = best_pts.index(quad[idx])
                    best_pts[p_idx] = [random.randint(-100, 100), random.randint(-100, 100)]
                    break
    
    return best_pts

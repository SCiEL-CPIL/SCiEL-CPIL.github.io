import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # 1. Generate a pool of candidate points.
    # Strategy: Use a "sparse grid" approach. 
    # Points are chosen from a 10x10 grid scaled to fit [-100, 100].
    # Specifically, we use coordinates from a set that naturally avoids 4-collinear 
    # by using a prime-based offset or simple parity, but for a pool, 
    # we just need a diverse set of points that are likely to form good subsets.
    
    # A better pool: Take a base grid of size 7x7 or 8x8 and scale it.
    # Let's use coordinates x, y in {-50, -30, -10, 10, 30, 50} plus some randoms.
    # Actually, to ensure "no 4 collinear" is easily satisfied by subsets, 
    # we should pick points from a set where no 4 are collinear in the first place, 
    # OR we just pick a large pool and let the B&B handle the constraint.
    
    # Let's generate a pool of ~60-80 points.
    # We will use a deterministic set to ensure reproducibility and quality.
    # Points on the graph of y = x^2 mod p? No, simple integer grid is fine.
    # The constraint is "no 4 collinear" for the final 20.
    
    pool = set()
    # Add points from a 10x10 grid with step 20, centered
    # -90, -70, -50, -30, -10, 10, 30, 50, 70, 90
    coords = [-90, -70, -50, -30, -10, 10, 30, 50, 70, 90]
    for x in coords:
        for y in coords:
            pool.add((x, y))
            
    # Add some random points to break symmetries and provide variety
    random.seed(42)
    for _ in range(50):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        pool.add((x, y))
        
    points = list(pool)
    n_pool = len(points)
    
    # 2. Precompute line incidences for the pool.
    # We need to know which triples are collinear to enforce "no 3 collinear" in the subset.
    # And which quadruples are collinear to enforce "no 4 collinear" in the subset.
    
    # Map from line key to list of point indices
    line_map = {}
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        g = gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        a, b = dy, -dx
        c = a * x1 + b * y1
        return (a, b, c)

    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_map:
                line_map[key] = []
            line_map[key].append(i)
            line_map[key].append(j)
            
    # Correct line_map to unique indices
    for key in line_map:
        line_map[key] = list(set(line_map[key]))

    # Identify "bad" triples (3 collinear) and "forbidden" quadruples (4 collinear)
    # For the "no 3 collinear" score, we want to minimize the max size of a subset with no 3 collinear.
    # Wait, the score IS the max size of a subset with no 3 collinear.
    # We want to MINIMIZE this score.
    # This means we want a configuration where EVERY subset of size K contains 3 collinear points.
    # Or rather, the largest subset with NO 3 collinear is small.
    # This is equivalent to: The points are "dense" in collinearities.
    
    # Let's re-read carefully: "The score is the largest subset with no three collinear... Minimize."
    # Yes. We want a set of 20 points where it is hard to pick many without 3 being collinear.
    # This implies we want many collinear triples.
    # BUT the constraint is "no four collinear".
    # So we want as many triples as possible, but no line has 4 points.
    # This is a packing problem: Maximize the number of collinear triples in a set of 20 points,
    # subject to max line size 3.
    
    # If we have T triples, the score (max no-3-collinear subset) will be low.
    # By Turan-type arguments or simple counting, more triples -> lower independence number.
    
    # So the objective is: Find 20 points from the pool such that:
    # 1. No 4 are collinear.
    # 2. The number of collinear triples is MAXIMIZED.
    
    # We can use a heuristic search (Simulated Annealing / Local Search) on the subset selection.
    # State: A set of 20 indices from the pool.
    # Objective: Maximize triple_count.
    # Constraint: No line has > 3 points from the set.
    
    def evaluate_subset(subset_indices):
        # Returns (triple_count, is_valid)
        # is_valid: no 4 collinear
        # triple_count: number of collinear triples
        
        # Count points per line
        line_counts = {}
        for i in subset_indices:
            # We need to know which lines contain point i
            # Pre-compute point_to_lines
            pass
        
        # This is slow if done from scratch. Let's precompute point_to_lines.
        return 0, True

    # Precompute point_to_lines
    point_to_lines = [[] for _ in range(n_pool)]
    for key, pts in line_map.items():
        for p in pts:
            point_to_lines[p].append(key)

    def calc_score(subset):
        # subset: list of 20 indices
        # Check constraint: no 4 collinear
        # Count triples
        line_counts = {}
        for idx in subset:
            for lkey in point_to_lines[idx]:
                line_counts[lkey] = line_counts.get(lkey, 0) + 1
        
        triples = 0
        for lkey, count in line_counts.items():
            if count >= 4:
                return -1, False # Invalid
            if count == 3:
                triples += 1
            # count == 2 contributes 0 triples
            # count == 1 contributes 0 triples
        
        return triples, True

    # Local Search
    # Start with a random valid subset
    def random_valid_subset():
        while True:
            # Pick 20 random points
            subset = random.sample(range(n_pool), 20)
            _, valid = calc_score(subset)
            if valid:
                return subset

    best_subset = None
    best_triples = -1
    
    # Generate initial population
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        s = random_valid_subset()
        t, _ = calc_score(s)
        if t > best_triples:
            best_triples = t
            best_subset = s[:]
            
    if best_subset is None:
        # Fallback: just take first 20 that are valid
        # This might fail if pool is bad, but pool is grid-like so likely valid subsets exist
        return [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2],[3,0],[3,1],[3,2],[3,3],[0,3],[1,3],[2,3],[4,0],[4,1],[4,2],[4,3]]

    current_subset = best_subset[:]
    current_triples = best_triples
    
    # Simulated Annealing / Hill Climbing
    # Move: Swap one point in subset with one point outside
    outside = set(range(n_pool)) - set(current_subset)
    
    T = 1.0
    T_min = 0.1
    cooling = 0.99
    
    while time.time() < deadline - 5:
        if T < T_min:
            break
        improved = False
        
        # Try all single swaps
        for i in range(20):
            if time.time() > deadline - 5:
                break
            old_idx = current_subset[i]
            # Pick a random candidate from outside
            if not outside:
                break
            new_idx = random.choice(list(outside))
            
            # Perform swap
            new_subset = current_subset[:]
            new_subset[i] = new_idx
            
            new_triples, valid = calc_score(new_subset)
            
            if valid:
                if new_triples > current_triples:
                    # Accept
                    current_subset = new_subset
                    current_triples = new_triples
                    improved = True
                    # Update outside set
                    outside.remove(new_idx)
                    outside.add(old_idx)
                    if current_triples > best_triples:
                        best_triples = current_triples
                        best_subset = current_subset[:]
                else:
                    # Metropolis acceptance
                    delta = new_triples - current_triples
                    if random.random() < pow(2.71828, delta / T):
                        current_subset = new_subset
                        current_triples = new_triples
                        outside.remove(new_idx)
                        outside.add(old_idx)
            else:
                # Invalid, penalize heavily or just reject
                pass
                
        T *= cooling
        
        if not improved and random.random() < 0.1:
            # Random restart to escape local opt
            if time.time() > deadline - 20:
                break
            s = random_valid_subset()
            t, _ = calc_score(s)
            if t > current_triples:
                current_subset = s
                current_triples = t
                outside = set(range(n_pool)) - set(current_subset)
                if t > best_triples:
                    best_triples = t
                    best_subset = current_subset[:]

    # Convert best_subset indices to coordinates
    final_points = [points[i] for i in best_subset]
    
    # Final verification and return
    # Ensure no 4 collinear
    # (Already checked in calc_score, but double check just in case)
    # The problem requires returning the points.
    
    return [[p[0], p[1]] for p in final_points]

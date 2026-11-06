import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    timeout = 110  # Leave 10s margin
    
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])
    
    def has_4_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                count = 0
                for k in range(n):
                    if k != i and k != j and are_collinear(points[i], points[j], points[k]):
                        count += 1
                if count >= 2:
                    return True
        return False

    def get_collinear_triples(points):
        n = len(points)
        triples = set()
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples.add((i, j, k))
        return triples

    def max_no_3_collinear(points):
        """Exact branch and bound for max subset with no 3 collinear."""
        n = len(points)
        if n <= 2:
            return n
        
        triples = get_collinear_triples(points)
        
        # Precompute for each point, which other points are in a collinear triple with it
        # conflict[i] is a bitmask of points that conflict with i
        # Actually, if we pick i, we cannot pick any pair (j, k) such that (i, j, k) is a triple.
        # This means if we pick i, we must remove j and k from consideration? No, we just can't have j and k together with i.
        # The constraint is: no triple {u, v, w} in S.
        
        # Backtracking
        # State: current set S (bitmask), available vertices (bitmask)
        # To speed up, we can use a greedy lower bound and branch on vertices.
        
        all_mask = (1 << n) - 1
        best_size = 0
        
        # Simple recursive search
        # To avoid recursion depth issues and overhead, we can use a stack or optimize the recursion.
        # N=20 is small.
        
        def solve(available_mask, current_size):
            nonlocal best_size
            if current_size + bin(available_mask).count('1') <= best_size:
                return
            
            # If no more vertices, update best
            if available_mask == 0:
                if current_size > best_size:
                    best_size = current_size
                return
            
            # Pick a vertex to branch on. 
            # Heuristic: pick the vertex with the most conflicts (highest degree in the hypergraph sense)
            # Or just pick the lowest set bit.
            
            # Find a vertex
            # Let's pick the first available vertex
            # Better: pick a vertex that is part of many triples among available vertices
            # For simplicity, pick lowest set bit.
            v = (available_mask & -available_mask).bit_length() - 1
            
            # Option 1: Exclude v
            solve(available_mask ^ (1 << v), current_size)
            
            # Option 2: Include v
            # If we include v, we cannot include any pair (u, w) such that (v, u, w) is a collinear triple.
            # This means for every triple (v, u, w) in the set of triples, we cannot have both u and w in the set.
            # This is a 2-SAT like constraint on the remaining vertices? No, it's simpler.
            # If we include v, the remaining set S' must not contain any pair {u, w} that completes a collinear triple with v.
            # So, for each such pair {u, w}, we must exclude at least one of u or w.
            # This is equivalent to saying: The remaining vertices must form an independent set in a graph G_v where edges are {u, w} for triples (v, u, w).
            # So we need the max independent set of G_v restricted to available vertices \ {v}.
            # But wait, the remaining vertices also have constraints among themselves (triples not involving v).
            # So we just recurse, but we must ensure that we don't pick both u and w.
            # This branching factor is high if we just remove u and w.
            
            # Correct approach for hypergraph independence:
            # If we pick v, we remove v from available.
            # Then, for every edge (triple) containing v, say {v, u, w}, we must not pick both u and w.
            # This doesn't mean we remove u or w immediately. It means the subproblem is:
            # Max independent set in the hypergraph restricted to available \ {v}, with the additional constraint that no edge {u, w} (derived from triples with v) is fully included.
            # This is getting complicated for a simple recursive solver.
            
            # Alternative: Standard branch and bound for 3-uniform hypergraph.
            # Branch on a vertex v.
            # 1. Exclude v.
            # 2. Include v. Then, for every edge {v, u, w}, we must branch on excluding u OR excluding w.
            # This is exponential.
            
            # Given N=20, a simpler greedy + local search for the score might be sufficient for evaluation, 
            # or we can use a CP-SAT style solver if available, but we are limited to stdlib/numpy.
            
            # Let's use a different approach for the score: 
            # Since N is small, we can iterate over all subsets? 2^20 = 1M. Too slow for many evaluations.
            # But we can use a heuristic that is very likely to find the max.
            
            # Heuristic for max_no_3_collinear:
            # 1. Greedy: Add points one by one if they don't form a triple with existing points.
            # 2. Local Search: Try to swap points to increase size.
            
            # Actually, for the *outer* search (finding the config), we need the score to be accurate.
            # If the score is underestimated, we might accept a bad config.
            # If the score is overestimated, we might reject a good config.
            # We want to MINIMIZE the score. So we need a LOWER bound on the max subset size?
            # No, we need the EXACT max subset size to know the true score.
            
            # Let's implement a fast exact solver using bitmasks and memoization or just optimized backtracking.
            # The "Include v" branch requires handling the constraints.
            # If we include v, we cannot include any pair (u, w) from triples with v.
            # This is equivalent to finding the max independent set in the graph G_v (edges from triples with v) 
            # intersected with the original hypergraph constraints on the remaining vertices.
            # This is still hard.
            
            # However, N=20 is small. Let's try a simple backtracking that branches on triples.
            # Pick the first triple (u, v, w) in the available set.
            # Branch 1: Exclude u.
            # Branch 2: Exclude v.
            # Branch 3: Exclude w.
            # This ensures no triple is fully included.
            # This is a standard algorithm for hitting sets / independent sets in hypergraphs.
            
            pass # Placeholder

        # Let's replace the inner solver with a robust one.
        # We'll use the triple-branching method.
        
        def solve_triple(available_mask):
            nonlocal best_size
            # Pruning
            if bin(available_mask).count('1') <= 2:
                size = bin(available_mask).count('1')
                if size > best_size:
                    best_size = size
                return
            
            # Find first triple in available_mask
            # We need to quickly find a triple.
            # Precompute list of triples.
            # Iterate through triples to find one where all 3 bits are in available_mask.
            
            triple_found = None
            for t in triples:
                i, j, k = t
                if (available_mask & (1 << i)) and (available_mask & (1 << j)) and (available_mask & (1 << k)):
                    triple_found = t
                    break
            
            if triple_found is None:
                # No triples, so all remaining points can be included
                size = bin(available_mask).count('1')
                if size > best_size:
                    best_size = size
                return
            
            i, j, k = triple_found
            
            # Branch 1: Exclude i
            solve_triple(available_mask & ~(1 << i))
            
            # Branch 2: Exclude j
            solve_triple(available_mask & ~(1 << j))
            
            # Branch 3: Exclude k
            solve_triple(available_mask & ~(1 << k))
            
        solve_triple(all_mask)
        return best_size

    # Optimization: Precompute triples list for speed in solve_triple
    # The solve_triple function iterates through all triples to find one in the mask.
    # This is O(E) per node. E=1140. Depth=20. 1140 * 2^20 is too slow.
    # We need to be smarter.
    
    # Alternative: Use a greedy approximation for the score during the search, 
    # and only use the exact solver for the final candidate or when a new best is found.
    # Or, use a very fast heuristic.
    
    def approx_score(points):
        # Greedy + Local Search
        n = len(points)
        selected = []
        # Sort by degree? Or random order?
        # Let's try random order multiple times
        best_sel = 0
        for _ in range(10):
            indices = list(range(n))
            random.shuffle(indices)
            sel = []
            for i in indices:
                can_add = True
                for j in range(len(sel)):
                    for k in range(j+1, len(sel)):
                        if are_collinear(points[i], points[sel[j]], points[sel[k]]):
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    sel.append(i)
            if len(sel) > best_sel:
                best_sel = len(sel)
        return best_sel

    # We will use approx_score for the search loop to save time.
    # We will verify with exact score if a new best is found.
    
    def exact_score(points):
        n = len(points)
        if n <= 2:
            return n
        triples = get_collinear_triples(points)
        all_mask = (1 << n) - 1
        best_size = 0
        
        # Use the triple branching but with pruning
        # To speed up, we can use a stack
        stack = [(all_mask, 0)]
        while stack:
            mask, size = stack.pop()
            if size + bin(mask).count('1') <= best_size:
                continue
            if mask == 0:
                if size > best_size:
                    best_size = size
                continue
            
            # Find a triple in mask
            triple_found = None
            for t in triples:
                i, j, k = t
                if (mask & (1 << i)) and (mask & (1 << j)) and (mask & (1 << k)):
                    triple_found = t
                    break
            
            if triple_found is None:
                size += bin(mask).count('1')
                if size > best_size:
                    best_size = size
                continue
            
            i, j, k = triple_found
            stack.append((mask & ~(1 << i), size))
            stack.append((mask & ~(1 << j), size))
            stack.append((mask & ~(1 << k), size))
            
        return best_size

    best_score = 20
    best_points = None
    
    # Initial configurations
    # 1. Random
    # 2. Structured: Points on lines
    
    random.seed(42)
    
    def generate_random_points():
        points = []
        used = set()
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                used.add((x, y))
                points.append([x, y])
        return points
    
    def generate_structured_points():
        # Try to create many collinear triples
        # Place points on 6 lines, 3 points each
        # Line 1: y=0, x in {-30, 0, 30}
        # Line 2: y=10, x in {-20, 10, 40}
        # ...
        points = []
        # Define lines
        lines = [
            [( -30, 0), ( 0, 0), ( 30, 0)],
            [( -20, 10), ( 10, 10), ( 40, 10)],
            [( -10, 20), ( 20, 20), ( 50, 20)],
            [(  0, 30), ( 30, 30), ( 60, 30)],
            [( 10, 40), ( 40, 40), ( 70, 40)],
            [( 20, 50), ( 50, 50), ( 80, 50)]
        ]
        for line in lines:
            for p in line:
                points.append(list(p))
        # We have 18 points. Add 2 more randomly
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if not any(p[0] == x and p[1] == y for p in points):
                points.append([x, y])
        return points

    candidates = [generate_random_points(), generate_structured_points()]
    
    # Add more random candidates
    for _ in range(10):
        candidates.append(generate_random_points())
        
    for pts in candidates:
        if has_4_collinear(pts):
            continue
        s = approx_score(pts)
        if s < best_score:
            # Verify with exact score
            s_exact = exact_score(pts)
            if s_exact < best_score:
                best_score = s_exact
                best_points = [p[:] for p in pts]
                
    # Local Search
    if best_points is None:
        best_points = generate_random_points()
        best_score = exact_score(best_points)
        
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    while time.time() - start_time < timeout:
        # Pick a random point to move
        i = random.randint(0, 19)
        old_x, old_y = current_points[i]
        
        # Try a few random moves
        for _ in range(10):
            new_x = random.randint(max(-100, old_x - 20), min(100, old_x + 20))
            new_y = random.randint(max(-100, old_y - 20), min(100, old_y + 20))
            
            if (new_x, new_y) == (old_x, old_y):
                continue
                
            # Check collision with other points
            conflict = False
            for j in range(20):
                if j != i and current_points[j][0] == new_x and current_points[j][1] == new_y:
                    conflict = True
                    break
            if conflict:
                continue
                
            current_points[i] = [new_x, new_y]
            
            if has_4_collinear(current_points):
                current_points[i] = [old_x, old_y]
                continue
                
            s_approx = approx_score(current_points)
            if s_approx < current_score:
                s_exact = exact_score(current_points)
                if s_exact < current_score:
                    current_score = s_exact
                    best_score = s_exact
                    best_points = [p[:] for p in current_points]
                else:
                    current_points[i] = [old_x, old_y]
            else:
                current_points[i] = [old_x, old_y]
                
            if time.time() - start_time > timeout:
                break
                
        if time.time() - start_time > timeout:
            break
            
    return best_points

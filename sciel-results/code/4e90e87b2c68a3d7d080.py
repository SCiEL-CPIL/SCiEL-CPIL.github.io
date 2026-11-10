import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s buffer
    
    # Grid bounds
    min_coord = -100
    max_coord = 100
    
    def are_collinear(p1, p2, p3):
        # (x2-x1)(y3-y1) == (y2-y1)(x3-x1)
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])
        
    def has_four_collinear(points):
        n = len(points)
        # Check all lines defined by pairs
        # To be efficient, map lines to counts
        line_counts = defaultdict(int)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                # Normalize line representation
                # A x + B y + C = 0
                A = y1 - y2
                B = x2 - x1
                C = x1*y2 - x2*y1
                # Normalize
                if A == 0 and B == 0:
                    continue
                if A < 0 or (A == 0 and B < 0):
                    A, B, C = -A, -B, -C
                elif A == 0 and B < 0:
                    A, B, C = -A, -B, -C
                elif A == 0 and B == 0:
                    continue
                # Simple normalization: divide by gcd
                import math
                g = math.gcd(A, B)
                if g == 0:
                    g = math.gcd(A, C)
                if g == 0:
                    g = math.gcd(B, C)
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                line_counts[(A, B, C)] += 1
        # If any line has count >= 3, it means at least 4 points are on it?
        # No, count is number of pairs.
        # If k points are on a line, there are k*(k-1)/2 pairs.
        # If count >= 3, possible k=3 (3 pairs) or k=4 (6 pairs).
        # We need to distinguish 3 points (3 pairs) from 4 points (6 pairs).
        # If count is 3, it could be 3 points.
        # If count is 6, it could be 4 points.
        # Actually, if we just check all subsets of 4? \binom{20}{4} = 4845. Very fast.
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if are_collinear(points[i], points[j], points[k]):
                            if are_collinear(points[i], points[j], points[l]):
                                return True
        return False
        
    def get_triples(points):
        triples = []
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples.append((i, j, k))
        return triples
        
    def calc_alpha(points):
        triples = get_triples(points)
        n = len(points)
        # Branch and bound for maximum independent set in 3-uniform hypergraph
        # Vertices: 0..n-1
        # Edges: triples
        # We want max set S such that no edge is subset of S.
        
        # Precompute for each vertex, the set of edges it belongs to?
        # Or just use a bitmask? n=20 fits in int.
        
        # List of triples as bitmasks
        triple_masks = []
        for t in triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_masks.append(mask)
            
        # If no triples, alpha = n
        if not triple_masks:
            return n
            
        # Heuristic: Sort triples by something?
        
        best = 0
        # Recursive search
        # State: remaining vertices mask, current selected mask
        # Optimization: If current selected + count(remaining) <= best, prune.
        
        def backtrack(remaining_mask, current_mask, current_count):
            nonlocal best
            if current_count + bin(remaining_mask).count('1') <= best:
                return
            if remaining_mask == 0:
                if current_count > best:
                    best = current_count
                return
                
            # Pick a vertex to decide
            # Pick the vertex with highest degree in the remaining hypergraph?
            # Or just the lowest index for determinism.
            # Let's pick the lowest set bit.
            v_bit = remaining_mask & -remaining_mask
            v_idx = v_bit.bit_length() - 1
            
            # Option 1: Exclude v
            backtrack(remaining_mask ^ v_bit, current_mask, current_count)
            
            # Option 2: Include v
            # If we include v, we cannot include any other vertex that forms a triple with v and a third vertex in remaining?
            # Actually, if we include v, we must ensure no triple is fully selected.
            # Any triple containing v and two other vertices in `remaining` would be invalid if we select those two.
            # So, if we select v, we must remove from `remaining` any pair of vertices that form a triple with v?
            # No, we just can't select the other two simultaneously.
            # This makes the state space complex.
            
            # Simpler approach for n=20:
            # Iterate through all subsets? 2^20 = 1M.
            # Check each subset if it contains a triple.
            # This is O(2^n * M). 1M * 1000 = 1B. Too slow in Python.
            
            # Let's stick to simple backtracking on vertices.
            # When adding v, we don't remove anything from remaining immediately, 
            # but we check validity at the leaf? No, that's too late.
            
            # Alternative: 
            # The condition "no triple" means for every triple {a,b,c}, not (a in S and b in S and c in S).
            # This is a 3-SAT style constraint.
            
            # Let's try a different B&B:
            # Sort vertices by degree.
            # This might be slow.
            
            # Given the small n, maybe we can use a greedy + local search for alpha too?
            # But we need exact alpha for the score.
            
            # Let's optimize the backtracking.
            # If we include v, we cannot include any (u, w) such that {v, u, w} is a triple.
            # This means the remaining problem is not just a subset of vertices, but a modified hypergraph.
            
            # Actually, since n=20 is small, we can use a simple recursive check:
            # def is_independent(mask):
            #    for t in triple_masks:
            #       if (mask & t) == t: return False
            #    return True
            
            # We want max popcount(mask) such that is_independent(mask).
            
            # We can iterate masks from 2^n - 1 down to 0?
            # No, we want max.
            
            # Let's use the property that if we fix a vertex, we can prune.
            
            pass # Placeholder for logic
            
        # Let's implement a faster exact solver for n=20.
        # Use bitmasks for vertices.
        # Precompute for each mask if it's independent? No, 1M masks is okay if check is fast.
        # Check is O(M). 1M * 1000 = 1B ops. Too slow.
        
        # Better B&B:
        # Sort triples.
        # Pick a triple. Either exclude one of its vertices.
        
        def solve_alpha():
            # Use a simple backtracking that picks a vertex and branches on include/exclude.
            # To make it fast, we can use the fact that if we include a vertex, 
            # we must exclude at least one vertex from every triple containing it.
            
            # Let's try a different approach:
            # The independence number is at most n.
            # We can check if there exists an independent set of size k.
            # Binary search on k?
            
            # For k=20, 19, ...
            # Check if independent set of size k exists.
            # This is still hard.
            
            # Let's go back to local search for the main problem, and use a heuristic for alpha?
            # No, the score is exact.
            
            # However, for n=20, we can precompute the "conflict" structure.
            
            # Let's try a simple recursive search with strong pruning.
            # State: (available_mask, selected_mask)
            # Pruning: if popcount(selected) + popcount(available) <= best: return
            # Also, if selected is not independent: return (should not happen)
            
            # To speed up "is selected independent", we can maintain a count of "active" triples?
            
            # Let's just use the bitmask iteration for the *evaluation* of a specific configuration,
            # but optimize the inner loop.
            
            # Actually, 2^20 is 1,048,576.
            # If we just loop i from 0 to 2^20-1:
            #   mask = i
            #   if bin(mask).count('1') <= best: continue
            #   check if mask is independent.
            #   if yes: best = popcount(mask)
            
            # The check `is_independent(mask)` takes O(M).
            # Total time: 1M * M.
            # If M is small (e.g., 10), it's 10M ops. Fast.
            # If M is large (e.g., 1000), it's 1B ops. Slow (~10s).
            
            # In our local search, M will grow as we create triples.
            # But we are minimizing alpha, so we want many triples.
            # So M will be large.
            
            # We need a faster alpha calculator.
            
            # Let's use the branch and bound on vertices.
            
            def calc_alpha_fast(points):
                triples = get_triples(points)
                n = len(points)
                if not triples:
                    return n
                    
                # Map vertex to list of triples containing it
                vertex_triples = [[] for _ in range(n)]
                for i, t in enumerate(triples):
                    for v in t:
                        vertex_triples[v].append(i)
                        
                best = 0
                
                # Recursive function
                # current_mask: vertices selected so far
                # remaining_mask: vertices not yet decided
                # blocked_mask: vertices that cannot be selected because they would form a triple with current_mask?
                # No, if we select v, we block pairs.
                
                # Let's use a different state:
                # We build the set S.
                # If we add v to S, we must ensure no triple is completed.
                # A triple is completed if all 3 are in S.
                
                # Simple B&B:
                def backtrack(remaining, current_count, current_mask):
                    nonlocal best
                    if current_count + bin(remaining).count('1') <= best:
                        return
                    if remaining == 0:
                        if current_count > best:
                            best = current_count
                        return
                        
                    # Pick a vertex from remaining
                    # Heuristic: pick vertex with most constraints?
                    v_bit = remaining & -remaining
                    v = v_bit.bit_length() - 1
                    
                    # Option 1: Don't pick v
                    backtrack(remaining ^ v_bit, current_count, current_mask)
                    
                    # Option 2: Pick v
                    # Check if picking v creates a triple with two other vertices in `remaining`?
                    # No, we don't know if they will be picked.
                    # But if we pick v, we *cannot* pick any pair (u, w) in `remaining` 
                    # such that {v, u, w} is a triple.
                    # This implies that for every triple {v, u, w}, we must exclude at least one of u, w.
                    # This is a constraint on the remaining subproblem.
                    
                    # This is getting complicated.
                    # Given the time, let's use the 2^n iteration but optimize the check.
                    
                    pass
                    
                # Let's try the 2^n iteration with early termination.
                # Sort masks by popcount descending?
                # No, just iterate and keep best.
                
                # To make it fast, we can precompute `is_independent` for all masks?
                # No, memory.
                
                # Let's use the fact that if a mask contains a triple, it's not independent.
                # We can check triples first.
                
                # If M is small, 2^n * M is fine.
                # If M is large, it's slow.
                # But if M is large, alpha is likely small, so we might find a small best quickly?
                # No, we need to prove max.
                
                # Let's assume M is manageable.
                
                # Optimization: If we find an independent set of size k, 
                # we can prune masks with popcount <= k.
                
                # Let's implement this.
                max_mask = (1 << n) - 1
                # Iterate masks
                # To find max, we can iterate popcount from n down to 1?
                # Number of masks with popcount k is C(n, k).
                # C(20, 10) is 184k. C(20, 5) is 15k.
                # Sum is 1M.
                
                # Let's just iterate all masks and track max.
                # To speed up, we can skip masks that are subsets of known non-independent masks?
                
                # Let's just do the simple loop.
                # 1M iterations * (check triples).
                # If we break early in check, it's fast.
                
                for mask in range(1, max_mask + 1):
                    pop = bin(mask).count('1')
                    if pop <= best:
                        continue
                    # Check if independent
                    independent = True
                    for t in triple_masks:
                        if (mask & t) == t:
                            independent = False
                            break
                    if independent:
                        best = pop
                        # If best == n, break? No, n is max.
                        # If best is close to n, we can prune more.
                        
                return best
                
            return calc_alpha_fast(points)
            
    # Initialization
    # Start with random points, ensuring no 4 collinear.
    points = []
    while len(points) < 20:
        x = random.randint(min_coord, max_coord)
        y = random.randint(min_coord, max_coord)
        p = [x, y]
        if p in points:
            continue
        # Check no 4 collinear with existing
        points.append(p)
        if has_four_collinear(points):
            points.pop()
            # Try again
            continue
        else:
            points.pop() # Wait, I appended then popped?
            # Logic:
            # points.append(p)
            # if has_four: points.pop()
            # else: keep it?
            # I need to keep it if valid.
            
    # Redo init logic
    points = []
    attempts = 0
    while len(points) < 20:
        if attempts > 10000: break
        attempts += 1
        x = random.randint(min_coord, max_coord)
        y = random.randint(min_coord, max_coord)
        p = [x, y]
        if p in points:
            continue
        points.append(p)
        if has_four_collinear(points):
            points.pop()
            
    if len(points) < 20:
        # Fallback to parabola if random fails (unlikely)
        points = [[x, x*x] for x in range(-10, 10)]
        
    current_alpha = calc_alpha(points)
    best_alpha = current_alpha
    best_points = points[:]
    
    # Local Search
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_p = points[idx]
        
        # Try new position
        new_x = random.randint(min_coord, max_coord)
        new_y = random.randint(min_coord, max_coord)
        new_p = [new_x, new_y]
        
        if new_p == old_p:
            continue
            
        # Check if new_p is already in points (excluding idx)
        if any(p == new_p for i, p in enumerate(points) if i != idx):
            continue
            
        # Swap
        points[idx] = new_p
        
        # Check validity
        if has_four_collinear(points):
            points[idx] = old_p
            continue
            
        # Calculate new alpha
        new_alpha = calc_alpha(points)
        
        if new_alpha < best_alpha:
            best_alpha = new_alpha
            best_points = points[:]
            current_alpha = new_alpha
        elif new_alpha == current_alpha:
            # Accept with some probability to explore
            if random.random() < 0.1:
                current_alpha = new_alpha
            else:
                points[idx] = old_p
        else:
            # Revert
            points[idx] = old_p
            
    return best_points

construct()

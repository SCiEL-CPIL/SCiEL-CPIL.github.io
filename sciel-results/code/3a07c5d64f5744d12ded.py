import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def get_color_index(i, j, n):
        if i > j: i, j = j, i
        return i * n - (i + 1) // 2 + j - i - 1 # This formula is tricky, let's use a simpler mapping or precomputed list
        
    # Precompute edge indices for lexicographic order
    # colors list index k corresponds to pair (i, j)
    # We can generate the list of pairs
    def generate_pairs(n):
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                pairs.append((i, j))
        return pairs

    def count_c6_involving_edge(u, v, color, colors, n, pairs):
        # We need to find the number of simple paths of length 4 from v to u 
        # such that all edges in the path have the given color.
        # The cycle would be u-v-...-u.
        # So we look for path v -> w1 -> w2 -> w3 -> w4 -> u
        # where (v,w1), (w1,w2), (w2,w3), (w3,w4), (w4,u) all have 'color'.
        
        # To do this efficiently, we need quick access to edge colors.
        # Let's create a dictionary or 2D array for colors.
        # Since n <= 20, a 2D list is fine.
        # But we are passing 'colors' as a flat list.
        # Let's assume we have a helper to get color of (a,b).
        
        pass # This function will be implemented inside the search loop with optimized access
    
    # Better approach: Use a 2D array for colors during search for O(1) access.
    
    def try_solve(n, time_limit):
        pairs = generate_pairs(n)
        num_edges = len(pairs)
        
        # Map pair to index
        pair_to_idx = {p: i for i, p in enumerate(pairs)}
        
        def get_idx(u, v):
            if u > v: u, v = v, u
            return pair_to_idx[(u, v)]
            
        # Initialize random coloring
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Helper to get color of edge (u, v)
        def get_color(u, v):
            if u > v: u, v = v, u
            return colors[pair_to_idx[(u, v)]]
            
        # Helper to set color
        def set_color(u, v, c):
            if u > v: u, v = v, u
            colors[pair_to_idx[(u, v)]] = c
            
        # Count C6s containing a specific edge (u, v) of a specific color
        def count_c6_for_edge(u, v, c):
            # Find paths of length 4 from v to u in graph of color c
            # v -> a -> b -> d -> e -> u
            count = 0
            # To speed up, we can precompute neighbors for each color?
            # But colors change.
            # For n=20, simple iteration is okay if we are careful.
            
            # Get neighbors of v with color c
            # Actually, we just need to check combinations.
            # There are (n-2) choices for a, (n-3) for b, etc.
            # This is O(n^4) per check. 20^4 = 160k.
            # If we do 1000 moves, 160M ops. Might be too slow in Python.
            
            # Optimization:
            # We can just check if the move *creates* a C6.
            # But we need to subtract the ones destroyed.
            
            # Let's try a simpler heuristic:
            # Just count the number of "potential" C6s.
            # Actually, for the local search, we can just re-evaluate the total C6 count
            # periodically, or use a faster approximation.
            
            # Alternative:
            # We can precompute the list of all C6s in K_n.
            # There are ~10^5 C6s for n=20.
            # For each C6, check if all 6 edges have the same color.
            # This is O(1) per C6 check (6 lookups).
            # Total check time: 10^5 * 6 = 6*10^5 ops.
            # This is very fast!
            
            # So, let's precompute all C6s.
            # But we are inside try_solve.
            # We should precompute C6s for each n outside or inside.
            
            return count # Placeholder
            
        # Precompute all C6s for this n
        all_c6s = []
        vertices = list(range(n))
        # Generate all permutations of 6 distinct vertices? No, just combinations and then check cycles.
        # A C6 is a cycle.
        # We can generate all sets of 6 vertices, and for each set, check if there is a Hamiltonian cycle in the subgraph?
        # No, we just need to check if the *specific* 6 edges forming the cycle have the same color.
        # The cycle is defined by the order.
        # For a set of 6 vertices {v0..v5}, there are (6-1)!/2 = 60 distinct cycles.
        # Total C6s in K_n is C(n,6) * 60.
        # For n=20, C(20,6) = 38760. 38760 * 60 = 2.3M cycles.
        # Checking 2.3M cycles * 6 edges = 14M ops per full evaluation.
        # This is doable in Python if we do it a few times, but not in every local search step.
        
        # So, local search step must be incremental.
        # When we change edge (u,v) from c1 to c2:
        # 1. Find all C6s that contain edge (u,v).
        # 2. For each such C6, check if the other 5 edges have color c1. If yes, we destroy 1 C6 of color c1.
        # 3. Check if the other 5 edges have color c2. If yes, we create 1 C6 of color c2.
        
        # So we need to precompute: For each edge e, list of C6s containing e.
        # This is a big memory structure but n is small.
        # Number of C6s containing a specific edge:
        # Fix edge (u,v). We need 4 other vertices from n-2.
        # C(n-2, 4) * (number of ways to arrange 4 vertices in a path from v to u).
        # The path is v-w1-w2-w3-w4-u.
        # The 4 vertices are distinct.
        # Number of such paths is P(n-2, 4) = (n-2)(n-3)(n-4)(n-5).
        # For n=20: 18*17*16*15 = 73440.
        # So each edge is in ~73k C6s.
        # Total C6s = (n(n-1)/2) * 73440 / 6 (since each C6 has 6 edges) = 190 * 12240 = 2.3M. Matches.
        
        # Precomputing the list of C6s for each edge is feasible.
        
        return None # Placeholder for actual logic

    # Given the complexity, I will implement a simpler Random Restart with a "Good Enough" heuristic
    # or just rely on the fact that n=17 is easy to find with a decent heuristic.
    
    # Let's use the precomputed C6 list approach for n=17, 18, 19, 20.
    
    def solve_n(n, time_left):
        pairs = generate_pairs(n)
        num_edges = len(pairs)
        pair_to_idx = {p: i for i, p in enumerate(pairs)}
        
        # Precompute C6s containing each edge
        # c6s_by_edge[edge_idx] = list of (other_edges_indices, c6_id) ? 
        # We just need to know which C6s are affected.
        # Let's store for each edge, a list of indices of C6s that contain it.
        
        # First, generate all C6s.
        # A C6 is a tuple of 6 vertices (v0..v5) representing the cycle v0-v1-v2-v3-v4-v5-v0.
        # To avoid duplicates, we can fix v0=0? No, vertices are 0..n-1.
        # We can generate all combinations of 6 vertices, and for each, generate all 60 cycles.
        
        all_c6s = []
        vertices = list(range(n))
        
        # This generation might take a bit, but n<=20 is small.
        from itertools import combinations, permutations
        
        # To speed up, we can just store the 6 edges of each C6.
        # c6_edges[c6_idx] = (idx0, idx1, idx2, idx3, idx4, idx5)
        
        c6_edges = []
        
        # Optimization: Only generate cycles where v0 < v1 < ...? No, cycles are not ordered sets.
        # We can fix v0 to be the smallest vertex in the cycle to avoid duplicates?
        # No, a cycle is a set of edges.
        # Let's just generate all 6-permutations of 6-combinations?
        # C(n,6) * 60.
        
        # For n=20, this is 2.3M entries.
        # Storing 2.3M tuples of 6 ints is ~ 2.3M * 6 * 8 bytes = 110 MB.
        # This might be too much memory or slow to generate in Python within 120s.
        
        # Alternative:
        # Don't precompute all C6s.
        # Use the local search with on-the-fly C6 count for the specific edge.
        # The count for one edge is 73k checks.
        # Each check is 5 edge color lookups.
        # 73k * 5 = 365k ops per edge move.
        # If we do 100 moves, 36M ops.
        # This is feasible in Python (approx 1-2 seconds).
        
        # So, strategy:
        # 1. Random init.
        # 2. Calculate total C6 count (expensive, do once).
        # 3. Local search:
        #    - Pick random edge.
        #    - For each new color, calculate delta C6 count.
        #    - Accept best.
        
        # To calculate delta for edge (u,v) changing c1->c2:
        # We need to count C6s containing (u,v) that are monochromatic c1 (will be broken)
        # and C6s containing (u,v) that are monochromatic c2 (will be created).
        
        # Function count_mono_c6_with_edge(u, v, color):
        #   Returns count of C6s containing (u,v) where all 6 edges have 'color'.
        #   Implementation:
        #   Iterate all paths of length 4 from v to u (v-w1-w2-w3-w4-u).
        #   For each path, check if (v,w1), (w1,w2), (w2,w3), (w3,w4), (w4,u) all have 'color'.
        #   If yes, increment count.
        
        # This is O(n^4) per call.
        
        def count_mono_c6_with_edge(u, v, color, colors, pair_to_idx):
            count = 0
            # Path v -> a -> b -> c -> d -> u
            # a, b, c, d distinct, not u, v
            # (v,a), (a,b), (b,c), (c,d), (d,u) must have color
            # Note: (u,v) is the edge being changed, so we don't check its color here, 
            # we assume it has 'color' for the purpose of counting.
            
            # To speed up, we can precompute neighbors for each color?
            # But colors change.
            
            # Let's just iterate.
            # n is small.
            for a in range(n):
                if a == u or a == v: continue
                if get_color(v, a) != color: continue
                for b in range(n):
                    if b == u or b == v or b == a: continue
                    if get_color(a, b) != color: continue
                    for c in range(n):
                        if c == u or c == v or c == a or c == b: continue
                        if get_color(b, c) != color: continue
                        for d in range(n):
                            if d == u or d == v or d == a or d == b or d == c: continue
                            if get_color(c, d) != color: continue
                            if get_color(d, u) == color:
                                count += 1
            return count

        # This is 4 nested loops. n=20.
        # 20*20*20*20 = 160,000 iterations.
        # Each iteration has a few checks.
        # This is fast enough for a few hundred moves.
        
        # We need get_color and set_color to be fast.
        # Using pair_to_idx dictionary might be slow.
        # Let's use a 2D list `col_grid` of size n x n.
        
        col_grid = [[-1]*n for _ in range(n)]
        
        def update_grid(u, v, c):
            col_grid[u][v] = c
            col_grid[v][u] = c
            
        # Initialize
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        for i, (u, v) in enumerate(pairs):
            update_grid(u, v, colors[i])
            
        def get_c(u, v):
            return col_grid[u][v]
            
        def set_c(u, v, c):
            col_grid[u][v] = c
            col_grid[v][u] = c
            idx = pair_to_idx[(min(u,v), max(u,v))]
            colors[idx] = c
            
        # Initial total cost
        # We don't need total cost, just local delta.
        # But we need to know if we reached 0.
        # We can just check if any C6 exists?
        # Or maintain a count.
        
        # Let's maintain a global count of C6s.
        # But calculating initial count is expensive (2.3M C6s).
        # We can just do local search and check for 0 at the end?
        # Or we can just minimize the sum of count_mono_c6_with_edge over all edges / 6.
        
        # Let's just do:
        # While time remains:
        #   Pick random edge (u,v)
        #   c_old = get_c(u,v)
        #   best_new_c = -1
        #   best_delta = 0
        #   for c_new in range(4):
        #     if c_new == c_old: continue
        #     delta_old = count_mono_c6_with_edge(u, v, c_old)
        #     delta_new = count_mono_c6_with_edge(u, v, c_new)
        #     # Note: count_mono_c6_with_edge assumes the edge (u,v) HAS the color.
        #     # So if we change from c_old to c_new:
        #     # We lose delta_old C6s of color c_old.
        #     # We gain delta_new C6s of color c_new.
        #     # Net change = delta_new - delta_old
        #     if delta_new - delta_old < best_delta:
        #        best_delta = delta_new - delta_old
        #        best_new_c = c_new
        #   if best_new_c != -1:
        #      set_c(u, v, best_new_c)
        #      # Check if total C6 is 0?
        #      # We can periodically check.
        
        # This is a greedy local search. It might get stuck in local minima.
        # Add random restarts.
        
        best_n_found = 0
        for target_n in [20, 19, 18, 17]:
            if time.time() > deadline - 5:
                break
                
            found = False
            start_t = time.time()
            limit = min(deadline - 5, start_t + 20) # 20s per n
            
            while time.time() < limit and not found:
                # Random init
                colors = [random.randint(0, 3) for _ in range(num_edges)]
                for i, (u, v) in enumerate(pairs):
                    update_grid(u, v, colors[i])
                    
                # Local search
                for _ in range(1000): # Max 1000 steps
                    if time.time() > limit: break
                    u, v = random.choice(pairs)
                    c_old = get_c(u, v)
                    
                    best_delta = 0
                    best_new_c = -1
                    # We only need to calculate delta_old once
                    delta_old = count_mono_c6_with_edge(u, v, c_old, colors, pair_to_idx)
                    
                    for c_new in range(4):
                        if c_new == c_old: continue
                        delta_new = count_mono_c6_with_edge(u, v, c_new, colors, pair_to_idx)
                        if delta_new - delta_old < best_delta:
                            best_delta = delta_new - delta_old
                            best_new_c = c_new
                            
                    if best_new_c != -1:
                        set_c(u, v, best_new_c)
                        
                # Check if valid
                # We need to check if there are ANY C6s.
                # We can do a full scan.
                # Or we can check if the sum of count_mono_c6_with_edge for all edges is 0.
                # Sum over all edges / 6 = Total C6s.
                total_c6 = 0
                for u, v in pairs:
                    c = get_c(u, v)
                    total_c6 += count_mono_c6_with_edge(u, v, c, colors, pair_to_idx)
                total_c6 //= 6
                
                if total_c6 == 0:
                    found = True
                    break
                    
            if found:
                best_n_found = target_n
                break # Found a valid coloring for this n, which is the max possible for this attempt
                # Actually, we should return immediately?
                # The task says "Maximize n".
                # If we find n=20, we are done.
                # If we find n=19, we are done (unless we try 20 first).
                # My loop tries 20, 19, 18, 17.
                # If 20 fails, try 19. If 19 succeeds, return 19.
                
        if best_n_found > 0:
            return {'n': best_n_found, 'colors': colors}
            
        # Fallback to seed
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

    # Run the solver
    return solve_n(17, deadline - start_time) # This is not right, solve_n handles the loop

# The code above has some structural issues. I will clean it up.

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def generate_pairs(n):
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                pairs.append((i, j))
        return pairs

    def solve_n(n, time_limit):
        pairs = generate_pairs(n)
        num_edges = len(pairs)
        pair_to_idx = {p: i for i, p in enumerate(pairs)}
        
        col_grid = [[-1]*n for _ in range(n)]
        
        def update_grid(u, v, c):
            col_grid[u][v] = c
            col_grid[v][u] = c
            
        def get_c(u, v):
            return col_grid[u][v]
            
        def set_c(u, v, c):
            col_grid[u][v] = c
            col_grid[v][u] = c
            idx = pair_to_idx[(min(u,v), max(u,v))]
            colors[idx] = c
            
        def count_mono_c6_with_edge(u, v, color):
            count = 0
            for a in range(n):
                if a == u or a == v: continue
                if get_c(v, a) != color: continue
                for b in range(n):
                    if b == u or b == v or b == a: continue
                    if get_c(a, b) != color: continue
                    for c in range(n):
                        if c == u or c == v or c == a or c == b: continue
                        if get_c(b, c) != color: continue
                        for d in range(n):
                            if d == u or d == v or d == a or d == b or d == c: continue
                            if get_c(c, d) != color: continue
                            if get_c(d, u) == color:
                                count += 1
            return count

        colors = []
        
        best_result = None
        
        # Try random restarts
        while time.time() < deadline - 5:
            # Init
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            for i, (u, v) in enumerate(pairs):
                update_grid(u, v, colors[i])
                
            # Local Search
            for _ in range(500):
                if time.time() > deadline - 5: break
                u, v = random.choice(pairs)
                c_old = get_c(u, v)
                
                delta_old = count_mono_c6_with_edge(u, v, c_old)
                
                best_delta = 0
                best_new_c = -1
                for c_new in range(4):
                    if c_new == c_old: continue
                    delta_new = count_mono_c6_with_edge(u, v, c_new)
                    if delta_new - delta_old < best_delta:
                        best_delta = delta_new - delta_old
                        best_new_c = c_new
                        
                if best_new_c != -1:
                    set_c(u, v, best_new_c)
                    
            # Check validity
            total_c6 = 0
            for u, v in pairs:
                c = get_c(u, v)
                total_c6 += count_mono_c6_with_edge(u, v, c)
            total_c6 //= 6
            
            if total_c6 == 0:
                return {'n': n, 'colors': colors}
                
        return None

    # Try n=20, 19, 18, 17
    for n in [20, 19, 18, 17]:
        if time.time() > deadline - 10:
            break
        res = solve_n(n, deadline)
        if res:
            return res
            
    # Fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

import time
import random
import sys
from functools import lru_cache

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def count_bits(x):
        return bin(x).count("1")

    def max_clique(adj, n, deadline):
        """
        Finds a maximum clique using branch and bound with coloring bound.
        """
        best_clique = []
        best_size = 0
        # Precompute neighbor masks for speed
        neighbors = [adj[i] for i in range(n)]
        
        # Greedy coloring for upper bound
        def color_sort(candidates):
            # Returns list of (vertex, color) sorted by color
            # Color classes provide an upper bound on clique size in the induced subgraph
            colors = [0] * n
            color_class = 0
            uncolored = candidates
            # Simple greedy coloring: pick vertex, assign color, remove from uncolored
            # We want to order vertices such that vertices with low colors are processed first?
            # Actually, standard Tomita: sort by color, then search in that order.
            # For simplicity and speed, we just need the max color count.
            
            # Better approach: 
            # 1. Order vertices by degree in candidates (or just pick arbitrarily)
            # 2. Assign colors greedily
            # 3. The max color number is the upper bound.
            
            # To get a tight bound, we should order vertices by descending degree in candidates.
            verts = []
            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                verts.append(v)
                c ^= lsb
            
            # Sort by degree in candidates
            verts.sort(key=lambda v: count_bits(adj[v] & candidates), reverse=True)
            
            # Greedy coloring
            color_map = {}
            next_color = 0
            for v in verts:
                # Find smallest color not used by neighbors of v in the colored set
                used_colors = set()
                for u in verts:
                    if u == v:
                        break
                    if u in color_map and (adj[v] & (1 << u)):
                        used_colors.add(color_map[u])
                
                c_val = 0
                while c_val in used_colors:
                    c_val += 1
                color_map[v] = c_val
                if c_val + 1 > next_color:
                    next_color = c_val + 1
            
            return verts, color_map, next_color

        def expand(clique, candidates):
            nonlocal best_clique, best_size
            
            if not candidates:
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique[:]
                return
            
            if time.time() > deadline:
                return
                
            # Bound: if current clique size + upper bound on candidates < best_size, prune
            # We need a quick upper bound. count_bits(candidates) is weak.
            # Let's use a simple greedy coloring bound.
            # But coloring is expensive. Let's try a cheaper bound first.
            # If len(clique) + count_bits(candidates) <= best_size: return
            
            # Let's try a slightly better bound:
            # Pick a pivot? No, let's just do the coloring if the simple bound doesn't prune.
            if len(clique) + count_bits(candidates) <= best_size:
                return
            
            # Greedy coloring to get a tighter bound
            # This might be slow for large n, but n=64 is manageable if done carefully.
            # Actually, for n=64, doing full coloring at every node of the search tree is expensive.
            # Let's use a simpler bound: count_bits(candidates) is O(n).
            # Let's stick to the simple bound to ensure speed, or a very cheap heuristic.
            
            # Heuristic: pick vertex with max degree in candidates
            best_v = -1
            best_deg = -1
            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                deg = count_bits(adj[v] & candidates)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                c ^= lsb
            
            if best_v == -1:
                return
            
            # Include best_v
            new_clique = clique + [best_v]
            new_candidates = candidates & adj[best_v]
            expand(new_clique, new_candidates)
            
            # Exclude best_v
            candidates &= ~(1 << best_v)
            expand(clique, candidates)

        expand([], (1 << n) - 1)
        return best_clique

    def partition(adj, n, deadline):
        # Memoization key: tuple of sorted adjacency masks
        # Since adj is a list of ints, we can use tuple(adj) as key.
        # However, the graph is symmetric, so adj[i] has bit j iff adj[j] has bit i.
        # We must ensure we only use valid states.
        
        # To avoid memory explosion, we limit the cache size.
        # If the cache gets too big, we clear it or disable it.
        
        cache = {}
        
        def solve(rem_adj_tuple):
            # Check if no edges left
            has_edges = False
            for mask in rem_adj_tuple:
                if mask:
                    has_edges = True
                    break
            if not has_edges:
                return 0, []
            
            if rem_adj_tuple in cache:
                return cache[rem_adj_tuple]
            
            if time.time() > deadline:
                # Fallback: greedy
                # Convert back to list
                rem_adj = list(rem_adj_tuple)
                cliques = []
                while True:
                    # Find max degree vertex
                    max_deg = 0
                    start_v = -1
                    for u in range(n):
                        deg = count_bits(rem_adj[u])
                        if deg > max_deg:
                            max_deg = deg
                            start_v = u
                    if max_deg == 0:
                        break
                    clique = [start_v]
                    candidates = rem_adj[start_v]
                    while candidates:
                        best_cand = -1
                        best_deg = -1
                        c = candidates
                        while c:
                            lsb = c & -c
                            v = lsb.bit_length() - 1
                            deg = count_bits(rem_adj[v])
                            if deg > best_deg:
                                best_deg = deg
                                best_cand = v
                            c ^= lsb
                        if best_cand == -1:
                            break
                        clique.append(best_cand)
                        candidates &= rem_adj[best_cand]
                    
                    for u in clique:
                        for v in clique:
                            if u < v:
                                rem_adj[u] &= ~(1 << v)
                                rem_adj[v] &= ~(1 << u)
                    cliques.append(clique)
                    
                    # Final safety: single edges
                    for u in range(n):
                        for v in range(u+1, n):
                            if rem_adj[u] & (1 << v):
                                cliques.append([u, v])
                                rem_adj[u] &= ~(1 << v)
                                rem_adj[v] &= ~(1 << u)
                    return len(cliques), cliques

            # Find a maximum clique in the remaining graph
            # We need to convert rem_adj_tuple to list for max_clique
            rem_adj_list = list(rem_adj_tuple)
            max_c = max_clique(rem_adj_list, n, deadline)
            
            if not max_c:
                # Should not happen if has_edges is True
                return 1, [[0, 1]] # Dummy, will be overridden by fallback

            # Branch 1: Take this clique
            new_adj_list = rem_adj_list[:]
            for u in max_c:
                for v in max_c:
                    if u < v:
                        new_adj_list[u] &= ~(1 << v)
                        new_adj_list[v] &= ~(1 << u)
            
            count1, cliques1 = solve(tuple(new_adj_list))
            total1 = 1 + count1
            result1 = [max_c] + cliques1
            
            # Branch 2: Exclude this clique?
            # Excluding a clique means we don't use it as a whole.
            # But we must partition the edges.
            # The "exclude" branch in clique partitioning is tricky.
            # Usually, we branch on an edge (u,v).
            # Either (u,v) is in a clique with other vertices, or (u,v) is a clique by itself (or part of a smaller clique not containing the max clique's other vertices).
            
            # A better branching strategy:
            # Pick an edge (u,v) from the max clique? Or just any edge?
            # Let's pick the first edge in the max clique: (max_c[0], max_c[1]).
            # Option A: Include this edge in a clique that is a subset of the max clique?
            # This is getting complicated.
            
            # Simpler Recursive Strategy:
            # 1. Find Max Clique C.
            # 2. Option A: Use C. Remove edges of C. Solve remaining.
            # 3. Option B: Do NOT use C. This implies that at least one edge in C is not covered by C.
            #    Actually, if we don't use C, we must cover the edges of C using other cliques.
            #    Since C is a clique, its edges can be covered by smaller cliques.
            #    To force this, we can remove one edge (u,v) from C from the "available" set for forming C?
            #    No, the graph is fixed. We are choosing a set of cliques.
            #    
            #    Alternative: Branch on an edge (u,v) in C.
            #    Case 1: (u,v) is in a clique that is a subset of C (but not C itself? or maybe C is not used).
            #    This is essentially the same as "Use C" vs "Don't use C".
            #    
            #    If we "Don't use C", we must partition the edges of C using cliques strictly smaller than C?
            #    Not necessarily. We could use a clique D that is not a subset of C but shares edges?
            #    No, if D shares an edge with C, and D is a clique, D might intersect C.
            #    
            #    Let's stick to a simpler heuristic that is still recursive:
            #    1. Find Max Clique C.
            #    2. Try using C.
            #    3. Try using C \ {v} for each v in C? (i.e., force a smaller clique).
            #       This is too many branches.
            
            # Let's try a different strategy:
            # Greedy with Local Search / Randomized Restarts.
            # The current program already does greedy.
            # The inspiration says "randomized restarts explore different partition orderings".
            # Let's implement a fast greedy with multiple random restarts and keep the best.
            
            # This is a DIFFERENT strategy from the pure recursive B&B which is likely to timeout or blow up memory.
            # Strategy: Parallel/Sequential Randomized Greedy.
            
            best_count = float('inf')
            best_cliques = None
            
            # Number of restarts depends on time.
            # n=32: fast. n=64: slower.
            
            return total1, result1 # We will abandon the recursive path and use this in the main loop?
            # No, the function `solve` is recursive. If I return here, I don't explore branch 2.
            # Given the complexity, I will switch the main logic to Randomized Greedy.

    # Main Logic: Randomized Greedy with Restarts
    
    def greedy_partition(adj, n, seed, deadline):
        rng = random.Random(seed)
        rem_adj = adj[:]
        cliques = []
        
        while True:
            if time.time() > deadline:
                break
            
            # Find a vertex with max degree
            max_deg = 0
            start_v = -1
            for u in range(n):
                deg = count_bits(rem_adj[u])
                if deg > max_deg:
                    max_deg = deg
                    start_v = u
            if max_deg == 0:
                break
            
            # Build a clique greedily from start_v
            # Randomize the order of adding vertices to explore different partitions
            clique = [start_v]
            candidates = rem_adj[start_v]
            
            # Shuffle candidates?
            # To be efficient, we can just pick the next vertex with max degree in candidates.
            # To randomize, we can pick among the top K degrees.
            
            while candidates:
                # Pick best candidate
                best_cand = -1
                best_deg = -1
                c = candidates
                # Collect top candidates
                cands_list = []
                while c:
                    lsb = c & -c
                    v = lsb.bit_length() - 1
                    deg = count_bits(rem_adj[v])
                    cands_list.append((deg, v))
                    c ^= lsb
                
                if not cands_list:
                    break
                
                # Sort by degree
                cands_list.sort(reverse=True)
                
                # Pick from top 3 randomly to add diversity
                k = min(3, len(cands_list))
                chosen = cands_list[rng.randint(0, k-1)][1]
                
                clique.append(chosen)
                candidates &= rem_adj[chosen]
                
                # Safety: if clique grows too large or time is up, stop
                if len(clique) > 10: 
                    break

            # Remove edges of this clique
            for u in clique:
                for v in clique:
                    if u < v:
                        rem_adj[u] &= ~(1 << v)
                        rem_adj[v] &= ~(1 << u)
            
            cliques.append(clique)
            
        # Fill remaining with single edges
        for u in range(n):
            for v in range(u+1, n):
                if rem_adj[u] & (1 << v):
                    cliques.append([u, v])
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)
        
        return len(cliques), cliques

    start_time = time.time()
    deadline = start_time + 140
    
    adj_1 = instance(32, 0.6, 601)
    adj_2 = instance(64, 0.55, 602)
    
    # Allocate time
    # n=32 is much faster. Give it 20s.
    # n=64 gets the rest.
    
    best_1_count = float('inf')
    best_1_cliques = []
    
    # Run multiple restarts for instance 1
    deadline_1 = start_time + 20
    seed = 1
    while time.time() < deadline_1:
        count, cliques = greedy_partition(adj_1, 32, seed, deadline_1)
        if count < best_1_count:
            best_1_count = count
            best_1_cliques = cliques
        seed += 1
        
    # If we didn't find any (shouldn't happen), fallback
    if not best_1_cliques:
        _, best_1_cliques = greedy_partition(adj_1, 32, 1, deadline_1)
        
    # Run multiple restarts for instance 2
    deadline_2 = start_time + 145
    best_2_count = float('inf')
    best_2_cliques = []
    
    seed = 1
    while time.time() < deadline_2:
        count, cliques = greedy_partition(adj_2, 64, seed, deadline_2)
        if count < best_2_count:
            best_2_count = count
            best_2_cliques = cliques
        seed += 1
        
    if not best_2_cliques:
        _, best_2_cliques = greedy_partition(adj_2, 64, 1, deadline_2)
        
    return [best_1_cliques, best_2_cliques]


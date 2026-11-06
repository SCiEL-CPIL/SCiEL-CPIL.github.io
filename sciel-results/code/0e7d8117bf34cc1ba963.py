<<<<<<< SEARCH
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Helper: Check if graph is C4-free using bitmasks
    def is_c4_free(adj_mask, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_mask[u] & adj_mask[v]
                if common & (common - 1):
                    return False
        return True

    # Helper: Check if there is an independent set of size >= target
    def has_independent_set(adj_mask, n, target=11):
        sys.setrecursionlimit(10000)
        
        def dfs(candidates, size):
            if size >= target:
                return True
            
            if candidates == 0:
                return False
            
            # Pruning: if even taking all candidates doesn't reach target
            if size + candidates.bit_count() < target:
                return False
                
            # Pick a vertex from candidates with highest degree in induced subgraph
            c = candidates
            best_v = -1
            max_deg = -1
            
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                
                deg = (adj_mask[v] & candidates).bit_count()
                
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
                
            # Branch 1: Include best_v
            new_candidates = candidates & ~(adj_mask[best_v] | (1 << best_v))
            if dfs(new_candidates, size + 1):
                return True
                
            # Branch 2: Exclude best_v
            new_candidates = candidates & ~(1 << best_v)
            if dfs(new_candidates, size):
                return True
                
            return False

        all_candidates = (1 << n) - 1
        return dfs(all_candidates, 0)
=======
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Helper: Check if graph is C4-free using bitmasks
    def is_c4_free(adj_mask, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_mask[u] & adj_mask[v]
                if common & (common - 1):
                    return False
        return True

    # Helper: Check if there is an independent set of size >= target
    # Uses branch-and-bound with greedy coloring upper bound for pruning
    def has_independent_set(adj_mask, n, target=11):
        sys.setrecursionlimit(10000)
        memo = {}
        
        def greedy_color_bound(candidates):
            """Upper bound on independence number via greedy coloring of complement."""
            remaining = candidates
            colors = 0
            while remaining:
                colors += 1
                # Greedily pick a maximal independent set in the complement (i.e., clique in original)
                # Actually for upper bound on alpha, we color the graph induced by candidates
                # Each color class is an independent set, so number of colors >= alpha
                # We want a tight upper bound, so use greedy coloring
                color_class = 0
                uncolored = remaining
                while uncolored:
                    # Pick vertex with fewest uncolored neighbors (MRV heuristic for coloring)
                    best_v = -1
                    min_neighbors = n + 1
                    c = uncolored
                    while c:
                        lsb = c & (-c)
                        v = lsb.bit_length() - 1
                        c ^= lsb
                        nb_in_uncolored = (adj_mask[v] & uncolored).bit_count()
                        if nb_in_uncolored < min_neighbors:
                            min_neighbors = nb_in_uncolored
                            best_v = v
                    if best_v == -1:
                        break
                    color_class |= (1 << best_v)
                    uncolored &= ~(adj_mask[best_v] | (1 << best_v))
                remaining &= ~color_class
            return colors
        
        def dfs(candidates, size):
            if size >= target:
                return True
            if candidates == 0:
                return False
            if size + candidates.bit_count() < target:
                return False
            
            # Memoization
            key = (candidates, size)
            if key in memo:
                return memo[key]
            
            # Upper bound via greedy coloring
            ub = greedy_color_bound(candidates)
            if size + ub < target:
                memo[key] = False
                return False
            
            # Pick vertex with highest degree in induced subgraph
            c = candidates
            best_v = -1
            max_deg = -1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                memo[key] = False
                return False
            
            # Branch 1: Include best_v
            new_candidates = candidates & ~(adj_mask[best_v] | (1 << best_v))
            if dfs(new_candidates, size + 1):
                memo[key] = True
                return True
            
            # Branch 2: Exclude best_v
            new_candidates = candidates & ~(1 << best_v)
            if dfs(new_candidates, size):
                memo[key] = True
                return True
            
            memo[key] = False
            return False

        all_candidates = (1 << n) - 1
        return dfs(all_candidates, 0)
>>>>>>> REPLACE


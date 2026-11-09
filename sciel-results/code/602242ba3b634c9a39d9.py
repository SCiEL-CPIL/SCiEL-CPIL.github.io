import time
import random

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Seed graph: Andrásfai And(9) on n=26
    best_n = 26
    best_edges = [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    
    # Try larger n
    # R(3,10) <= 41, so max n is 40.
    for n in [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29]:
        if time.time() > deadline:
            break
        
        # Allocate time based on n
        # n=40: 60s, n=39: 40s, others: 5s
        if n == 40:
            time_limit = 60.0
        elif n == 39:
            time_limit = 40.0
        else:
            time_limit = 5.0
            
        result = search_for_graph(n, time_limit)
        if result:
            best_n = n
            best_edges = result
            if n == 40:
                break # Can't do better than 40
        
    return {'n': best_n, 'edges': best_edges}

def search_for_graph(n, time_limit):
    start = time.time()
    
    # Helper: Check if graph is valid (alpha < 10)
    def is_valid(neighbors):
        # Build complement adjacency
        comp_adj = [0] * n
        full_mask = (1 << n) - 1
        for i in range(n):
            # Complement of neighbors[i] excluding i
            comp_adj[i] = full_mask ^ neighbors[i] ^ (1 << i)
        
        # B&B to check for clique of size 10 in comp_adj
        # Use a simple recursive search with pruning
        # To speed up, sort vertices by degree in comp_adj
        degrees = [bin(comp_adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        # Remap vertices? Or just use the order in recursion?
        # It's easier to just use the original indices but pick vertices in the order of 'order'
        
        def expand(clique_size, candidates_mask):
            if clique_size == 10:
                return True
            if candidates_mask == 0:
                return False
            if clique_size + bin(candidates_mask).count('1') < 10:
                return False
            
            # Pick the vertex with highest degree from candidates
            # This requires iterating through candidates
            # For n=40, this is fast
            
            # Optimization: Pre-compute best vertex?
            # Just pick one.
            # To improve pruning, pick the vertex with max degree in the subgraph induced by candidates
            
            best_v = -1
            best_deg = -1
            temp_mask = candidates_mask
            while temp_mask:
                v_bit = temp_mask & -temp_mask
                v = v_bit.bit_length() - 1
                temp_mask ^= v_bit
                # Degree of v in subgraph induced by candidates
                deg = bin(comp_adj[v] & candidates_mask).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
            
            # Branch 1: Include best_v
            new_candidates = candidates_mask & comp_adj[best_v]
            if expand(clique_size + 1, new_candidates):
                return True
            
            # Branch 2: Exclude best_v
            new_candidates = candidates_mask ^ (1 << best_v)
            if expand(clique_size, new_candidates):
                return True
            
            return False
        
        initial_mask = (1 << n) - 1
        return not expand(0, initial_mask)

    def find_large_independent_set(neighbors):
        # Greedy: pick vertex with min degree in remaining graph
        remaining = (1 << n) - 1
        indep = []
        while remaining:
            # Find vertex with min degree in subgraph induced by remaining
            best_v = -1
            best_deg = n + 1
            temp_mask = remaining
            while temp_mask:
                v_bit = temp_mask & -temp_mask
                v = v_bit.bit_length() - 1
                temp_mask ^= v_bit
                deg = bin(neighbors[v] & remaining).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            indep.append(best_v)
            remaining ^= (1 << best_v)
            # Remove neighbors of best_v from remaining
            remaining &= ~neighbors[best_v]
        return indep

    # Main search loop
    while time.time() - start < time_limit:
        # 1. Generate random maximal triangle-free graph
        neighbors = [0] * n
        edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if (neighbors[u] & neighbors[v]) == 0:
                neighbors[u] |= 1 << v
                neighbors[v] |= 1 << u
        
        # 2. Repair loop
        for _ in range(50):
            indep_set = find_large_independent_set(neighbors)
            if len(indep_set) < 10:
                # Candidate
                if is_valid(neighbors):
                    return [[i, j] for i in range(n) for j in range(i+1, n) if (neighbors[i] >> j) & 1]
                else:
                    # B&B found a set, greedy didn't.
                    # Try to repair using the set found by B&B?
                    # We don't have the set, just a boolean.
                    # Let's just break and restart.
                    break
            
            # Repair: add edges within indep_set
            improved = False
            pairs = [(u, v) for i, u in enumerate(indep_set) for v in indep_set[i+1:]]
            random.shuffle(pairs)
            for u, v in pairs:
                if (neighbors[u] & neighbors[v]) == 0:
                    neighbors[u] |= 1 << v
                    neighbors[v] |= 1 << u
                    improved = True
            if not improved:
                break
    
    return None

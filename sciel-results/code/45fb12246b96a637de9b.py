import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11
    
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)
    
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
        
    adj = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in range(i + 1, N_S):
            if (masks[i] & masks[j]).bit_count() == 3:
                adj[i][j] = True
                adj[j][i] = True
                
    # Complement adjacency for coloring bound
    # comp_adj[i][j] is True if i and j are NOT adjacent in original (and i!=j)
    comp_adj = ~adj
    np.fill_diagonal(comp_adj, False)
    
    best_size = 0
    best_set = set()
    
    def greedy_color_bound(candidates):
        """
        Computes an upper bound on the size of the independent set in the subgraph induced by candidates.
        Uses greedy coloring of the complement graph.
        """
        if not candidates:
            return 0
            
        pool = set(candidates)
        colors = 0
        
        while pool:
            colors += 1
            # Pick a vertex, e.g., the one with max degree in original graph within pool
            # Or just arbitrary for speed? Max degree in original (min in comp) is good.
            # Let's pick arbitrary for now, or min index.
            v = next(iter(pool))
            
            # Find all vertices in pool that are NOT adjacent to v in original graph
            # These are neighbors of v in complement graph.
            # They must have different colors, so they are removed from this color class.
            # Wait. In greedy coloring of G^c:
            # We pick v. We assign color c.
            # We remove v and all neighbors of v in G^c from the pool for this color.
            # Neighbors of v in G^c are u such that comp_adj[v][u] is True.
            
            to_remove = {v}
            # Vectorized check
            v_idx = v
            # Get neighbors in comp_adj
            neighbors_in_comp = np.where(comp_adj[v_idx])[0]
            # Filter to those in pool
            for u in neighbors_in_comp:
                if u in pool:
                    to_remove.add(u)
            
            pool -= to_remove
            
        return colors

    def branch_and_bound(current_set, candidates):
        nonlocal best_size, best_set
        
        if time.time() > deadline - 5.0:
            return
            
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
            
        # Upper bound
        ub = greedy_color_bound(candidates)
        if len(current_set) + ub <= best_size:
            return
            
        # Select vertex
        # Heuristic: Max degree in original graph within candidates
        max_deg = -1
        v = None
        for u in candidates:
            deg = sum(1 for w in candidates if adj[u][w])
            if deg > max_deg:
                max_deg = deg
                v = u
                
        # Branch 1: Include v
        new_current = list(current_set) + [v]
        # Remove v and its neighbors in original graph from candidates
        new_candidates = set(candidates)
        new_candidates.discard(v)
        for u in candidates:
            if adj[v][u]:
                new_candidates.discard(u)
                
        branch_and_bound(new_current, new_candidates)
        
        # Branch 2: Exclude v
        new_candidates = set(candidates)
        new_candidates.discard(v)
        
        branch_and_bound(current_set, new_candidates)

    # Initial candidates
    initial_candidates = set(range(N_S))
    
    # Run BB
    branch_and_bound([], initial_candidates)
    
    # Construct vectors
    vectors = []
    for s_idx in best_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    return final_vectors

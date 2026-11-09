import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build adjacency bitmasks for G
    adj = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j1 = (i + d) % m
            j2 = (i - d) % m
            mask |= (1 << j1)
            mask |= (1 << j2)
        adj[i] = mask
        
    # Build adjacency bitmasks for complement graph G_bar
    # In G_bar, i and j are connected if they are NOT connected in G (and i!=j)
    full_mask = (1 << m) - 1
    adj_bar = [0] * m
    for i in range(m):
        # Neighbors in G_bar are all vertices except i and neighbors in G
        adj_bar[i] = full_mask & ~(1 << i) & ~adj[i]

    best_set = [0]
    best_size = 0
    start_time = time.time()
    
    def get_bound(candidates_mask):
        # Greedy coloring of G_bar induced by candidates
        # This gives an upper bound on the clique size in G_bar, 
        # which is the independence number in G.
        if candidates_mask == 0:
            return 0
            
        colors = 0
        remaining = candidates_mask
        color_mask = 0
        
        # Simple greedy: 
        # 1. Find a maximal independent set in G_bar (which is a clique in G? No.)
        # Wait. We want to color G_bar.
        # A greedy coloring algorithm:
        # While remaining != 0:
        #   Find a maximal clique in G_bar[remaining]? No, that's expensive.
        #   Find a maximal independent set in G_bar[remaining]? 
        #   An independent set in G_bar is a clique in G.
        #   This is also expensive.
        
        # Simpler greedy coloring heuristic:
        # Order vertices. For each vertex, assign the smallest color not used by neighbors.
        # This is O(V^2). V <= 205. Very fast.
        
        # Let's extract list of vertices
        verts = []
        temp = candidates_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            verts.append(idx)
            temp ^= lsb
            
        # Heuristic ordering: sort by degree in G_bar[verts] descending
        # Degree of v in G_bar[verts] is popcount(adj_bar[v] & candidates_mask)
        verts.sort(key=lambda v: (adj_bar[v] & candidates_mask).bit_count(), reverse=True)
        
        color_of = {}
        num_colors = 0
        for v in verts:
            # Find colors used by neighbors of v in G_bar
            neighbor_colors = set()
            neigh_mask = adj_bar[v] & candidates_mask
            # Iterate neighbors
            temp_n = neigh_mask
            while temp_n:
                lsb_n = temp_n & -temp_n
                idx_n = lsb_n.bit_length() - 1
                if idx_n in color_of:
                    neighbor_colors.add(color_of[idx_n])
                temp_n ^= lsb_n
                
            # Assign smallest available color
            c = 0
            while c in neighbor_colors:
                c += 1
            color_of[v] = c
            if c + 1 > num_colors:
                num_colors = c + 1
                
        return num_colors

    def solve(candidates_mask, current_mask, current_size):
        nonlocal best_set, best_size
        nonlocal start_time
        
        # Time check
        if time.time() - start_time > 500: # 500s limit, 100s buffer
            return

        # Upper bound
        bound = get_bound(candidates_mask)
        if current_size + bound <= best_size:
            return
        
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                s = []
                temp = current_mask
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    s.append(idx)
                    temp ^= lsb
                best_set = s
            return

        # Pick vertex v
        # Heuristic: pick vertex with max degree in G_bar (min degree in G) among candidates?
        # Or max degree in G?
        # Let's pick the vertex with max degree in G_bar to maximize pruning?
        # Actually, for Independent Set, picking a high-degree vertex in G is good 
        # because it removes many candidates.
        # Degree in G is popcount(adj[v] & candidates_mask).
        
        verts = []
        temp = candidates_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            verts.append(idx)
            temp ^= lsb
            
        # Sort by degree in G descending
        verts.sort(key=lambda v: (adj[v] & candidates_mask).bit_count(), reverse=True)
        v = verts[0]
        
        # Branch 1: Include v
        new_candidates_1 = candidates_mask & ~adj[v] & ~(1 << v)
        solve(new_candidates_1, current_mask | (1 << v), current_size + 1)
        
        # Branch 2: Exclude v
        new_candidates_2 = candidates_mask & ~(1 << v)
        solve(new_candidates_2, current_mask, current_size)

    initial_candidates = (1 << m) - 1
    solve(initial_candidates, 0, 0)
    
    return np.array(best_set, dtype=np.int64)

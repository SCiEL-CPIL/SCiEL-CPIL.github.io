import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 580
    
    # 1. Compute Forbidden Differences F
    forbidden = set()
    for x in range(1, m):
        sq = pow(x, 2, m)
        if sq != 0:
            forbidden.add(sq)
    
    # 2. Build Neighbor Masks
    # neighbors[i] is a bitmask where bit j is set if i and j are adjacent (conflict)
    neighbors = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j1 = (i + d) % m
            j2 = (i - d) % m
            mask |= (1 << j1)
            mask |= (1 << j2)
        neighbors[i] = mask

    # 3. Branch and Bound
    best_size = 0
    best_set_mask = 0
    best_set_list = []
    
    # Global time check
    def time_check():
        return time.time() > deadline - 5

    # Greedy Coloring Bound
    # Returns an upper bound on the size of an independent set in the subgraph induced by 'available'
    def color_bound(available):
        if available == 0:
            return 0
        
        # Heuristic: Order vertices by degree in the subgraph 'available'
        # To be fast, we just pick vertices greedily.
        # A better bound is obtained by picking the vertex with the highest degree in the remaining graph.
        
        colors = 0
        remaining = available
        
        while remaining:
            # Find a maximal independent set in 'remaining'
            # This is one color class
            independent = 0
            candidates = remaining
            
            # To make it faster and tighter, we can sort candidates by degree
            # But for N=205, simple iteration is often sufficient if we pick "good" vertices.
            # Let's pick the vertex with the highest degree in 'candidates'.
            
            # Optimization: Precompute degrees for the current 'candidates'?
            # It might be overhead. Let's try a simple greedy: pick first available?
            # No, picking high degree vertices first usually leads to fewer colors (tighter bound).
            
            # Calculate degrees in 'candidates'
            # deg(v) = popcount(neighbors[v] & candidates)
            
            # To avoid O(N^2) overhead in every step of coloring, we can just iterate.
            # N=205 is small.
            
            while candidates:
                # Find vertex with max degree in candidates
                max_deg = -1
                best_v = -1
                # Iterate through bits in candidates
                # Using bit manipulation to iterate set bits
                c = candidates
                while c:
                    # Get lowest set bit
                    lsb = c & -c
                    v = lsb.bit_length() - 1
                    deg = (neighbors[v] & candidates).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        best_v = v
                    c ^= lsb
                
                if best_v == -1:
                    break
                    
                # Add best_v to independent set
                independent |= (1 << best_v)
                
                # Remove best_v and its neighbors from candidates
                # candidates = candidates & ~(1 << best_v) & ~neighbors[best_v]
                # Note: neighbors[best_v] might contain vertices not in candidates, but & candidates handles that.
                candidates &= ~(1 << best_v)
                candidates &= ~neighbors[best_v]
                
            # This 'independent' is one color class
            remaining &= ~independent
            colors += 1
            
        return colors

    def dfs(available, current_mask, current_size):
        nonlocal best_size, best_set_mask, best_set_list
        
        if time_check():
            return
            
        if available == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
                best_set_list = [i for i in range(m) if (current_mask >> i) & 1]
            return
            
        # Pruning: Upper bound
        # current_size + color_bound(available) <= best_size
        bound = color_bound(available)
        if current_size + bound <= best_size:
            return
            
        # Heuristic: Pick a vertex to branch on
        # Strategy: Pick vertex with highest degree in 'available' to reduce branching factor quickly?
        # Or pick vertex with lowest degree?
        # For MIS, branching on high degree vertices is often good because including them removes many options.
        
        # Find vertex with max degree in available
        max_deg = -1
        branch_v = -1
        a = available
        while a:
            lsb = a & -a
            v = lsb.bit_length() - 1
            deg = (neighbors[v] & available).bit_count()
            if deg > max_deg:
                max_deg = deg
                branch_v = v
            a ^= lsb
            
        if branch_v == -1:
            return
            
        # Branch 1: Include branch_v
        # New available = available \ {branch_v} \ neighbors[branch_v]
        new_available_1 = available & ~(1 << branch_v) & ~neighbors[branch_v]
        new_mask_1 = current_mask | (1 << branch_v)
        
        dfs(new_available_1, new_mask_1, current_size + 1)
        
        # Branch 2: Exclude branch_v
        new_available_2 = available & ~(1 << branch_v)
        
        dfs(new_available_2, current_mask, current_size)

    # Initial available: all vertices
    initial_available = (1 << m) - 1
    
    # To improve initial lower bound, run a greedy MIS first
    # Greedy MIS: Pick vertex with max degree, add to set, remove it and neighbors.
    greedy_avail = initial_available
    greedy_set_mask = 0
    greedy_size = 0
    while greedy_avail:
        # Pick max degree
        max_deg = -1
        v = -1
        a = greedy_avail
        while a:
            lsb = a & -a
            vi = lsb.bit_length() - 1
            deg = (neighbors[vi] & greedy_avail).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = vi
            a ^= lsb
        
        if v != -1:
            greedy_set_mask |= (1 << v)
            greedy_size += 1
            greedy_avail &= ~(1 << v)
            greedy_avail &= ~neighbors[v]
            
    best_size = greedy_size
    best_set_mask = greedy_set_mask
    best_set_list = [i for i in range(m) if (best_set_mask >> i) & 1]

    # Run B&B
    dfs(initial_available, 0, 0)

    return np.array(best_set_list, dtype=np.int64)

import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    n = 256
    # Vertices are integers 1..256 (index 0..255)
    # Weight of vertex i is (i+1)
    weights = list(range(1, n + 1))
    
    # Build the conflict graph: edge between i and j if gcd(i+1, j+1) > 1
    # Use bitsets for fast operations
    # conflict_mask[i] is a bitmask of all j != i where gcd(i+1, j+1) > 1
    conflict_mask = [0] * n
    for i in range(n):
        val_i = i + 1
        mask = 0
        for j in range(n):
            if i == j:
                continue
            if gcd(val_i, j + 1) > 1:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # For branch and bound, we need to find max weight independent set
    # Upper bound: for a set of candidate vertices S, the max weight independent set
    # is at most the sum of weights in S (trivial bound)
    # Better bound: greedy coloring bound on the subgraph induced by S
    
    # Precompute: for each vertex, its neighbors as a set for quick checking
    # Actually, let's use the bitset approach directly
    
    # Sort vertices by weight descending for better pruning
    # We'll work with the original indices but process in weight order
    
    # Branch and bound for MWIS
    # State: current set (as bitmask), current weight, available candidates (as bitmask)
    # Upper bound: use a greedy coloring of the available subgraph
    #   For each color class, at most one vertex can be chosen
    #   Sum of max weight in each color class gives an upper bound
    
    # Precompute prime factors for faster gcd checks (though we already have the masks)
    
    # For the coloring bound, we need to color the subgraph induced by available vertices
    # A simple greedy coloring: process vertices in some order, assign smallest available color
    
    def greedy_color_bound(available_mask):
        """
        Given a set of available vertices (bitmask), compute an upper bound
        on the max weight independent set using greedy coloring.
        Returns the sum of max weights per color class.
        """
        if available_mask == 0:
            return 0
        
        # Extract vertices from mask
        # We'll do greedy coloring: for each vertex in order of decreasing weight,
        # assign the smallest color not used by its neighbors that are already colored
        
        # Get list of vertices in available_mask, sorted by weight descending
        verts = []
        m = available_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        
        # Sort by weight descending
        verts.sort(key=lambda v: weights[v], reverse=True)
        
        # Greedy coloring
        color_of = {}  # vertex -> color
        color_max_weight = []  # color_max_weight[c] = max weight in color class c
        
        for v in verts:
            # Find neighbors of v that are in available_mask and already colored
            neighbors_in_avail = conflict_mask[v] & available_mask
            used_colors = set()
            nm = neighbors_in_avail
            while nm:
                lsb = nm & (-nm)
                nidx = lsb.bit_length() - 1
                if nidx in color_of:
                    used_colors.add(color_of[nidx])
                nm ^= lsb
            
            # Assign smallest available color
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            if c >= len(color_max_weight):
                color_max_weight.append(weights[v])
            else:
                color_max_weight[c] = max(color_max_weight[c], weights[v])
        
        return sum(color_max_weight)
    
    # Branch and bound
    best_weight = 0
    best_set = set()
    
    def branch_and_bound(available_mask, current_mask, current_weight):
        nonlocal best_weight, best_set
        
        if time.time() - start_time > DEADLINE:
            return
        
        if available_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set()
                cm = current_mask
                while cm:
                    lsb = cm & (-cm)
                    idx = lsb.bit_length() - 1
                    best_set.add(idx + 1)
                    cm ^= lsb
            return
        
        # Upper bound
        ub = current_weight + greedy_color_bound(available_mask)
        if ub <= best_weight:
            return
        
        # Select a vertex to branch on
        # Heuristic: pick the vertex with highest weight in available
        m = available_mask
        best_v = -1
        best_w = -1
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            if weights[idx] > best_w:
                best_w = weights[idx]
                best_v = idx
            m ^= lsb
        
        v = best_v
        
        # Branch 1: include v
        new_available = available_mask & ~(1 << v) & ~conflict_mask[v]
        new_current = current_mask | (1 << v)
        branch_and_bound(new_available, new_current, current_weight + weights[v])
        
        # Branch 2: exclude v
        new_available2 = available_mask & ~(1 << v)
        branch_and_bound(new_available2, current_mask, current_weight)
    
    # Initial available mask: all vertices
    initial_available = (1 << n) - 1
    branch_and_bound(initial_available, 0, 0)
    
    if not best_set:
        # Fallback: just return [1] or a simple valid set
        return [1]
    
    result = list(best_set)
    
    # Verify pairwise coprimality
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Should not happen, but just in case
                # Remove the smaller one
                if result[i] < result[j]:
                    result.pop(i)
                else:
                    result.pop(j)
                break
    
    return result

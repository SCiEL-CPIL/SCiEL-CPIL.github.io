"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import sys

def construct():
    """
    Find a maximum cardinality subset of [1, 256] such that a+b is never a perfect square
    for any a <= b in the set (including a=b).
    
    We model this as a maximum independent set problem on the conflict graph.
    Uses an optimized Branch-and-Bound with bitsets and greedy coloring bounds.
    """
    N = 256
    
    # Precompute perfect squares up to 2*256 = 512
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    
    # Filter out self-conflicting vertices (2*x is a square)
    valid_numbers = []
    for x in range(1, N + 1):
        if 2 * x not in squares:
            valid_numbers.append(x)
    
    m = len(valid_numbers)
    if m == 0:
        return [1] # Fallback, though 1 is self-valid (2 is not a square)
    
    # Map original number to index
    # We work with indices 0..m-1
    
    # Build adjacency bitmasks
    # adj[i] contains bits for all j != i such that valid_numbers[i] + valid_numbers[j] is a square
    adj = [0] * m
    for i in range(m):
        a = valid_numbers[i]
        mask = 0
        for j in range(m):
            if i == j:
                continue
            b = valid_numbers[j]
            if a + b in squares:
                mask |= (1 << j)
        adj[i] = mask
    
    full_mask = (1 << m) - 1
    
    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')
    
    # Greedy Coloring Upper Bound
    # This function returns an upper bound on the size of the MIS in the subgraph induced by 'mask'
    # by greedily coloring the vertices. The number of colors is an upper bound on the independence number.
    def color_bound(mask):
        if mask == 0:
            return 0
        
        # Get list of vertices in mask
        verts = []
        mm = mask
        while mm:
            lsb = mm & -mm
            v = lsb.bit_length() - 1
            verts.append(v)
            mm ^= lsb
        
        # Sort by degree in the subgraph (descending) for better coloring
        # Degree of v in mask is popcount(adj[v] & mask)
        verts.sort(key=lambda v: popcount(adj[v] & mask), reverse=True)
        
        colors = {}
        max_color = 0
        for v in verts:
            # Check colors of neighbors that are already colored
            neighbor_colors = set()
            # Iterate through neighbors of v in the mask
            neigh_mask = adj[v] & mask
            while neigh_mask:
                lsb = neigh_mask & -neigh_mask
                u = lsb.bit_length() - 1
                if u in colors:
                    neighbor_colors.add(colors[u])
                neigh_mask ^= lsb
            
            c = 0
            while c in neighbor_colors:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
        
        return max_color + 1 # Number of colors is max_color + 1
    
    # Initial Greedy Solution for Lower Bound
    def greedy_mis(mask):
        sol = []
        remaining = mask
        while remaining:
            # Pick vertex with max degree in remaining
            best_v = -1
            best_deg = -1
            mm = remaining
            while mm:
                lsb = mm & -mm
                v = lsb.bit_length() - 1
                deg = popcount(adj[v] & remaining)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                mm ^= lsb
            
            if best_v == -1:
                break
            
            sol.append(best_v)
            # Remove best_v and its neighbors
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
        
        return sol
    
    initial_sol = greedy_mis(full_mask)
    best_size = len(initial_sol)
    best_set = list(initial_sol)
    
    # Branch and Bound
    # We use a recursive function. To avoid recursion depth issues and overhead,
    # we can implement it iteratively or just ensure it's fast.
    # Given N~240, recursion depth is manageable.
    
    def branch_and_bound(candidates_mask, current_size, current_set):
        nonlocal best_set, best_size
        
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
        
        # Upper bound 1: Trivial count
        if current_size + popcount(candidates_mask) <= best_size:
            return
        
        # Upper bound 2: Coloring bound
        cb = color_bound(candidates_mask)
        if current_size + cb <= best_size:
            return
        
        # Select pivot vertex: Max degree in candidates
        best_v = -1
        best_deg = -1
        mm = candidates_mask
        while mm:
            lsb = mm & -mm
            v = lsb.bit_length() - 1
            deg = popcount(adj[v] & candidates_mask)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            mm ^= lsb
        
        if best_v == -1:
            return
        
        # Branch 1: Include best_v
        # New candidates = candidates \ {best_v} \ neighbors(best_v)
        new_mask_incl = candidates_mask & ~(1 << best_v) & ~adj[best_v]
        current_set.append(best_v)
        branch_and_bound(new_mask_incl, current_size + 1, current_set)
        current_set.pop()
        
        # Branch 2: Exclude best_v
        new_mask_excl = candidates_mask & ~(1 << best_v)
        branch_and_bound(new_mask_excl, current_size, current_set)
    
    branch_and_bound(full_mask, 0, [])
    
    # Convert indices back to values
    result = [valid_numbers[i] for i in best_set]
    
    if not result:
        return [1]
        
    return result

# EVOLVE-BLOCK-END

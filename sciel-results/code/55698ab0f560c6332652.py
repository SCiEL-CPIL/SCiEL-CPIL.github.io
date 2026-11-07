import math

def construct():
    N = 256
    # weights[i] = i+1
    # We want the Maximum Weight Independent Set in the "conflict" graph (gcd > 1)
    # which is equivalent to the Maximum Weight Clique in the "coprime" graph (gcd == 1)
    
    # Build the coprime graph as a list of bitmasks
    # coprime[i] has bit j set if gcd(i+1, j+1) == 1 and i != j
    coprime = [0] * N
    for i in range(N):
        vi = i + 1
        mask = 0
        for j in range(N):
            if i != j:
                vj = j + 1
                if math.gcd(vi, vj) == 1:
                    mask |= (1 << j)
        coprime[i] = mask
    
    # weights
    weights = [i + 1 for i in range(N)]
    
    # Precompute sum of weights for any mask? No, masks are 256-bit, too large for array.
    # We will compute sum on the fly or maintain it in the recursion.
    
    # Greedy Coloring for the bound
    # We need to color the subgraph induced by a set of vertices (mask)
    # The graph is the coprime graph.
    # We want to minimize the number of colors, or rather, compute the sum of max weights per color.
    # Standard Tomita algorithm for Max Clique uses a coloring where we greedily assign colors
    # such that adjacent vertices have different colors.
    # The bound is the sum of the maximum weight in each color class.
    
    # To speed up coloring, we can process vertices in a specific order.
    # A common heuristic is to process vertices with high degree or high weight first.
    # Here, we can just iterate through the bits in the mask.
    
    def get_color_bound(mask):
        """
        Returns the upper bound on the max weight clique in the subgraph induced by 'mask'.
        Uses greedy coloring of the coprime graph.
        """
        if mask == 0:
            return 0
        
        # We will greedily color the vertices in 'mask'.
        # We process vertices one by one.
        # To make the bound tight, we should assign the smallest possible color to each vertex.
        # However, the order matters.
        # Let's try ordering by weight descending (highest index first) to potentially pack heavier weights into lower color indices?
        # Actually, the bound is sum of max weights.
        # If we assign a heavy vertex to color 1, and a light vertex to color 1 (if not adjacent),
        # the contribution of color 1 is max(heavy, light) = heavy.
        # If we assign heavy to color 1 and light to color 2, contribution is heavy + light.
        # So we want to pack as many heavy vertices as possible into the same color if they are not adjacent.
        # This suggests we should process heavy vertices first, so they "claim" a color,
        # and subsequent non-adjacent vertices can share that color.
        
        # Let's extract the vertices from the mask, sorted by weight descending.
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        
        # Sort by weight descending (index descending)
        verts.sort(reverse=True)
        
        # color_assignment: list of colors for each vertex in 'verts'?
        # Or we can just track the max weight per color used so far.
        # Since we process in descending weight, the first vertex in a color class will be the heaviest.
        # So we just need to count how many color classes we have, and sum the weight of the first vertex in each class.
        
        # Wait, if we process v1 (weight 10), v2 (weight 9), v3 (weight 8).
        # v1 gets color 1. Max weight color 1 = 10.
        # v2: if not adjacent to v1, gets color 1. Max weight color 1 is still 10.
        # if adjacent to v1, gets color 2. Max weight color 2 = 9.
        # v3: ...
        
        # So we just need to simulate the coloring.
        # colors[v] = color assigned to vertex v.
        # We can use a dictionary or a list of size N.
        
        # Optimization: We only care about the sum of max weights.
        # Let's maintain a list `color_max_weights`.
        # When we assign a vertex v to color c:
        # if c is new, append weights[v] to color_max_weights.
        # if c exists, do nothing (since weights are processed descending, existing max is >= current weight).
        
        # However, checking adjacency for each color is O(k) where k is number of colors.
        # Number of colors is at most N.
        
        # Let's use a list `used_colors` to track which colors are used.
        # And for each color, we need to know which vertices have that color to check adjacency?
        # No, we can just check adjacency against all vertices already assigned that color.
        
        # Efficient approach:
        # For each vertex v in verts:
        #   Try color 1, 2, ...
        #   Check if v is adjacent to ANY vertex already assigned color c.
        #   If not, assign v to color c.
        
        # To make this fast, we can maintain for each color a bitmask of vertices assigned to it.
        # Then check (coprime[v] & color_mask[c]) == 0.
        
        color_masks = []
        color_max_weights = []
        
        for v in verts:
            assigned = False
            for c in range(len(color_masks)):
                # Check if v is coprime to all vertices in color c
                # In the coprime graph, edges mean coprime.
                # We want v to be adjacent (coprime) to NO vertex in color c?
                # Wait. Coloring requires adjacent vertices to have DIFFERENT colors.
                # So v can take color c only if v is NOT adjacent to any vertex in color c.
                # In our graph, "adjacent" means "coprime".
                # So v can take color c if gcd(v, u) != 1 for all u in color c.
                # i.e., (coprime[v] & color_masks[c]) == 0.
                
                if (coprime[v] & color_masks[c]) == 0:
                    # Assign v to color c
                    color_masks[c] |= (1 << v)
                    # Since we process in descending weight, the first vertex is the max.
                    # No need to update max weight.
                    assigned = True
                    break
            
            if not assigned:
                # New color
                color_masks.append(1 << v)
                color_max_weights.append(weights[v])
                
        return sum(color_max_weights)
    
    best_sum = 0
    best_indices = []
    
    def solve(available_mask, current_sum, current_indices):
        nonlocal best_sum, best_indices
        
        if available_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = current_indices[:]
            return
        
        # Bound check
        # The max weight we can add is at most get_color_bound(available_mask)
        if current_sum + get_color_bound(available_mask) <= best_sum:
            return
            
        # Select vertex to branch on
        # Heuristic: Pick vertex with highest degree in available subgraph?
        # Or highest weight?
        # For Max Clique, picking a vertex with high degree in the candidate set is often good.
        # Let's pick the vertex with the highest weight (highest index) to try to get a good solution early.
        # Or pick the vertex that maximizes the bound reduction?
        # Let's stick to highest weight (highest index) for simplicity and to find heavy solutions early.
        
        # Find highest index in available_mask
        v = available_mask.bit_length() - 1
        
        # Branch 1: Include v
        # If we include v, the remaining candidates must be coprime to v.
        # So new_mask = available_mask & coprime[v] & ~(1 << v)
        new_mask_include = available_mask & coprime[v] & ~(1 << v)
        solve(new_mask_include, current_sum + weights[v], current_indices + [v])
        
        # Branch 2: Exclude v
        # new_mask = available_mask & ~(1 << v)
        new_mask_exclude = available_mask & ~(1 << v)
        solve(new_mask_exclude, current_sum, current_indices)
        
    initial_mask = (1 << N) - 1
    solve(initial_mask, 0, [])
    
    result = [i + 1 for i in best_indices]
    return [int(x) for x in result]

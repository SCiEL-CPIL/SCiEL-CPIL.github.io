import math

def construct():
    N = 256
    # weights[i] = i+1
    # coprime_mask[i] is a bitset of j such that gcd(i+1, j+1) == 1
    
    # Precompute coprime masks
    coprime_mask = [0] * N
    for i in range(N):
        vi = i + 1
        mask = 0
        for j in range(N):
            if i != j:
                vj = j + 1
                if math.gcd(vi, vj) == 1:
                    mask |= (1 << j)
        coprime_mask[i] = mask
    
    # We want Max Weight Clique in graph G_bar (edges = coprime)
    # This is equivalent to Max Weight Independent Set in G (edges = non-coprime)
    
    best_sum = 0
    best_clique = [] # List of indices
    
    # To speed up finding max weight vertex in a bitset,
    # we can precompute for chunks or just iterate.
    # Given N=256, iteration is O(N).
    
    def get_max_weight_vertex(mask):
        # Returns index of vertex with max weight (i+1) in mask
        # Since weight = index+1, max weight is max index.
        # So we just need the highest set bit.
        if mask == 0:
            return -1
        return mask.bit_length() - 1

    def expand(P, current_sum, current_clique):
        nonlocal best_sum, best_clique
        
        if P == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_clique = current_clique[:]
            return
        
        # Upper Bound using Greedy Coloring of P in G_bar
        # We color P such that adjacent vertices in G_bar have different colors.
        # Adjacent in G_bar means coprime.
        # So if u, v are coprime, they must have different colors.
        # This is equivalent to coloring the graph where edges are "coprime".
        
        # We process vertices in P in order of decreasing weight (decreasing index).
        # We maintain a list of color classes (bitsets).
        # For each vertex v, we find the first color c such that v is NOT coprime to any vertex in color_class[c].
        # i.e., for all u in color_class[c], gcd(v, u) > 1.
        # This means v and u are NOT adjacent in G_bar.
        
        # To make this fast, we maintain `color_masks` as a list of ints.
        # `color_masks[c]` is a bitset of vertices assigned color c.
        
        # Optimization: We only need the sum of max weights per color.
        # But we need to assign colors correctly.
        
        # Let's iterate through vertices in P, sorted by index descending.
        # We can extract indices from P.
        
        # Since N=256, we can just iterate 255 down to 0 and check if bit is set.
        
        color_masks = []
        # We will store (max_weight_in_color, mask) for each color
        # Actually, we just need the sum of max weights.
        # But we need to assign colors to check constraints.
        
        # To avoid creating lists repeatedly, we can use a fixed array of size N (max colors).
        # Max colors <= |P| <= 256.
        
        # Let's use a list of masks.
        # For each v in P (desc order):
        #   for c in range(len(color_masks)):
        #      if (coprime_mask[v] & color_masks[c]) == 0:
        #         # v is not coprime to anyone in color c
        #         color_masks[c] |= (1 << v)
        #         break
        #   else:
        #      color_masks.append(1 << v)
        
        # Then UB = sum of max weight for each color.
        # Since we process in desc order, the first vertex added to a color has the max weight for that color.
        
        # Let's implement this bound.
        
        # To get vertices in desc order from bitset P:
        # We can iterate i from 255 down to 0.
        
        # However, this loop is O(N * Colors).
        
        # Let's try to optimize the bound calculation.
        
        # We'll use a local variable for best_sum to check pruning early.
        
        # Calculate UB
        # We process vertices in P in descending order of index (weight).
        
        # To avoid iterating all 256 bits if P is sparse, we can extract set bits.
        # But P is likely dense initially.
        
        # Let's just iterate 255..0.
        
        # To speed up the inner loop, we can precompute coprime_mask.
        
        # We'll use a list `cm` for color masks.
        # And a list `cm_max` for max weights (indices+1) in each color.
        
        # Actually, since we process in desc order, the first vertex assigned to a color is the max.
        
        cm = []
        ub = 0
        
        # Iterate vertices in P in descending order
        # We can use a while loop on P?
        # It's easier to just loop i from 255 to 0.
        
        # To make it faster, we can check if bit is set.
        
        # We can prune the coloring if ub + current_sum <= best_sum?
        # No, we need the full UB to decide.
        # But we can stop adding colors if the potential max is exceeded?
        # The max possible addition is the sum of remaining weights.
        
        # Let's just compute it.
        
        for i in range(255, -1, -1):
            if (P >> i) & 1:
                # Check colors
                assigned = False
                # Try to assign to existing colors
                for c in range(len(cm)):
                    # If v is NOT coprime to any vertex in color c
                    # i.e., (coprime_mask[i] & cm[c]) == 0
                    if (coprime_mask[i] & cm[c]) == 0:
                        cm[c] |= (1 << i)
                        assigned = True
                        break
                if not assigned:
                    cm.append(1 << i)
                    ub += (i + 1) # Max weight for this new color is i+1
        
        if current_sum + ub <= best_sum:
            return
        
        # Branch
        # Pick vertex with highest weight in P
        # Since we want max weight, and weight = index+1, we pick highest index.
        v = P.bit_length() - 1
        
        # Branch 1: Include v
        # New P: P intersect neighbors of v in G_bar
        # Neighbors in G_bar are coprime to v.
        # So new_P = P & coprime_mask[v]
        # Note: coprime_mask[v] does not include v itself.
        new_P = P & coprime_mask[v]
        expand(new_P, current_sum + (v + 1), current_clique + [v])
        
        # Branch 2: Exclude v
        new_P_ex = P & ~(1 << v)
        expand(new_P_ex, current_sum, current_clique)

    initial_P = (1 << N) - 1
    # Remove 0? No, indices are 0..255 corresponding to 1..256.
    # Vertex 0 corresponds to number 1.
    # gcd(1, x) = 1 for all x. So 1 is coprime to all.
    # It will be connected to all in G_bar.
    
    expand(initial_P, 0, [])
    
    # Convert indices to values
    result = [i + 1 for i in best_clique]
    return [int(x) for x in result]

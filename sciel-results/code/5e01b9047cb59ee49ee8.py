# EVOLVE-BLOCK-START
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s margin
    
    best_n = 37
    best_edges = None
    
    # Seed graph for fallback
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    best_edges = seed_edges
    
    def _find_clique_bitmask(n, adj_masks, k):
        # Find clique of size k
        if k == 0:
            return []
        if k == 1:
            return [0]
        if k == 2:
            for i in range(n):
                for j in range(i+1, n):
                    if (adj_masks[i] & (1 << j)):
                        return [i, j]
            return None
        if k == 3:
            for i in range(n):
                mask_i = adj_masks[i]
                mask_j_candidates = mask_i
                while mask_j_candidates:
                    j = (mask_j_candidates & -mask_j_candidates).bit_length() - 1
                    mask_j_candidates &= mask_j_candidates - 1
                    mask_k_candidates = mask_i & adj_masks[j]
                    if mask_k_candidates:
                        k = (mask_k_candidates & -mask_k_candidates).bit_length() - 1
                        return [i, j, k]
            return None
        if k == 4:
            for i in range(n):
                mask_i = adj_masks[i]
                mask_j_candidates = mask_i
                while mask_j_candidates:
                    j = (mask_j_candidates & -mask_j_candidates).bit_length() - 1
                    mask_j_candidates &= mask_j_candidates - 1
                    common_ij = mask_i & adj_masks[j]
                    mask_k_candidates = common_ij
                    while mask_k_candidates:
                        k = (mask_k_candidates & -mask_k_candidates).bit_length() - 1
                        mask_k_candidates &= mask_k_candidates - 1
                        common_ijk = common_ij & adj_masks[k]
                        if common_ijk:
                            l = (common_ijk & -common_ijk).bit_length() - 1
                            return [i, j, k, l]
            return None
        if k == 5:
            for v in range(n):
                mask_v = adj_masks[v]
                mask_u_candidates = mask_v
                while mask_u_candidates:
                    u = (mask_u_candidates & -mask_u_candidates).bit_length() - 1
                    mask_u_candidates &= mask_u_candidates - 1
                    common_vu = mask_v & adj_masks[u]
                    mask_w_candidates = common_vu
                    while mask_w_candidates:
                        w = (mask_w_candidates & -mask_w_candidates).bit_length() - 1
                        mask_w_candidates &= mask_w_candidates - 1
                        common_vuw = common_vu & adj_masks[w]
                        mask_x_candidates = common_vuw
                        while mask_x_candidates:
                            x = (mask_x_candidates & -mask_x_candidates).bit_length() - 1
                            mask_x_candidates &= mask_x_candidates - 1
                            common_vuwx = common_vuw & adj_masks[x]
                            if common_vuwx:
                                y = (common_vuwx & -common_vuwx).bit_length() - 1
                                return [v, u, w, x, y]
            return None
        return None

    def _find_independent_set_bitmask(n, adj_masks, k):
        comp_masks = [0] * n
        full_mask = (1 << n) - 1
        for i in range(n):
            comp_masks[i] = full_mask & ~adj_masks[i] & ~(1 << i)
        return _find_clique_bitmask(n, comp_masks, k)

    for n in range(45, 37, -1):
        if time.time() > deadline:
            break
            
        for restart in range(100):
            if time.time() > deadline:
                break
                
            edges = set()
            
            # Initialize with seed edges for the first 37 vertices
            for u, v in seed_edges:
                if u < n and v < n:
                    edges.add((u, v))
                    
            # Add random edges for the new vertices (37 to n-1)
            # And between new vertices
            for i in range(37, n):
                # Connect to existing vertices (0 to i-1)
                for j in range(i):
                    if random.random() < 0.5:
                        edges.add((j, i))
                # Connect to other new vertices (handled by j < i loop above if we iterate i correctly? 
                # No, the loop above connects i to all j < i. So it covers all pairs.
            
            # Build adjacency masks
            adj_masks = [0] * n
            for u, v in edges:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                
            # Repair loop
            for step in range(1000):
                if time.time() > deadline:
                    break
                    
                k5 = _find_clique_bitmask(n, adj_masks, 5)
                if k5:
                    # Remove a random edge from the clique
                    # To be smarter, we could remove the edge that is part of the fewest other K5s?
                    # For now, random.
                    i, j = random.sample(k5, 2)
                    if i > j: i, j = j, i
                    if (i, j) in edges:
                        edges.remove((i, j))
                        adj_masks[i] &= ~(1 << j)
                        adj_masks[j] &= ~(1 << i)
                    continue
                    
                indep = _find_independent_set_bitmask(n, adj_masks, 5)
                if indep:
                    # Add a random edge within the independent set
                    i, j = random.sample(indep, 2)
                    if i > j: i, j = j, i
                    if (i, j) not in edges:
                        edges.add((i, j))
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
                    continue
                    
                # No violations found
                if n > best_n:
                    best_n = n
                    best_edges = list(edges)
                break
                
            if best_n >= n:
                break
                
    return best_n, best_edges
# EVOLVE-BLOCK-END

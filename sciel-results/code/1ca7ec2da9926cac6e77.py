"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']
    
    # Build adjacency masks
    adj_masks = [0] * n
    for u, v in edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    def get_score(mask):
        """Calculate number of distinct degrees in the subgraph defined by mask."""
        degs = set()
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = (adj_masks[v] & mask).bit_count()
            degs.add(deg)
            m ^= lsb
        return len(degs)
    
    best_val = 0
    best_mask = 0
    
    # Search for each subset size k >= min_verts
    for k in range(min_verts, n + 1):
        # Random Restart Hill Climbing
        for _restart in range(50):
            # Initialize random subset of size k
            init_verts = random.sample(range(n), k)
            mask = 0
            for v in init_verts:
                mask |= (1 << v)
                
            current_score = get_score(mask)
            best_local_mask = mask
            best_local_score = current_score
            
            # Hill Climbing steps
            for _step in range(1000):
                # Pick a vertex to remove and a vertex to add
                # Get list of vertices in mask and not in mask
                # To speed up, we can precompute lists? No, mask changes.
                # n is small, just iterate.
                in_verts = [i for i in range(n) if (mask >> i) & 1]
                out_verts = [i for i in range(n) if not ((mask >> i) & 1)]
                
                if not in_verts or not out_verts:
                    break
                    
                u = random.choice(in_verts)
                w = random.choice(out_verts)
                
                new_mask = mask ^ (1 << u) ^ (1 << w)
                new_score = get_score(new_mask)
                
                if new_score > current_score:
                    mask = new_mask
                    current_score = new_score
                    if current_score > best_local_score:
                        best_local_score = current_score
                        best_local_mask = mask
                    # If we reached theoretical max for this k, break early?
                    # Max distinct degrees is k-1 (as 0 and k-1 can't coexist? 
                    # Actually, can they? No. 
                    # So max is k-1. 
                    # If current_score == k-1, we can't do better for this k.
                    if current_score == k - 1:
                        break
                # We don't accept worse moves (Hill Climbing)
                # To escape local minima, we rely on random restarts.
            
            if best_local_score > best_val:
                best_val = best_local_score
                best_mask = best_local_mask
                
    # Convert best_mask to list of vertices
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i)
            
    return result

# EVOLVE-BLOCK-END

import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Generate Clebsch Graph
    vectors = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vectors.append(i)
            
    adj = [0] * 16
    clebsch_edges = []
    for u_idx in range(16):
        for v_idx in range(u_idx + 1, 16):
            xor = vectors[u_idx] ^ vectors[v_idx]
            if bin(xor).count('1') == 4:
                clebsch_edges.append([u_idx, v_idx])
                adj[u_idx] |= (1 << v_idx)
                adj[v_idx] |= (1 << u_idx)
                
    best_edges = clebsch_edges
    best_count = len(clebsch_edges)  # 40
    
    # 2. Efficient 3-colorability check using backtracking with dynamic ordering
    def is_3_colorable(adj_mask):
        n = 16
        # colors[v] is not stored, we use color_masks
        color_masks = [0, 0, 0]
        assigned = [0] * n # 0: unassigned, 1: color 0, 2: color 1, 3: color 2
        
        # Precompute neighbor masks
        # We need to track available colors dynamically
        # available_colors[v] is a bitmask of 3 bits, 1 if color is available
        
        def backtrack(count_assigned):
            if count_assigned == n:
                return True
            
            # Find unassigned vertex with fewest available colors (MRV heuristic)
            # If tied, pick highest degree
            best_v = -1
            best_avail_mask = 0
            best_deg = -1
            min_avail_count = 4
            
            for v in range(n):
                if assigned[v] != 0:
                    continue
                # Calculate available colors
                # Color c is available if no neighbor has color c
                # Neighbors of v are adj_mask[v]
                # color_masks[c] contains vertices with color c
                # So color c is forbidden if (color_masks[c] & adj_mask[v]) != 0
                
                avail = 0
                for c in range(3):
                    if (color_masks[c] & adj_mask[v]) == 0:
                        avail |= (1 << c)
                        
                # If no colors available, this branch is invalid
                if avail == 0:
                    return False
                
                avail_count = bin(avail).count('1')
                deg = bin(adj_mask[v]).count('1')
                
                if avail_count < min_avail_count or (avail_count == min_avail_count and deg > best_deg):
                    min_avail_count = avail_count
                    best_v = v
                    best_avail_mask = avail
                    best_deg = deg
            
            if best_v == -1:
                return count_assigned == n # Should not happen if count_assigned < n
            
            v = best_v
            # Try each available color
            avail = best_avail_mask
            while avail:
                # Get lowest set bit
                bit = avail & (-avail)
                c = bit.bit_length() - 1
                avail ^= bit
                
                # Assign color c
                color_masks[c] |= (1 << v)
                assigned[v] = c + 1
                
                if backtrack(count_assigned + 1):
                    return True
                
                # Undo
                color_masks[c] &= ~(1 << v)
                assigned[v] = 0
            
            return False
            
        return backtrack(0)
        
    # 3. Local search / Random restarts
    # We know Clebsch is maximal triangle-free.
    # We try to find a 4-chromatic graph with > 40 edges.
    # This requires a graph that is NOT maximal triangle-free, or a different structure.
    # But any triangle-free graph with > 40 edges on 16 vertices must be very dense.
    # Max edges is 64.
    # If we assume 40 is max, we just return it.
    # But to be safe, we run a few quick checks.
    
    all_pairs = [(u, v) for u in range(16) for v in range(u+1, 16)]
    clebsch_set = set((min(u,v), max(u,v)) for u,v in clebsch_edges)
    candidate_edges = [(u,v) for u,v in all_pairs if (u,v) not in clebsch_set]
    
    def can_add_edge(adj_mask, u, v):
        return (adj_mask[u] & adj_mask[v]) == 0
        
    # Try a few random constructions
    for iteration in range(50):
        if time.time() > deadline:
            break
            
        random.shuffle(candidate_edges)
        adj_mask = [0] * 16
        edges = []
        
        for u, v in candidate_edges:
            if can_add_edge(adj_mask, u, v):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.append([u, v])
                
        if len(edges) > best_count:
            if not is_3_colorable(adj_mask):
                best_edges = edges
                best_count = len(edges)
                
    return best_edges

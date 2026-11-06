import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def find_bad_masks(n, adj_mask):
        """
        Returns a list of masks that induce a regular subgraph of size >= 6.
        """
        bad_masks = []
        # Iterate masks
        # Optimization: Skip masks with < 6 bits
        # We can iterate by combinations for k=6 to n?
        # itertools.combinations is slower than bit iteration in Python?
        # Bit iteration is 2^n. 2^22 = 4M.
        # Combinations sum is also ~4M.
        # Let's stick to bit iteration for simplicity and speed of bit_count.
        
        for mask in range(1, 1 << n):
            k = mask.bit_count()
            if k < 6:
                continue
            
            # Check regularity
            m = mask
            first_deg = -1
            is_regular = True
            
            while m:
                v_bit = m & -m
                v = v_bit.bit_length() - 1
                m ^= v_bit
                
                deg = (adj_mask[v] & mask).bit_count()
                
                if first_deg == -1:
                    first_deg = deg
                elif deg != first_deg:
                    is_regular = False
                    break
            
            if is_regular:
                bad_masks.append(mask)
                
        return bad_masks

    def flip_edge(adj_mask, n, u, v):
        if (adj_mask[u] >> v) & 1:
            # Remove
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
        else:
            # Add
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)

    def get_edges(adj_mask, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_mask[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    # Start with n=22
    n = 22
    # Initialize random graph
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    best_n = 0
    best_edges = []
    
    # Local search loop
    while time.time() < deadline - 5.0:
        bad_masks = find_bad_masks(n, adj_mask)
        
        if not bad_masks:
            # Found a valid graph
            best_n = n
            best_edges = get_edges(adj_mask, n)
            break
        
        # If we found bad masks, try to fix them
        # Pick a random bad mask
        bad_mask = random.choice(bad_masks)
        
        # Pick a random edge within this mask to flip
        # Get vertices in mask
        verts = []
        m = bad_mask
        while m:
            v_bit = m & -m
            v = v_bit.bit_length() - 1
            verts.append(v)
            m ^= v_bit
        
        if len(verts) < 2:
            continue
            
        # Pick two random distinct vertices
        u, v = random.sample(verts, 2)
        
        # Flip edge (u, v)
        flip_edge(adj_mask, n, u, v)
        
        # Optional: If we've been stuck for a while, try a different n?
        # Or restart?
        # For now, just keep flipping.
        
    # If we didn't find a valid graph for n=22, try n=21, 20
    if best_n == 0:
        for target_n in [21, 20]:
            if time.time() > deadline - 10.0:
                break
                
            adj_mask = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
            
            # Run a few iterations of local search
            for _ in range(20): # Limit iterations
                if time.time() > deadline - 5.0:
                    break
                bad_masks = find_bad_masks(target_n, adj_mask)
                if not bad_masks:
                    best_n = target_n
                    best_edges = get_edges(adj_mask, target_n)
                    break
                
                bad_mask = random.choice(bad_masks)
                verts = []
                m = bad_mask
                while m:
                    v_bit = m & -m
                    v = v_bit.bit_length() - 1
                    verts.append(v)
                    m ^= v_bit
                
                if len(verts) < 2:
                    continue
                    
                u, v = random.sample(verts, 2)
                flip_edge(adj_mask, target_n, u, v)
                
            if best_n > 0:
                break
                
    # Fallback
    if best_n == 0:
        # Return a small valid graph
        # n=5 is always valid
        return {'n': 5, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}

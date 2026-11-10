import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    def find_violation(adj_masks, n):
        """
        Find a regular induced subgraph of size >= 6.
        Returns (mask, size) if found, else (None, 0).
        """
        # Iterate over all masks
        # For n=21, 2^21 = 2,097,152 masks.
        # This loop must be as fast as possible.
        
        # Optimization: 
        # We can skip masks with < 6 bits.
        # Generating only masks with >= 6 bits is complex in pure Python.
        # Just checking bit_count is fast.
        
        for mask in range(1 << n):
            if mask.bit_count() < 6:
                continue
            
            # Get first vertex
            lsb = mask & (-mask)
            v0 = lsb.bit_length() - 1
            d0 = (adj_masks[v0] & mask).bit_count()
            
            is_regular = True
            m = mask ^ lsb
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                if (adj_masks[v] & mask).bit_count() != d0:
                    is_regular = False
                    break
                m ^= lsb
            
            if is_regular:
                return mask, mask.bit_count()
        
        return None, 0

    def mask_to_edges(adj_masks, n):
        edges = []
        for i in range(n):
            m = adj_masks[i] & ((1 << i) - 1) # Only look at lower bits? No, adj is symmetric.
            # Simpler:
            for j in range(i+1, n):
                if adj_masks[i] & (1 << j):
                    edges.append([i, j])
        return edges

    best_result = {'n': 5, 'edges': []}
    
    # Try n from 21 down to 6
    # 21 is the target for improvement.
    for n in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline - 10:
            break
            
        # Local Search for this n
        found = False
        # More restarts for smaller n, fewer for larger n if time is tight
        max_restarts = 20 if n > 20 else 50
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize random graph
            # Density 0.5
            p = 0.5
            adj_masks = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            # Local Search Loop
            max_iters = 100
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                    
                # Check for violation
                viol_mask, viol_size = find_violation(adj_masks, n)
                
                if viol_mask is None:
                    # Valid graph found
                    edges = mask_to_edges(adj_masks, n)
                    best_result = {'n': n, 'edges': edges}
                    found = True
                    break
                
                # If violation found, try to fix it
                verts = []
                m = viol_mask
                while m:
                    lsb = m & (-m)
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    m ^= lsb
                
                if len(verts) < 2:
                    break
                    
                # Pick two random vertices from the violating subgraph
                v1, v2 = random.sample(verts, 2)
                
                # Flip edge (v1, v2)
                if adj_masks[v1] & (1 << v2):
                    # Remove
                    adj_masks[v1] ^= (1 << v2)
                    adj_masks[v2] ^= (1 << v1)
                else:
                    # Add
                    adj_masks[v1] |= (1 << v2)
                    adj_masks[v2] |= (1 << v1)
            
            if found:
                break
        
        if found:
            break
    
    return best_result

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _is_valid_mask_relaxed(adj_mask, k, m):
    """
    Check if assigning neighbor mask m to vertex k creates a regular induced subgraph
    of size 6 involving k.
    """
    if k < 5:
        return True
        
    vertices = range(k)
    
    # Iterate over all subsets S of size 5 from {0..k-1}
    for S in combinations(vertices, 5):
        mask_S = 0
        for u in S:
            mask_S |= (1 << u)
        
        d_k = (m & mask_S).bit_count()
        
        valid = True
        for u in S:
            d_u = (adj_mask[u] & mask_S).bit_count()
            if m & (1 << u):
                d_u += 1
            if d_u != d_k:
                valid = False
                break
        
        if valid:
            return False
    
    return True

def _check_validity_full(adj_mask, n):
    """
    Full check using bitmasks.
    """
    full_mask = [0] * n
    for i in range(n):
        full_mask[i] |= adj_mask[i]
    
    for i in range(n):
        for j in range(i):
            if adj_mask[j] & (1 << i):
                full_mask[i] |= (1 << j)
    
    for mask in range(1 << n):
        k = mask.bit_count()
        if k < 6:
            continue
        
        v0 = (mask & -mask).bit_length() - 1
        d0 = (full_mask[v0] & mask).bit_count()
        
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if v != v0:
                dv = (full_mask[v] & mask).bit_count()
                if dv != d0:
                    break
            m ^= lsb
        else:
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    best_result = {'n': 5, 'edges': []}
    
    for n in range(22, 9, -1):
        if time.time() > deadline:
            break
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            adj_mask = [0] * n
            
            def dfs(k):
                if k == n:
                    return True
                
                valid_mask = None
                tries = 0
                max_tries = 50
                
                while tries < max_tries:
                    m = random.getrandbits(k)
                    if _is_valid_mask_relaxed(adj_mask, k, m):
                        valid_mask = m
                        break
                    tries += 1
                
                if valid_mask is None:
                    return False
                
                adj_mask[k] = valid_mask
                
                if dfs(k + 1):
                    return True
                
                adj_mask[k] = 0
                return False
            
            if dfs(0):
                if _check_validity_full(adj_mask, n):
                    edges = []
                    for i in range(n):
                        mask = adj_mask[i]
                        while mask:
                            lsb = mask & -mask
                            j = lsb.bit_length() - 1
                            edges.append([j, i])
                            mask ^= lsb
                    return {'n': n, 'edges': edges}
        
    return best_result

# EVOLVE-BLOCK-END

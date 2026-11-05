import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute masks for subsets of size 6 for n=20 and n=21
    # We will do this inside the loop for specific n to save memory if needed, 
    # but it's small enough.
    
    for target_n in [20, 21]:
        if time.time() > deadline - 10:
            break
            
        n = target_n
        # Precompute 6-subsets masks
        masks6 = []
        for subset in combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            masks6.append(mask)
            
        # Also precompute subsets of size 7? 
        # If we get 0 size-6 violations, we check size 7..n.
        
        max_restarts = 20
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random init
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            # Helper to get count and one example
            def get_violation_info(adj):
                count = 0
                example_mask = 0
                for mask in masks6:
                    # Check regularity
                    # Get vertices in mask
                    # We can iterate bits
                    # To speed up, we can check:
                    # Is (adj[v] & mask).bit_count() constant for all v in mask?
                    
                    # Optimization: 
                    # Pick one vertex v0 in mask.
                    # d0 = (adj[v0] & mask).bit_count()
                    # Check others.
                    
                    # Find v0
                    # We can precompute the vertices for each mask? 
                    # Or just use bit manipulation.
                    
                    # Let's precompute the list of vertices for each mask to avoid bit iteration overhead?
                    # 38k masks * 6 vertices = 230k entries. Fine.
                    
                    pass # We will refactor this into a tighter loop
                    
                return count, example_mask

            # Refactored: Precompute vertices for masks
            mask_vertices = []
            for mask in masks6:
                verts = []
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    verts.append(idx)
                    m ^= lsb
                mask_vertices.append(verts)
                
            def count_violations(adj):
                count = 0
                example_mask = 0
                for i, mask in enumerate(masks6):
                    verts = mask_vertices[i]
                    d0 = (adj[verts[0]] & mask).bit_count()
                    regular = True
                    for v in verts[1:]:
                        if (adj[v] & mask).bit_count() != d0:
                            regular = False
                            break
                    if regular:
                        count += 1
                        if count == 1:
                            example_mask = mask
                return count, example_mask

            current_count, current_example = count_violations(adj)
            
            T = 1.0
            cooling = 0.99
            min_T = 0.01
            iterations = 0
            max_iterations = 2000
            
            while T > min_T and iterations < max_iterations:
                if time.time() > deadline - 5:
                    break
                    
                # If we have a violation, try to fix it
                if current_count > 0:
                    # Flip an edge in the example violation
                    mask = current_example
                    verts = []
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        verts.append(idx)
                        m ^= lsb
                    
                    # Pick two random vertices from this subset
                    u, v = random.sample(verts, 2)
                    
                    # Flip edge (u, v)
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    
                    new_count, new_example = count_violations(adj)
                    
                    delta = new_count - current_count
                    
                    if delta < 0 or (random.random() < math.exp(-delta / T)):
                        current_count = new_count
                        current_example = new_example
                    else:
                        # Revert
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                else:
                    # No size-6 violations. Check full validity.
                    # This is expensive, so do it rarely? 
                    # Or just break and check outside.
                    break
                
                T *= cooling
                iterations += 1
                
            # Check if current graph is fully valid
            if current_count == 0:
                # Verify no regular subsets of size > 6
                valid = True
                # Check sizes 7 to n
                # This is expensive. 
                # Optimization: If we have no size 6, it's likely valid.
                # But we must be sure.
                
                # We can check size 7, 8...
                # For n=20, this is ~1M checks.
                # Let's do it.
                
                for k in range(7, n + 1):
                    for subset in combinations(range(n), k):
                        mask = 0
                        for v in subset:
                            mask |= (1 << v)
                        
                        # Check regularity
                        # Get first vertex
                        # We can't easily get vertices from mask here without loop
                        # But we can use a helper
                        verts = []
                        m = mask
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            verts.append(idx)
                            m ^= lsb
                            
                        d0 = (adj[verts[0]] & mask).bit_count()
                        regular = True
                        for v in verts[1:]:
                            if (adj[v] & mask).bit_count() != d0:
                                regular = False
                                break
                        if regular:
                            valid = False
                            break
                    if not valid:
                        break
                            
                if valid:
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
    
    return {'n': 5, 'edges': []}

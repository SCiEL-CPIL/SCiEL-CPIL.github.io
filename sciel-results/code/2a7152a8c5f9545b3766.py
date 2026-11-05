import time
import random
import numpy as np

def _popcount(x):
    return x.bit_count()

def _is_valid_graph_fast(adj_masks, n):
    """
    Check if the graph has no regular induced subgraph on 6 or more vertices.
    adj_masks: list of integers, adj_masks[i] is bitmask of neighbors of i.
    n: number of vertices.
    """
    # Iterate over all masks with popcount >= 6
    # We can generate masks by size or just iterate all and check popcount
    # Iterating all 2^n masks is ~1M for n=20.
    
    for mask in range(1, 1 << n):
        k = mask.bit_count()
        if k < 6:
            continue
        
        # Check if induced subgraph on 'mask' is regular
        # Get vertices in mask
        # degrees = []
        # for v in range(n):
        #     if mask & (1 << v):
        #         deg = (adj_masks[v] & mask).bit_count()
        #         degrees.append(deg)
        
        # Optimization: 
        # If any degree is not equal to the first, return False
        # We need to check if all degrees are equal.
        
        # Extract vertices
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m -= lsb
            
        if not verts:
            continue
            
        first_deg = (adj_masks[verts[0]] & mask).bit_count()
        is_regular = True
        for v in verts[1:]:
            deg = (adj_masks[v] & mask).bit_count()
            if deg != first_deg:
                is_regular = False
                break
        
        if is_regular:
            return False
            
    return True

def _find_violation(adj_masks, n):
    """
    Find one regular induced subgraph of size >= 6.
    Returns (mask, verts) or (None, None).
    """
    for mask in range(1, 1 << n):
        k = mask.bit_count()
        if k < 6:
            continue
        
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m -= lsb
            
        if not verts:
            continue
            
        first_deg = (adj_masks[verts[0]] & mask).bit_count()
        is_regular = True
        for v in verts[1:]:
            deg = (adj_masks[v] & mask).bit_count()
            if deg != first_deg:
                is_regular = False
                break
        
        if is_regular:
            return mask, verts
            
    return None, None

def _edges_from_adj_masks(adj_masks, n):
    edges = []
    for i in range(n):
        m = adj_masks[i]
        # Only count j > i to avoid duplicates
        j = i + 1
        while j < n:
            if m & (1 << j):
                edges.append([i, j])
            j += 1
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try n=20 first, then 21 if time permits (though 21 is very hard)
    # The problem states 20 is known, so we aim for 20.
    
    best_n = 5
    best_edges = []
    
    for target_n in [20, 21, 22]:
        if time.time() > deadline - 5:
            break
            
        n = target_n
        found_valid = False
        
        # Initialize with random graph
        # p=0.5 is standard, but maybe slightly off helps?
        # Let's try a few random restarts
        
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Generate random graph
            adj_masks = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
                        
            # Check validity
            if _is_valid_graph_fast(adj_masks, n):
                edges = _edges_from_adj_masks(adj_masks, n)
                best_n = n
                best_edges = edges
                found_valid = True
                break
                
            # Local Search
            # We have a violation. Find it.
            mask, verts = _find_violation(adj_masks, n)
            
            if mask is None:
                # Should not happen if _is_valid_graph_fast returned False
                continue
                
            # Try to fix by flipping edges
            # We will do a limited number of flips
            for step in range(500):
                if time.time() > deadline - 5:
                    break
                    
                # Pick a random edge within the violation to flip
                # Or pick random pair in violation
                if len(verts) < 2:
                    break
                    
                u, v = random.sample(verts, 2)
                
                # Flip edge u-v
                bit_u = 1 << v
                bit_v = 1 << u
                
                if adj_masks[u] & bit_u:
                    adj_masks[u] ^= bit_u
                    adj_masks[v] ^= bit_v
                else:
                    adj_masks[u] |= bit_u
                    adj_masks[v] |= bit_v
                    
                # Check if this specific violation is fixed
                # Re-calculate degrees for 'verts'
                # Note: 'mask' is the same set of vertices
                first_deg = (adj_masks[verts[0]] & mask).bit_count()
                is_regular = True
                for vv in verts[1:]:
                    deg = (adj_masks[vv] & mask).bit_count()
                    if deg != first_deg:
                        is_regular = False
                        break
                        
                if not is_regular:
                    # Violation fixed. Check global validity.
                    # This is expensive, so we do it only when a violation is fixed.
                    if _is_valid_graph_fast(adj_masks, n):
                        edges = _edges_from_adj_masks(adj_masks, n)
                        best_n = n
                        best_edges = edges
                        found_valid = True
                        break
                    else:
                        # New violations might exist. Find a new one.
                        mask, verts = _find_violation(adj_masks, n)
                        if mask is None:
                            # Should be caught by _is_valid_graph_fast, but just in case
                            found_valid = True
                            break
                        # If we found a new violation, continue loop with new verts
                        # But we need to be careful not to loop infinitely on the same graph
                        # The flip changed the graph, so it's a new state.
                        
                # If violation still exists, we might have created new ones or just failed to fix this one.
                # We can re-find a violation to guide the next flip.
                # To save time, we might just flip another random edge in the current violation.
                
            if found_valid:
                break
                
        if found_valid:
            continue
            
    if best_n <= 5:
        return {'n': 5, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}

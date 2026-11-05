import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def construct():
    """
    Try to build a C4-free graph on n vertices (n up to 47) with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 145  # Leave 5s margin for 150s limit
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple random restarts with the "start dense, remove C4s" strategy
        max_attempts = 2000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            result = _try_construct_dense(n, required_min_deg, deadline, attempt)
            if result is not None:
                n_found, edges_found = result
                return {'n': n_found, 'edges': edges_found}
        
    # Fallback: return a valid smaller graph
    # For n=39, min degree >= 0, so empty graph works
    return {'n': 39, 'edges': []}

def _try_construct_dense(n, required_min_deg, deadline, seed):
    """
    Strategy: Start with a random graph, remove edges to break C4s, check min degree.
    """
    rng = np.random.RandomState(seed)
    
    # We want to end with min degree >= required_min_deg.
    # Start with a graph that is likely to survive C4 removal.
    # C4 removal removes at most ~n^2 edges? No, usually much fewer.
    # Let's start with p=0.6 or 0.7.
    # For n=47, p=0.6 -> deg ~ 27. Required is 8. Plenty of margin.
    
    p = 0.7
    
    # Initialize adjacency matrix as bitmasks
    # adj[i] is an int, bit j is 1 if (i,j) is an edge
    adj = [0] * n
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if rng.rand() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append((i, j))
    
    # Function to check if adding/removing edge creates C4?
    # We are REMOVING edges. So we just need to find existing C4s and break them.
    
    # Iteratively find and break C4s
    # A C4 exists if there are u, v such that |N(u) ∩ N(v)| >= 2.
    # Let common = adj[u] & adj[v].
    # If common has >= 2 bits, say a and b are set.
    # Then u-a-v-b-u is a C4.
    # We can remove edge (a, v) to break this specific C4.
    
    max_iter = 5000
    for _ in range(max_iter):
        if time.time() > deadline:
            return None
            
        c4_found = False
        # Iterate through all pairs to find a C4
        # Optimization: Only check pairs that might have common neighbors?
        # With n=47, n^2/2 ~ 1000 pairs. Fast.
        for u in range(n):
            if time.time() > deadline:
                return None
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                # Check if at least 2 bits are set
                if common & (common - 1): # This checks if >= 2 bits are set
                    # Find two bits
                    # Get the index of the lowest set bit
                    a = (common & -common).bit_length() - 1
                    # Remove the lowest bit and find the next
                    common2 = common & (common - 1)
                    b = (common2 & -common2).bit_length() - 1
                    
                    # We have a C4: u-a-v-b-u
                    # Remove edge (a, v)
                    # Check if edge (a, v) exists
                    if adj[a] & (1 << v):
                        adj[a] &= ~(1 << v)
                        adj[v] &= ~(1 << a)
                        # Update edges list? 
                        # We can just reconstruct edges at the end from adj.
                        c4_found = True
                        break
            if c4_found:
                break
        
        if not c4_found:
            # No C4s left
            break
    
    # Check min degree
    min_deg = min(adj[i].bit_count() for i in range(n))
    
    if min_deg < required_min_deg:
        return None
    
    # Extract edges
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                final_edges.append([i, j])
                
    return (n, final_edges)

# EVOLVE-BLOCK-END

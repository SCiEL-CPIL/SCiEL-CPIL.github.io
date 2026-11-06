import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # We want to maximize n.
    # Known good n is 20.
    # Try 21 first? If it fails quickly, drop to 20.
    
    for n in [21, 20, 19]:
        if time.time() > deadline - 10.0:
            break
            
        # Try to find a valid graph on n vertices
        # Use Local Search
        
        # 1. Generate initial random graph
        # p=0.5 is standard
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # 2. Local Search
        # We will try to remove regular induced subgraphs
        # We keep track of "bad" subsets found recently to prioritize?
        
        max_iter = 5000
        for _ in range(max_iter):
            if time.time() > deadline - 10.0:
                break
                
            bad_subset = find_bad_subset(n, adj)
            if bad_subset is None:
                # Valid!
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # Flip a random edge in or near bad_subset to break regularity
            # To break a regular induced subgraph on S, we can flip an edge inside S.
            # Flipping an edge inside S changes degrees of 2 vertices in S by +/- 1.
            # This will likely break regularity.
            
            # Pick two vertices from bad_subset
            u, v = random.sample(bad_subset, 2)
            
            # Flip edge (u, v)
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        # If local search fails, try a different random start
        # Or just break and try smaller n

    # Fallback
    return {'n': 5, 'edges': []}

def find_bad_subset(n, adj):
    # Check sizes 6, 7, ... up to n
    # To be fast, we check size 6 first.
    # If none, check 7, etc.
    
    # Optimization: Precompute bitmasks?
    # adj is list of ints.
    
    for k in range(6, n + 1):
        # Generate combinations
        # For k=6, n=20, 38760 subsets.
        # For each, check regularity.
        
        # Early exit if found
        for subset in combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Check regularity
            # Get degree of first vertex
            v0 = subset[0]
            d0 = (adj[v0] & mask).bit_count()
            
            is_reg = True
            for v in subset[1:]:
                if (adj[v] & mask).bit_count() != d0:
                    is_reg = False
                    break
            
            if is_reg:
                return list(subset)
                
    return None

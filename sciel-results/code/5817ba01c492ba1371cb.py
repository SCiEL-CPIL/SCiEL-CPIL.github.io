import numpy as np
import time
import itertools
import random

def construct():
    # Target n values in descending order
    target_ns = [21, 20, 15]
    
    # Deadline guard
    start_time = time.time()
    total_time_limit = 115.0 # Leave 5s for return/overhead
    
    for n in target_ns:
        # Allocate time for this n
        # Prioritize larger n
        if n == 21:
            time_budget = 60.0
        elif n == 20:
            time_budget = 40.0
        else:
            time_budget = 10.0
            
        if time.time() - start_time > total_time_limit - time_budget - 5:
            continue
            
        result = search_graph(n, time_budget)
        if result is not None:
            return result
            
    # Fallback: return a small valid graph
    return {'n': 5, 'edges': []}

def search_graph(n, time_budget):
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Initialize random graph
    # Use a density around 0.5
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i, j] = True
                adj[j, i] = True
                
    # Precompute all subsets of size >= 6
    # This is expensive for n=21, but we can generate them on the fly or cache them?
    # For n=21, sum(C(21,k) for k=6..21) is large.
    # C(21,6) = 54264
    # C(21,7) = 116280
    # Total ~ 300k subsets.
    # Generating them once is fine.
    
    subsets = []
    for k in range(6, n+1):
        for combo in itertools.combinations(range(n), k):
            subsets.append(combo)
            
    # Shuffle subsets to avoid bias
    random.shuffle(subsets)
    
    # Repair loop
    for _ in range(10000):
        if time.time() > deadline:
            break
            
        violation_found = False
        # Check subsets. We can break early if we find one and fix it.
        # But we need to ensure we don't miss any.
        # Strategy: Find ALL violations in one pass, then fix one.
        # Or find the first one, fix it, restart scan.
        # Restarting scan is simpler but slower.
        # Let's find the first violation.
        
        for s in subsets:
            if is_regular(adj, s):
                violation_found = True
                # Pick an edge in s to flip
                # Any edge inside s will break regularity of s
                # Choose random edge in s
                u, v = random.sample(s, 2)
                adj[u, v] = not adj[u, v]
                adj[v, u] = adj[u, v]
                break
                
        if not violation_found:
            # No violations found in this scan
            # Verify once more to be sure (in case of race condition or logic error, though single thread)
            # Actually, if we iterated all subsets and found none, it's valid.
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    return None

def is_regular(adj, s):
    # s is a tuple of indices
    if len(s) < 2:
        return True
    # Extract subgraph
    # Using numpy fancy indexing
    sub_adj = adj[np.ix_(s, s)]
    degrees = sub_adj.sum(axis=1)
    # Check if all degrees are equal
    return (degrees == degrees[0]).all()

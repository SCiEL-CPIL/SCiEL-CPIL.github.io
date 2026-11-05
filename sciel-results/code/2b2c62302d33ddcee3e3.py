import numpy as np

def priority(adjacency):
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    degrees = adjacency.sum(axis=1).astype(int)
    
    # Compute Core Numbers
    # Standard peeling algorithm
    core = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    deg = degrees.copy()
    
    # We need to process vertices in increasing degree order
    # But degrees change.
    # Efficient way: bucket sort.
    max_deg = np.max(deg) if n > 0 else 0
    buckets = [[] for _ in range(max_deg + 1)]
    for i in range(n):
        buckets[deg[i]].append(i)
        
    current_k = 0
    processed = 0
    while processed < n:
        # Find smallest k with non-empty bucket
        while current_k <= max_deg and (not buckets[current_k] or not any(remaining[v] for v in buckets[current_k])):
            current_k += 1
        if current_k > max_deg:
            break
            
        v = buckets[current_k].pop()
        if not remaining[v]:
            continue
        remaining[v] = False
        core[v] = current_k
        processed += 1
        
        for u in np.where(adjacency[v])[0]:
            if remaining[u]:
                deg[u] -= 1
                # Move u to new bucket
                # We don't need to physically move, we can just check when we pick from buckets
                # But for simplicity and correctness, let's just update deg.
                # The bucket structure is static, so we need to handle "out of order" picks.
                # Actually, the standard algorithm uses a priority queue or careful bucket management.
                # For N=200, a simple O(N^2) peeling is fine.
                pass
                
    # The bucket implementation above is tricky to get right quickly.
    # Let's use a simpler O(N^2) peeling for small N.
    
    core = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    deg = degrees.copy()
    
    for _ in range(n):
        # Find min degree among remaining
        mask = remaining
        if not np.any(mask): break
        min_deg = np.min(deg[mask])
        # Find all vertices with this min degree
        candidates = np.where(mask & (deg == min_deg))[0]
        # Pick one (arbitrary, e.g., first)
        v = candidates[0]
        remaining[v] = False
        core[v] = min_deg
        # Update neighbors
        for u in np.where(adjacency[v])[0]:
            if remaining[u]:
                deg[u] -= 1
                
    # Score: Lower core number is better. 
    # Tie break by lower degree.
    # Score = - (core * 1000 + degree)
    scores = -(core.astype(float) * 1000 + degrees.astype(float))
    
    return scores

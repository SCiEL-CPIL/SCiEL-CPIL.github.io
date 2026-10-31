import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    D = 11
    
    # Generate all vectors with norm squared 3
    # These have exactly three non-zero entries, each +/- 1
    vectors = []
    indices = []
    
    # Iterate over all combinations of 3 indices
    for combo in itertools.combinations(range(D), 3):
        i, j, k = combo
        # 8 sign combinations
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    v[k] = s3
                    vectors.append(v)
                    
    V = np.array(vectors, dtype=np.int64)
    n = len(V)
    
    # Compute dot product matrix
    # V is (n, D). V.T @ V is (n, n)
    # We need to check if dot > 1
    # Diagonal will be 3. We ignore self-conflicts.
    
    # To save memory/time, we can compute in chunks or just do it all at once
    # 1320x1320 float64 is ~13MB. Fine.
    dots = V @ V.T
    
    # Conflict matrix: True if dots[i,j] > 1
    # We only care about i != j
    conflict = dots > 1
    
    # Set diagonal to False
    np.fill_diagonal(conflict, False)
    
    # Greedy Independent Set
    # We want to select a subset of indices such that no two are connected in conflict graph.
    # Strategy: Repeatedly pick a node with the minimum degree in the remaining graph, add it, and remove it and its neighbors.
    
    # Maintain a set of available indices
    available = set(range(n))
    selected = []
    
    # Precompute degrees? Or calculate on the fly?
    # With n=1320, calculating degree for all available nodes each step is O(n^2) per step -> O(n^3) total.
    # 1320^3 is too slow (2.3 * 10^9).
    
    # Better: Maintain a list of degrees.
    # When we remove a node v, we decrement the degree of all its neighbors.
    
    # Let's use a list of lists for neighbors
    neighbors = [np.where(conflict[i])[0] for i in range(n)]
    
    # Current degrees
    degrees = np.array([len(neighbors[i]) for i in range(n)], dtype=np.int64)
    
    # We need to efficiently find the min degree among available nodes.
    # Since n is small (1320), we can just scan the available list.
    # But we need to update degrees.
    
    # Optimization:
    # Instead of a set, use a boolean array `active`.
    active = np.ones(n, dtype=bool)
    
    while active.any():
        # Find index with min degree among active
        # Mask out inactive
        curr_degrees = np.where(active, degrees, np.inf)
        v = np.argmin(curr_degrees)
        
        if not active[v]:
            break
            
        selected.append(v)
        
        # Deactivate v
        active[v] = False
        
        # Deactivate neighbors of v
        # Get neighbors that are still active
        neigh = neighbors[v]
        # Update degrees of neighbors' neighbors?
        # When we remove v, the degree of each neighbor u decreases by 1 (edge v-u removed).
        # Also, edges between neighbors are still there?
        # Wait, the graph is static. We are removing vertices.
        # The degree of a remaining vertex u is the number of its neighbors that are still active.
        
        # So, for each neighbor u of v:
        # 1. active[u] = False
        # 2. For each neighbor w of u:
        #    if active[w] is True, degrees[w] -= 1
        
        # This is O(sum of degrees of removed nodes).
        # Total work is O(E) where E is number of edges.
        # E is approx 1320 * 49 / 2 ~ 32000.
        # This is very fast.
        
        for u in neigh:
            if active[u]:
                active[u] = False
                # Update degrees of u's neighbors
                for w in neighbors[u]:
                    if active[w]:
                        degrees[w] -= 1
                        
        # Check time
        if time.time() > deadline:
            break
            
    # Construct result
    if not selected:
        # Fallback to seed if something went wrong or time ran out immediately
        # But 1320 is small, so it should finish quickly.
        vectors_seed = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors_seed.append(v)
        return np.array(vectors_seed, dtype=np.int64)
        
    result = V[selected]
    return result

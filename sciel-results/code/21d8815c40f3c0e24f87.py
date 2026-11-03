import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # 1. Generate all weight-3 vectors
    vectors = []
    indices = []
    for support in itertools.combinations(range(D), 3):
        i, j, k = support
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = s1, s2, s3
                    vectors.append(v)
                    indices.append(support)
    
    V = np.array(vectors, dtype=np.int8)
    N = len(V)
    
    # 2. Build Conflict Graph
    # Conflict if dot product >= 2
    # We can compute dot products in chunks to save memory/time
    # Or just check conflicts on the fly during greedy?
    # Precomputing the neighbor list is better for local search.
    
    # Since N=1320, N^2 = 1.7M.
    # Dot product of 1320x11 and 11x1320 is fast.
    # But we only need to know if dot >= 2.
    
    # Let's compute all pairwise dot products
    # This is 1320x1320 matrix.
    # 1320*1320*11 operations ~ 19M. Fast.
    
    dots = V @ V.T
    # Mask out self (diagonal) and valid conflicts
    # Conflict if dots[i][j] >= 2 and i != j
    
    # Build adjacency list
    adj = [set() for _ in range(N)]
    for i in range(N):
        # Find j > i such that dots[i][j] >= 2
        # Using numpy to find indices
        conflicts = np.where(dots[i] >= 2)[0]
        for j in conflicts:
            if j != i:
                adj[i].add(j)
                adj[j].add(i)
                
    # 3. Greedy Independent Set
    # Sort by degree
    degrees = [len(adj[i]) for i in range(N)]
    order = sorted(range(N), key=lambda x: degrees[x])
    
    in_set = [False] * N
    S = []
    
    for i in order:
        if in_set[i]:
            continue
        # Check if i conflicts with any in S
        # Since we process in order, we can just check neighbors in S
        # Optimization: maintain a set of available nodes?
        # For N=1320, simple check is fine.
        
        conflict_found = False
        for j in adj[i]:
            if in_set[j]:
                conflict_found = True
                break
        
        if not conflict_found:
            in_set[i] = True
            S.append(i)
            
    current_size = len(S)
    
    # 4. Local Search
    # Try to improve by swapping
    # Simple 1-for-1 swap: remove u, add v if v has fewer conflicts in S\{u} + {u}?
    # Actually, standard move:
    # Pick u in S. Remove u.
    # Try to add v not in S.
    # If v has no conflicts in S\{u}, add v. Size unchanged.
    # This doesn't increase size.
    
    # To increase size, we need 1-for-k where k < 1? Impossible.
    # So we must look for structures where we remove k and add k+1.
    # Or, simply restart with different orderings.
    
    best_S = S.copy()
    best_size = current_size
    
    # Random Restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Shuffle order
        order = list(range(N))
        np.random.shuffle(order)
        
        in_set = [False] * N
        S = []
        
        for i in order:
            if in_set[i]:
                continue
            conflict_found = False
            for j in adj[i]:
                if in_set[j]:
                    conflict_found = True
                    break
            if not conflict_found:
                in_set[i] = True
                S.append(i)
                
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
            
    # Convert best_S to array
    result = V[best_S]
    return result

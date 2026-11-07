import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def _find_maximal_biclique(uncovered, n, start_edge, deadline):
    """
    Find a maximal biclique in the uncovered subgraph starting from start_edge.
    Returns (L, R) as sets.
    """
    u, v = start_edge
    L = {u}
    R = {v}
    
    # Precompute adjacency in uncovered graph for speed
    # adj_uncov[i] = set of j such that (i,j) is in uncovered
    adj_uncov = [set() for _ in range(n)]
    for edge in uncovered:
        a, b = edge
        adj_uncov[a].add(b)
        adj_uncov[b].add(a)
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        
        # Try to add to L
        new_L = set(L)
        for w in range(n):
            if w not in L and w not in R:
                # w must be connected to all in R in uncovered graph
                if all(r in adj_uncov[w] for r in R):
                    new_L.add(w)
                    changed = True
        L = new_L
        
        # Try to add to R
        new_R = set(R)
        for w in range(n):
            if w not in R and w not in L:
                # w must be connected to all in L in uncovered graph
                if all(l in adj_uncov[w] for l in L):
                    new_R.add(w)
                    changed = True
        R = new_R
        
    return L, R

def _greedy_biclique_partition(uncovered, n, deadline, seed=None):
    """
    Greedy set cover approach:
    1. Find a set of candidate bicliques (maximal or near-maximal).
    2. Greedily select the one that covers the most uncovered edges.
    """
    if seed is not None:
        random.seed(seed)
    
    bicliques = []
    remaining = set(uncovered)
    
    # Precompute all edges as a list for iteration
    all_edges = list(uncovered)
    
    while remaining:
        if time.time() > deadline:
            break
            
        best_biclique = None
        best_cover = 0
        
        # Candidate strategy: 
        # 1. Pick top-K edges by degree in remaining graph
        # 2. For each, try to expand into a biclique
        # 3. Also try some random edges
        
        # Compute degrees in remaining graph
        deg = [0] * n
        for edge in remaining:
            a, b = edge
            deg[a] += 1
            deg[b] += 1
        
        # Get top candidates
        # Sort edges by sum of degrees of endpoints
        sorted_edges = sorted(remaining, key=lambda e: deg[e[0]] + deg[e[1]], reverse=True)
        
        # Take top 10 + some random
        candidates = sorted_edges[:10]
        if len(remaining) > 20:
            random_candidates = random.sample(list(remaining), min(10, len(remaining)))
            candidates.extend(random_candidates)
        
        for edge in candidates:
            if time.time() > deadline:
                break
            L, R = _find_maximal_biclique(remaining, n, edge, deadline)
            cover_count = len(L) * len(R)
            # Verify all edges are actually in remaining
            actual_cover = 0
            for l in L:
                for r in R:
                    if (l, r) in remaining:
                        actual_cover += 1
            if actual_cover > best_cover:
                best_cover = actual_cover
                best_biclique = (L, R)
        
        if best_biclique is None or best_cover == 0:
            # Fallback: single edge
            edge = next(iter(remaining))
            u, v = edge
            best_biclique = ({u}, {v})
            best_cover = 1
        
        L, R = best_biclique
        bicliques.append((L, R))
        
        # Remove covered edges from remaining
        for l in L:
            for r in R:
                edge = _edge_key(l, r)
                if edge in remaining:
                    remaining.remove(edge)
    
    # Handle any leftover edges
    for edge in remaining:
        u, v = edge
        bicliques.append(({u}, {v}))
    
    return bicliques

def _try_merge(bicliques, remaining_edges_set, n):
    """
    Try to merge compatible bicliques.
    Two bicliques (L1, R1) and (L2, R2) can be merged if:
    - The union forms a valid biclique (all cross edges exist in original graph)
    - The union covers a subset of the original edges (no overlap in covered edges)
    
    Actually, for a partition, merging is tricky. Let's try a different approach:
    After initial partition, try to see if we can combine two bicliques into one
    that still covers exactly their union of edges.
    """
    # This is complex. Let's skip for now and rely on better initial selection.
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin (90s limit, 15s buffer)
    
    n = P['n']
    edges = P['edges']
    
    # Build edge set
    edge_set = set()
    for u, v in edges:
        edge_set.add(_edge_key(u, v))
    
    best_solution = None
    best_count = float('inf')
    
    # Try multiple random seeds to get different partitions
    for seed in range(50):
        if time.time() > deadline:
            break
        
        bicliques = _greedy_biclique_partition(edge_set, n, deadline, seed=seed)
        
        count = len(bicliques)
        if count < best_count:
            best_count = count
            best_solution = bicliques
            
        # Early stop if we get a very good solution
        if count <= 10:
            break
    
    if best_solution is None:
        # Fallback: single edge bicliques
        best_solution = [({u}, {v}) for u, v in edge_set]
    
    # Convert to required format
    result = []
    for L, R in best_solution:
        result.append([list(L), list(R)])
    
    return result

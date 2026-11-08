"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # 1. Generate Candidates: Weight-4 vectors
    # C(11, 4) * 16 = 330 * 16 = 5280
    candidates = []
    for support in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)

    cand_arr = np.array(candidates, dtype=np.int64)
    n_cands = len(cand_arr)

    # 2. Build Conflict Graph
    # Conflict if dot product > 2
    G = (cand_arr.astype(np.int8) @ cand_arr.T.astype(np.int8))
    
    # Build adjacency lists
    neighbors = [[] for _ in range(n_cands)]
    for i in range(n_cands):
        js = np.where(G[i] > 2)[0]
        js = js[js != i]
        neighbors[i] = js.tolist()
        
    # 3. Spectral Graph Theory Approach
    # Construct adjacency matrix A (0/1) where A[i,j] = 1 if i,j conflict
    A = np.zeros((n_cands, n_cands), dtype=np.float32)
    for i in range(n_cands):
        if neighbors[i]:
            A[i, neighbors[i]] = 1.0
            
    # The adjacency matrix is symmetric.
    # We want to find an independent set.
    # An independent set corresponds to a subset of vertices with no edges between them.
    # In the context of the adjacency matrix, if we look at the eigenvectors,
    # the eigenvector corresponding to the largest eigenvalue often highlights the densest cluster.
    # However, for independent sets, we are looking for a "sparse" structure in the graph.
    # 
    # Alternative view:
    # Let x be a vector where x_i = 1 if i is in the independent set, 0 otherwise.
    # We want to maximize sum(x) subject to x_i + x_j <= 1 for all (i,j) in E.
    # This is the Maximum Independent Set problem.
    #
    # Spectral relaxation:
    # Consider the Laplacian L = D - A.
    # The smallest non-zero eigenvalue of L relates to connectivity.
    # 
    # A simpler heuristic based on eigenvalues of A:
    # The eigenvector associated with the largest eigenvalue (lambda_max) has large components
    # on the nodes that are most "centrally connected".
    # Conversely, nodes with small absolute values in the principal eigenvector
    # might be less connected to the "core" and thus easier to include in an independent set
    # without conflicting with many others?
    # 
    # Actually, a better spectral heuristic for Independent Set:
    # Look at the eigenvector corresponding to the smallest eigenvalue of A?
    # Or, consider that an independent set S implies that for i in S, sum_{j in N(i)} x_j = 0.
    # 
    # Let's try a different spectral approach:
    # The adjacency matrix A can be decomposed.
    # If we take the eigenvector v associated with the largest eigenvalue,
    # the entries v_i indicate the "influence" of node i.
    # 
    # Let's try selecting nodes based on the sign of the components of the
    # eigenvector corresponding to the largest eigenvalue.
    # In a bipartite-like structure, this separates the two parts.
    # Our graph is not bipartite (it has triangles? 
    # v1=(1,1,1,1,0...), v2=(1,1,-1,-1,0...), v3=(1,-1,1,-1,0...)
    # v1.v2 = 1+1-1-1 = 0. v1.v3 = 1-1+1-1 = 0. v2.v3 = 1-1-1+1 = 0.
    # Wait, v1, v2, v3 are mutually orthogonal?
    # Let's check v1=(1,1,1,1,0,0,0,0,0,0,0) and v2=(1,1,1,-1,0,0,0,0,0,0,0).
    # Dot = 1+1+1-1 = 2. Conflict!
    # So triangles exist.
    
    # Let's use the eigenvector of the largest eigenvalue.
    # Nodes with large positive values are in the "core".
    # Nodes with large negative values are in the "anti-core".
    # An independent set can be formed by taking the positive part or the negative part?
    # No, that's for bipartite graphs.
    
    # Let's try a greedy algorithm guided by the spectral embedding.
    # Sort nodes by their component in the principal eigenvector.
    # Try to build an independent set by picking nodes in order of decreasing "spectral centrality"
    # or maybe increasing?
    
    # Let's compute the principal eigenvector.
    # For speed, we can use power iteration or just np.linalg.eigh on a subspace?
    # n=5280 is small enough for full eigh if needed, but let's try power iteration for speed.
    
    def power_iteration(A, k=1, max_iter=100):
        n = A.shape[0]
        v = np.ones(n, dtype=np.float32)
        v /= np.linalg.norm(v)
        for _ in range(max_iter):
            v_new = A @ v
            norm = np.linalg.norm(v_new)
            if norm == 0:
                break
            v_new /= norm
            if np.linalg.norm(v_new - v) < 1e-6:
                break
            v = v_new
        return v
    
    v_principal = power_iteration(A)
    
    # Heuristic 1: Sort by v_principal values.
    # Pick the node with the highest value, remove its neighbors, repeat.
    # This is a greedy independent set with a fixed ordering.
    
    # Heuristic 2: Randomized greedy with spectral tie-breaking.
    # When choosing among min-degree nodes, prefer those with higher v_principal value.
    
    rng = np.random.default_rng(42)
    best_set = np.array([], dtype=np.int64)
    best_size = 0
    
    def run_greedy_spectral(seed_offset=0):
        if time.time() > deadline - 5:
            return None
            
        deg = np.array([len(neighbors[i]) for i in range(n_cands)], dtype=np.int32)
        selected = []
        remaining = set(range(n_cands))
        
        while remaining:
            if time.time() > deadline - 2:
                break
                
            # Find min degree among remaining
            min_deg = min(deg[i] for i in remaining)
            
            # Get all nodes with min degree
            candidates = [i for i in remaining if deg[i] == min_deg]
            
            # Sort candidates by spectral value (v_principal)
            # Higher value means more "central"
            # Maybe we should pick the one with the LOWEST value to avoid the core?
            # Or the HIGHEST?
            # Let's try HIGHEST first.
            candidates_sorted = sorted(candidates, key=lambda i: v_principal[i], reverse=True)
            
            # Pick the first one (or random among top few?)
            # Let's pick the top one deterministically for this variant
            u = candidates_sorted[0]
            
            selected.append(u)
            remaining.remove(u)
            
            # Remove neighbors
            for v in neighbors[u]:
                if v in remaining:
                    remaining.remove(v)
                    for w in neighbors[v]:
                        if w in remaining:
                            deg[w] -= 1
                            
        return selected

    def run_greedy_spectral_random(seed_offset=0):
        if time.time() > deadline - 5:
            return None
            
        deg = np.array([len(neighbors[i]) for i in range(n_cands)], dtype=np.int32)
        selected = []
        remaining = set(range(n_cands))
        
        while remaining:
            if time.time() > deadline - 2:
                break
                
            # Find min degree among remaining
            min_deg = min(deg[i] for i in remaining)
            
            # Get all nodes with min degree
            candidates = [i for i in remaining if deg[i] == min_deg]
            
            # Sort candidates by spectral value
            candidates_sorted = sorted(candidates, key=lambda i: v_principal[i], reverse=True)
            
            # Pick randomly from the top 10% or top 5
            k = max(1, len(candidates_sorted) // 10)
            u = rng.choice(candidates_sorted[:k])
            
            selected.append(u)
            remaining.remove(u)
            
            # Remove neighbors
            for v in neighbors[u]:
                if v in neighbors[u]:
                    if v in remaining:
                        remaining.remove(v)
                        for w in neighbors[v]:
                            if w in remaining:
                                deg[w] -= 1
                            
        return selected

    # Also try the standard random greedy as a baseline
    def run_greedy_random(seed_offset=0):
        if time.time() > deadline - 5:
            return None
            
        deg = np.array([len(neighbors[i]) for i in range(n_cands)], dtype=np.int32)
        selected = []
        remaining = set(range(n_cands))
        
        while remaining:
            if time.time() > deadline - 2:
                break
                
            min_deg = min(deg[i] for i in remaining)
            candidates = [i for i in remaining if deg[i] == min_deg]
            u = rng.choice(candidates)
            
            selected.append(u)
            remaining.remove(u)
            
            for v in neighbors[u]:
                if v in remaining:
                    remaining.remove(v)
                    for w in neighbors[v]:
                        if w in remaining:
                            deg[w] -= 1
                            
        return selected

    # Run multiple restarts
    for r in range(50):
        if time.time() > deadline - 10:
            break
        
        if r % 3 == 0:
            result = run_greedy_spectral(r)
        elif r % 3 == 1:
            result = run_greedy_spectral_random(r)
        else:
            result = run_greedy_random(r)
            
        if result is not None and len(result) > best_size:
            best_size = len(result)
            best_set = cand_arr[result].copy()
            
    # Fallback
    if best_size == 0:
        best_set = cand_arr[:1]
        best_size = 1

    return best_set

# EVOLVE-BLOCK-END

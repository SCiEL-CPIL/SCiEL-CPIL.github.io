import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 115  # Leave buffer for overhead
    
    # Try to find a graph for n=35, then 34, etc.
    # 35 is the known maximum.
    for n in range(35, 16, -1):
        # Multiple spectral restarts for each n
        for restart in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # Spectral Initialization
            # Generate random symmetric matrix
            A = np.random.randn(n, n)
            A = (A + A.T) / 2.0
            
            # Get eigenvectors
            # Choose k to balance structure and randomness
            # k=1 gives a cut, k=n/2 is dense.
            # Ramsey graphs are roughly 50% dense.
            # Top eigenvectors capture the largest variance.
            k = random.randint(3, max(3, n // 3))
            
            try:
                eigvals, eigvecs = np.linalg.eigh(A)
            except np.linalg.LinAlgError:
                continue
            
            # Sort by eigenvalue magnitude (descending)
            idx = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, idx]
            
            # Take top k eigenvectors
            U = eigvecs[:, :k]
            
            # Compute Gram matrix G = U U^T
            G = U @ U.T
            
            # Binarize
            # We want a graph with ~50% edges.
            # The diagonal of G is sum of squares of U's columns = 1 (if eigenvectors are normalized).
            # Off-diagonal entries are dot products.
            # Median threshold ensures 50% edges (roughly).
            mask = np.ones((n, n), dtype=bool)
            np.fill_diagonal(mask, False)
            vals = G[mask]
            if len(vals) == 0:
                continue
            thresh = np.median(vals)
            
            adj = (G > thresh).astype(np.int8)
            np.fill_diagonal(adj, 0)
            
            # Local Search to repair violations
            result = local_search(adj, n, start_time, time_limit)
            if result:
                return result
                
    # Fallback: Return the best known small graph (Paley 17)
    return paley_graph(17)

def local_search(adj, n, start_time, time_limit):
    # adj is a numpy array n x n
    for step in range(20000):
        if time.time() - start_time > time_limit:
            return None
            
        # Check for K4
        k4 = find_k4_fast(adj, n)
        if k4:
            # Pick random edge in K4 to remove
            u, v, w, x = k4
            edges = [(u, v), (u, w), (u, x), (v, w), (v, x), (w, x)]
            a, b = random.choice(edges)
            adj[a, b] = 0
            adj[b, a] = 0
            continue
            
        # Check for Ind6
        ind6 = find_ind6_fast(adj, n)
        if ind6:
            # Pick random pair in Ind6 to add edge
            pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i+1, 6)]
            a, b = random.choice(pairs)
            adj[a, b] = 1
            adj[b, a] = 1
            continue
            
        # No violations found
        # Verify exactly
        if not find_k4_fast(adj, n) and not find_ind6_fast(adj, n):
            # Convert to edge list
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    return None

def find_k4_fast(adj, n):
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                # Common neighbors
                common = np.where(adj[i] & adj[j])[0]
                if len(common) < 2:
                    continue
                # Check edges within common
                for k_idx in range(len(common)):
                    u = common[k_idx]
                    for l_idx in range(k_idx+1, len(common)):
                        v = common[l_idx]
                        if adj[u, v]:
                            return [i, j, u, v]
    return None

def find_ind6_fast(adj, n):
    # Order vertices by degree (ascending) to prune DFS
    degrees = np.sum(adj, axis=1)
    order = np.argsort(degrees).tolist()
    
    def dfs(candidates, current_set):
        if len(current_set) == 6:
            return current_set[:]
        if len(candidates) + len(current_set) < 6:
            return None
        
        for idx in range(len(candidates)):
            v = candidates[idx]
            # New candidates: those in candidates[idx+1:] not connected to v
            new_candidates = []
            for w in candidates[idx+1:]:
                if not adj[v, w]:
                    new_candidates.append(w)
            
            res = dfs(new_candidates, current_set + [v])
            if res:
                return res
        return None
        
    return dfs(order, [])

def paley_graph(q):
    squares = {x * x % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i+1, q):
            if (j - i) % q in squares:
                edges.append([i, j])
    return {'n': q, 'edges': edges}

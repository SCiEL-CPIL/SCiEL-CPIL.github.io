import time
import random
import numpy as np

_DEADLINE = time.time() + 145.0

def _time_left():
    return _DEADLINE - time.time()

def _k4_count(n, adj):
    """Exact number of K4 subgraphs."""
    c = 0
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            common = adj[i] & adj[j]
            for k in common:
                if k <= j:
                    continue
                c += len(adj[i] & adj[j] & adj[k])
    return c

def _indep6_count(n, adj, limit=200):
    """Exact number of independent 6-sets, capped at `limit`."""
    comp = [set() for _ in range(n)]
    for i in range(n):
        for j in adj[i]:
            if j > i:
                comp[i].add(j)
                comp[j].add(i)

    count = 0

    def rec(cands, size):
        nonlocal count
        if size == 0:
            count += 1
            return
        if len(cands) < size:
            return
        if count >= limit:
            return
        for v in list(cands):
            if size == 1:
                count += 1
                if count >= limit:
                    return
            else:
                nc = cands & comp[v]
                if len(nc) >= size - 1:
                    rec(nc, size - 1)
            cands.discard(v)

    rec(set(range(n)), 6)
    return count

def _is_valid(n, adj):
    return _k4_count(n, adj) == 0 and _indep6_count(n, adj, limit=1) == 0

def _to_dict(n, adj):
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    return {'n': n, 'edges': edges}

def _spectral_graph(n, seed_offset=0):
    """Generate a graph using spectral methods."""
    rng = np.random.default_rng(42 + seed_offset)
    # Random symmetric matrix
    A = rng.normal(size=(n, n))
    A = (A + A.T) / 2.0
    
    # Get eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eigh(A)
    except np.linalg.LinAlgError:
        return None
        
    # Use top k eigenvectors? Or just sign of projection?
    # Simple approach: Sign of the largest eigenvector component?
    # Or: Threshold the matrix entries based on some statistic.
    # Better: Use the sign of the elements of the top eigenvector to define a partition?
    # No, we need a graph.
    # Let's try: Create a matrix M from top eigenvectors.
    # Take top 3 eigenvectors, concatenate, then threshold?
    # Or simpler: Use the adjacency matrix of the graph defined by:
    # Edge (i,j) exists if eigvec_top[i] * eigvec_top[j] > 0 ?
    # This creates a complete bipartite graph structure (or similar).
    # Let's try a more robust "random regular" approach via spectral gap?
    
    # Alternative: Generate random graph with given degree, then check.
    # But the directive says "spectral construction... top-k eigenvectors... binarizing signs".
    
    # Let's use the top eigenvector of a random symmetric matrix.
    # v = eigvecs[:, 0]
    # This splits vertices into two sets.
    # If we connect i,j if v[i]*v[j] > 0, we get two cliques. K4 will exist if clique size >= 4.
    # So we need to be careful.
    
    # Let's try: M = eigvecs[:, :k] @ eigvecs[:, :k].T
    # Then threshold M.
    k = 3
    V = eigvecs[:, :k]
    M = V @ V.T
    
    # Normalize
    M = M / np.max(np.abs(M))
    
    # Threshold: edge if M[i,j] > 0?
    # This might create dense graphs.
    # We want sparse-ish to avoid K4.
    # Let's try thresholding at 0.
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if M[i, j] > 0.1: # Tune threshold
                adj[i].add(j)
                adj[j].add(i)
    return adj

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    random.seed(42)
    np.random.seed(42)

    best_n = 0
    best_adj = None

    # Try to find a valid graph with n=35 or 36 using spectral methods
    # Start with high n and decrease if not valid? Or start low and increase?
    # Spectral methods are better for specific n.
    # Let's try n=35, 36, 30, 25, 20...
    
    target_ns = [36, 35, 34, 33, 32, 31, 30, 28, 25, 20]
    
    for n in target_ns:
        if _time_left() < 5:
            break
            
        found = False
        attempts = 0
        while _time_left() > 5 and not found and attempts < 50:
            attempts += 1
            adj = _spectral_graph(n, attempts)
            if adj is None:
                continue
            if _is_valid(n, adj):
                best_n = n
                best_adj = adj
                found = True
                break
        
        if found:
            # If we found 35 or 36, we can stop if we want max n.
            # But maybe we can find 36 after 35?
            # The list is descending, so first found is max.
            break

    # Fallback to Paley 17 if nothing found
    if best_n < 17:
        n = 17
        sq = {x * x % n for x in range(1, n)}
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n in sq:
                    adj[i].add(j)
                    adj[j].add(i)
        best_n = n
        best_adj = adj

    return _to_dict(best_n, best_adj)

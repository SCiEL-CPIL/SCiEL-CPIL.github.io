"""Construct a graph with no regular induced subgraph on >= 6 vertices."""
import time
import random
import numpy as np
from itertools import combinations

_START_TIME = time.time()
DEADLINE = 125.0

def _elapsed():
    return time.time() - _START_TIME

def _is_regular_subset_numpy(subset, adj_np):
    """Check if induced subgraph on subset is regular using NumPy."""
    if len(subset) < 6:
        return False
    # Extract submatrix
    sub_adj = adj_np[np.ix_(subset, subset)]
    # Calculate degrees (sum of rows)
    degrees = sub_adj.sum(axis=1)
    # Check if all degrees are equal
    if len(np.unique(degrees)) == 1:
        return True
    return False

def _has_violation_numpy(n, adj_np):
    """Check all subsets of size >= 6 for regularity."""
    for size in range(6, n + 1):
        for sub in combinations(range(n), size):
            if _is_regular_subset_numpy(sub, adj_np):
                return True
    return False

def _spectral_gap(adj_np):
    """Calculate a measure of expansion. Returns higher for better expanders."""
    # Normalize adjacency matrix for eigenvalues?
    # For regular-ish graphs, lambda_1 is approx degree.
    # We want small lambda_2.
    # Use algebraic connectivity or just 1 - |lambda_2|/|lambda_1|
    try:
        eigvals = np.linalg.eigvalsh(adj_np)
        # eigvals are sorted ascending
        lambda_1 = eigvals[-1]
        lambda_2 = eigvals[-2]
        if lambda_1 == 0:
            return 0
        return 1 - abs(lambda_2) / abs(lambda_1)
    except Exception:
        return 0

def _edges_from_adj_np(adj_np, n):
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj_np[u, v] == 1:
                edges.append([u, v])
    return edges

def construct():
    global _START_TIME, DEADLINE
    _START_TIME = time.time()
    DEADLINE = _START_TIME + 125.0
    
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    
    best_n = 5
    best_edges = []
    
    # Try to find a graph for n=21, then 20, etc.
    for target_n in [21, 20]:
        if _elapsed() > DEADLINE - 20:
            break
            
        # Generate a batch of random graphs
        candidates = []
        num_candidates = 1000
        for _ in range(num_candidates):
            if _elapsed() > DEADLINE - 10:
                break
            # Generate random graph
            # Use symmetric matrix
            adj_list = np_rng.random((target_n, target_n))
            adj_np = (adj_list > 0.5).astype(int)
            # Symmetrize
            adj_np = (adj_np + adj_np.T) // 2
            # Zero diagonal
            np.fill_diagonal(adj_np, 0)
            
            # Calculate spectral gap
            gap = _spectral_gap(adj_np)
            candidates.append((gap, adj_np))
        
        if not candidates:
            continue
            
        # Sort by spectral gap (highest first)
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Verify top candidates
        # We check top 50 to save time
        top_k = 50
        for gap, adj_np in candidates[:top_k]:
            if _elapsed() > DEADLINE - 5:
                break
                
            # Exact check
            if not _has_violation_numpy(target_n, adj_np):
                edges = _edges_from_adj_np(adj_np, target_n)
                return {'n': target_n, 'edges': edges}
                
    # Fallback to smaller n if 20/21 failed
    for target_n in [19, 18]:
        if _elapsed() > DEADLINE - 5:
            break
        # Try a few random graphs
        for _ in range(100):
            if _elapsed() > DEADLINE - 2:
                break
            adj_list = np_rng.random((target_n, target_n))
            adj_np = (adj_list > 0.5).astype(int)
            adj_np = (adj_np + adj_np.T) // 2
            np.fill_diagonal(adj_np, 0)
            
            if not _has_violation_numpy(target_n, adj_np):
                edges = _edges_from_adj_np(adj_np, target_n)
                return {'n': target_n, 'edges': edges}
                
    return {'n': 5, 'edges': []}

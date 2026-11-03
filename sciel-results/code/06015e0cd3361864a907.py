import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 5]

def _build_tensor_indices(n, m, p):
    """Return list of (a, b, c) tuples for the non-zero entries of the matmul tensor."""
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
    return indices

def _verify_decomposition_fast(U, V, W, n, m, p, indices, tol=1e-6):
    """
    Fast verification using matrix operations.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    """
    r = U.shape[1]
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the full tensor from decomposition
    # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    # Use einsum for efficiency
    T = np.einsum('at,bt,ct->abc', U, V, W)
    
    # Check non-zero entries
    for (a, b, c) in indices:
        if abs(T[a, b, c] - 1.0) > tol:
            return False
    
    # Check zero entries
    nonzero_set = set(indices)
    for a in range(dim_u):
        for b in range(dim_v):
            for c in range(dim_w):
                if (a, b, c) in nonzero_set:
                    continue
                if abs(T[a, b, c]) > tol:
                    return False
    return True

def _construct_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition of rank n*m*p."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    rank = n * m * p
    
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    return U, V, W

def _try_rank_reduction(U, V, W, n, m, p, indices, deadline, max_reduction=5):
    """
    Try to reduce the rank by finding linear dependencies.
    Returns (U, V, W, new_rank) or (None, None, None, original_rank) if no reduction found.
    """
    dim_u, dim_v, dim_w = n * m, m * p, p * n
    original_rank = U.shape[1]
    
    # Try to find if we can eliminate columns by checking if one rank-1 tensor
    # can be expressed as a linear combination of others
    # This is a simplified approach - try greedy elimination
    
    for reduction in range(1, max_reduction + 1):
        if time.time() > deadline:
            break
        
        # Try to find a column that is linearly dependent on others
        # For each column t, check if removing it still allows reconstruction
        # by adjusting other columns
        
        # Simplified: try to find if any two columns have the same (a,b,c) pattern
        # and can be merged
        found_merge = False
        for t1 in range(original_rank):
            if time.time() > deadline:
                break
            for t2 in range(t1 + 1, original_rank):
                # Check if column t1 and t2 have identical patterns
                if np.allclose(U[:, t1], U[:, t2]) and \
                   np.allclose(V[:, t1], V[:, t2]) and \
                   np.allclose(W[:, t1], W[:, t2]):
                    # They are identical, so we can merge (but sum would be 2, not 1)
                    # Actually for matmul tensor, each (a,b,c) appears exactly once,
                    # so this shouldn't happen in schoolbook. Skip.
                    pass
                # Check if one is a scalar multiple that allows cancellation
                # This is complex; skip for now
        # No simple merges found in schoolbook for distinct indices
        
    return U, V, W, original_rank

def _structured_search(n, m, p, indices, deadline, start_rank=60, target_rank=50):
    """
    Structured search for lower-rank decompositions.
    Uses the fact that matmul tensors have structure that can be exploited.
    """
    dim_u, dim_v, dim_w = n * m, m * p, p * n
    
    # Strategy: Try to find a decomposition with rank < n*m*p
    # by exploiting the tensor structure
    
    # For now, return schoolbook as baseline
    U, V, W = _construct_schoolbook(n, m, p)
    
    # Try to reduce rank by 1, 2, ... using optimization
    # This is a placeholder for more sophisticated search
    
    return U, V, W, U.shape[1]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120s limit with safety margin
    
    # Get schoolbook as baseline
    U0, V0, W0 = _construct_schoolbook(n, m, p)
    indices = _build_tensor_indices(n, m, p)
    
    # Verify schoolbook
    if _verify_decomposition_fast(U0, V0, W0, n, m, p, indices):
        best_U, best_V, best_W = U0, V0, W0
        best_rank = U0.shape[1]
    else:
        # Should not happen, but handle it
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    # Try to find lower rank
    if time.time() < deadline - 30:  # Leave 30s safety margin
        U, V, W, rank = _structured_search(n, m, p, indices, deadline)
        if rank < best_rank:
            if _verify_decomposition_fast(U, V, W, n, m, p, indices):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

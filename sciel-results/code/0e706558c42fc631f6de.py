import time
import numpy as np

DIMENSIONS = [3, 5, 5]
_START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _build_schoolbook(n, m, p):
    """Build full schoolbook decomposition, return (U, V, W) as numpy arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for idx in range(rank):
        i = idx // (m * p)
        j = (idx // p) % m
        k = idx % p
        U[i * m + j, idx] = 1.0
        V[j * p + k, idx] = 1.0
        W[k * n + i, idx] = 1.0
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that U ⊗ V ⊗ W equals the matmul tensor."""
    rank = U.shape[1]
    # Tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Reshape: U is (n*m, r) -> (n, m, r), V is (m*p, r) -> (m, p, r), W is (p*n, r) -> (p, n, r)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    # Expected: T[i,j,k] = 1 if i,j,k are "aligned" for matmul
    # For matmul C = A*B: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i,j,k] = 1 for all valid (i,j,k) since it's the identity tensor for matmul
    # Actually the matmul tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because it represents the bilinear form: sum_{i,j,k} A[i,j]*B[j,k]*C[i,k]
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)

def _try_reduce_rank(U, V, W, n, m, p, max_iters=5000, seed=42):
    """Try to reduce rank by finding linear dependencies and eliminating columns.
    Returns best (U, V, W, rank) found."""
    rng = np.random.RandomState(seed)
    U = U.copy()
    V = V.copy()
    W = W.copy()
    current_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    for it in range(max_iters):
        if time.time() - _START_TIME > TIME_LIMIT:
            break
        if current_rank <= 0:
            break
            
        # Try to find a column r that can be eliminated
        # by expressing it as a linear combination of others
        # Strategy: pick a random column, check if it's in the span of others
        # for all three matrices simultaneously
        
        # Simpler approach: try removing each column and check validity
        # But that's O(rank) checks per step. Instead, use a heuristic:
        # Look for columns where U[:,r] is a linear combo of other U columns,
        # AND V[:,r] is a linear combo of other V columns,
        # AND W[:,r] is a linear combo of other W columns.
        
        # Even simpler: randomly permute and try to remove columns one by one
        # Check if the remaining decomposition still works.
        
        # Pick a random column to try removing
        r = rng.randint(current_rank)
        
        # Try removing column r
        mask = np.ones(current_rank, dtype=bool)
        mask[r] = False
        U_test = U[:, mask]
        V_test = V[:, mask]
        W_test = W[:, mask]
        
        if _verify(U_test, V_test, W_test, n, m, p):
            U, V, W = U_test, V_test, W_test
            current_rank -= 1
            if current_rank < best_rank:
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = current_rank
        else:
            # Try a small perturbation: modify column r slightly
            # This won't help with exact verification, so skip
            pass
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Start with schoolbook (rank 75)
    U, V, W = _build_schoolbook(n, m, p)
    
    # Try to reduce rank
    U, V, W, rank = _try_reduce_rank(U, V, W, n, m, p, max_iters=10000, seed=42)
    
    if time.time() - _START_TIME > TIME_LIMIT - 5:
        # Fallback to known good result
        U, V, W = _build_schoolbook(n, m, p)
        rank = n * m * p
    
    # Convert to list of lists with integer/half-integer entries
    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _validate_fast(U, V, W, n, m, p):
    """Fast validation using numpy vectorization."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 for all i,j,k
    
    # Precompute index arrays
    I, J, K = np.indices((n, m, p))
    idx_u = I * m + J  # (n,m,p)
    idx_v = J * p + K  # (n,m,p)
    idx_w = K * n + I  # (n,m,p)
    
    # U[idx_u, r] * V[idx_v, r] * W[idx_w, r], summed over r
    # Shape: (n,m,p,rank) -> sum over last axis
    U_exp = U[idx_u, :]  # (n,m,p,rank)
    V_exp = V[idx_v, :]  # (n,m,p,rank)
    W_exp = W[idx_w, :]  # (n,m,p,rank)
    
    T = np.sum(U_exp * V_exp * W_exp, axis=3)  # (n,m,p)
    
    return np.allclose(T, 1.0, atol=1e-9)

def _tensor_flattening_ranks(n, m, p):
    """Compute the three matrix-flattening ranks of the multiplication tensor."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    r1 = np.linalg.matrix_rank(T.reshape(nm, mp * pn))
    r2 = np.linalg.matrix_rank(T.transpose(1, 0, 2).reshape(mp, nm * pn))
    r3 = np.linalg.matrix_rank(T.transpose(2, 0, 1).reshape(pn, nm * mp))
    return r1, r2, r3

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Compute lower bound from tensor flattening ranks
    r1, r2, r3 = _tensor_flattening_ranks(n, m, p)
    lower_bound = max(r1, r2, r3)
    
    # Start with schoolbook decomposition
    rank = n * m * p  # 63
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Verify schoolbook is valid
    if not _validate_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # If lower bound equals schoolbook rank, no improvement possible
    if lower_bound >= rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to reduce rank by removing columns (greedy)
    current_rank = rank
    improved = True
    while improved and current_rank > lower_bound and time.time() < deadline - 10:
        improved = False
        for col_idx in range(current_rank):
            if time.time() >= deadline - 10:
                break
            
            U_test = np.delete(U, col_idx, axis=1)
            V_test = np.delete(V, col_idx, axis=1)
            W_test = np.delete(W, col_idx, axis=1)
            
            if _validate_fast(U_test, V_test, W_test, n, m, p):
                U = U_test
                V = V_test
                W = W_test
                current_rank -= 1
                improved = True
                break
    
    # Try pairwise column removal: can we remove two columns at once?
    # This handles cases where no single column is removable but a pair is
    if time.time() < deadline - 10 and current_rank > lower_bound:
        improved_pair = True
        while improved_pair and current_rank > lower_bound and time.time() < deadline - 10:
            improved_pair = False
            for c1 in range(current_rank):
                if time.time() >= deadline - 10:
                    break
                for c2 in range(c1 + 1, current_rank):
                    if time.time() >= deadline - 10:
                        break
                    # Remove both columns
                    cols_to_remove = [c1, c2]
                    U_test = np.delete(U, cols_to_remove, axis=1)
                    V_test = np.delete(V, cols_to_remove, axis=1)
                    W_test = np.delete(W, cols_to_remove, axis=1)
                    
                    if _validate_fast(U_test, V_test, W_test, n, m, p):
                        U = U_test
                        V = V_test
                        W = W_test
                        current_rank -= 2
                        improved_pair = True
                        break
                if improved_pair:
                    break
    
    if current_rank < best_rank:
        best_rank = current_rank
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

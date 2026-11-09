"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _tensor_from_decomp(U, V, W, n, m, p):
    """Compute the full multiplication tensor T[i,j,k] from the decomposition."""
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Build using einsum
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    T = np.einsum('imc,jpc,knc->ijk', U_t, V_t, W_t)
    return T

def _target_tensor(n, m, p):
    """The standard matrix multiplication tensor: T[i,j,k] = 1 if i==k, else 0."""
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i == k:
                    T[i, j, k] = 1
    return T

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly matches the target tensor."""
    T = _tensor_from_decomp(U, V, W, n, m, p)
    T_target = _target_tensor(n, m, p)
    return np.allclose(T, T_target, atol=0)

def _try_remove_term(U, V, W, n, m, p, col_idx):
    """Try to remove column col_idx by expressing it as a linear combination of others."""
    rank = U.shape[1]
    if rank <= 1:
        return None
    
    # Keep all columns except col_idx
    keep = [c for c in range(rank) if c != col_idx]
    U_keep = U[:, keep]
    V_keep = V[:, keep]
    W_keep = W[:, keep]
    
    # We need to find coefficients alpha_c such that:
    # U[:, col_idx] = sum_c alpha_c * U_keep[:, c]
    # V[:, col_idx] = sum_c alpha_c * V_keep[:, c]  
    # W[:, col_idx] = sum_c alpha_c * W_keep[:, c]
    
    # This is a system: for each matrix, U[:, col] = U_keep @ alpha
    # Solve least squares for U
    U_col = U[:, col_idx]
    V_col = V[:, col_idx]
    W_col = W[:, col_idx]
    
    # Check if U_col is in span of U_keep
    try:
        alpha_U, residuals_U, rank_U, _ = np.linalg.lstsq(U_keep, U_col, rcond=None)
    except:
        return None
    
    # Check if this alpha also works for V and W
    V_pred = V_keep @ alpha_U
    W_pred = W_keep @ alpha_U
    
    if not np.allclose(V_col, V_pred, atol=1e-9):
        return None
    if not np.allclose(W_col, W_pred, atol=1e-9):
        return None
    
    # Build new decomposition with col_idx removed
    U_new = U_keep
    V_new = V_keep
    W_new = W_keep
    
    return U_new, V_new, W_new

def _try_merge_terms(U, V, W, n, m, p, col_a, col_b):
    """Try to merge two terms by finding if their span can be reduced."""
    rank = U.shape[1]
    if rank <= 2:
        return None
    
    # Check if columns col_a and col_b are linearly dependent in all three matrices
    # i.e., U[:,a] = c*U[:,b] AND V[:,a] = c*V[:,b] AND W[:,a] = c*W[:,b] for some c
    U_a = U[:, col_a]
    U_b = U[:, col_b]
    V_a = V[:, col_a]
    V_b = V[:, col_b]
    W_a = W[:, col_a]
    W_b = W[:, col_b]
    
    # Find c from U
    nonzero = np.nonzero(U_b)[0]
    if len(nonzero) == 0:
        return None
    
    c_candidates = []
    for idx in nonzero:
        if U_b[idx] != 0:
            c_candidates.append(U_a[idx] / U_b[idx])
    
    if not c_candidates:
        return None
    
    # Check consistency
    c = c_candidates[0]
    for cc in c_candidates[1:]:
        if abs(cc - c) > 1e-9:
            return None
    
    # Verify with V and W
    if not np.allclose(V_a, c * V_b, atol=1e-9):
        return None
    if not np.allclose(W_a, c * W_b, atol=1e-9):
        return None
    
    # Remove col_a (keep col_b)
    keep = [i for i in range(rank) if i != col_a]
    U_new = U[:, keep]
    V_new = V[:, keep]
    W_new = W[:, keep]
    
    return U_new, V_new, W_new

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120 second deadline
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try removing terms
    while time.time() < deadline - 10:
        rank = U.shape[1]
        if rank <= 1:
            break
        
        improved = False
        for col in range(rank):
            if time.time() > deadline - 10:
                break
            
            result = _try_remove_term(U, V, W, n, m, p, col)
            if result is not None:
                U, V, W = result
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                improved = True
                break  # Restart scanning from beginning
        
        if not improved:
            break
    
    # Try merging terms
    while time.time() < deadline - 5:
        rank = U.shape[1]
        if rank <= 2:
            break
        
        improved = False
        for a in range(rank):
            if time.time() > deadline - 5:
                break
            for b in range(a + 1, rank):
                if time.time() > deadline - 5:
                    break
                
                result = _try_merge_terms(U, V, W, n, m, p, a, b)
                if result is not None:
                    U, V, W = result
                    if U.shape[1] < best_rank:
                        best_rank = U.shape[1]
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    # Verify final result
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fall back to schoolbook if verification fails
        best_U, best_V, best_W = _build_schoolbook(n, m, p)
        best_rank = best_U.shape[1]
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

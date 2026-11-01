import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _strassen_2x2x2(U, V, W, col, i_offset, j_offset, k_offset, n, m, p):
    """Strassen decomposition for a 2x2x2 block (rank 7)."""
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[i_offset*m + j_offset, col] = 1
    U[(i_offset+1)*m + (j_offset+1), col] = 1
    V[j_offset*p + k_offset, col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = 1
    W[k_offset*n + i_offset, col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = 1
    col += 1
    
    # M2: (A10+A11)B00 -> C10, -C11
    U[(i_offset+1)*m + j_offset, col] = 1
    U[(i_offset+1)*m + (j_offset+1), col] = 1
    V[j_offset*p + k_offset, col] = 1
    W[k_offset*n + (i_offset+1), col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = -1
    col += 1
    
    # M3: A00(B01-B11) -> C01, C11
    U[i_offset*m + j_offset, col] = 1
    V[j_offset*p + (k_offset+1), col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = -1
    W[(k_offset+1)*n + i_offset, col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = 1
    col += 1
    
    # M4: A11(B10-B00) -> C00, C10
    U[(i_offset+1)*m + (j_offset+1), col] = 1
    V[(j_offset+1)*p + k_offset, col] = 1
    V[j_offset*p + k_offset, col] = -1
    W[k_offset*n + i_offset, col] = 1
    W[k_offset*n + (i_offset+1), col] = 1
    col += 1
    
    # M5: (A00+A01)B11 -> -C00, C01
    U[i_offset*m + j_offset, col] = 1
    U[i_offset*m + (j_offset+1), col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = 1
    W[k_offset*n + i_offset, col] = -1
    W[(k_offset+1)*n + i_offset, col] = 1
    col += 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[(i_offset+1)*m + j_offset, col] = 1
    U[i_offset*m + j_offset, col] = -1
    V[j_offset*p + k_offset, col] = 1
    V[j_offset*p + (k_offset+1), col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = 1
    col += 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U[i_offset*m + (j_offset+1), col] = 1
    U[(i_offset+1)*m + (j_offset+1), col] = -1
    V[(j_offset+1)*p + k_offset, col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = 1
    W[k_offset*n + i_offset, col] = 1
    col += 1
    
    return col

def _build_rank28():
    """Build the rank-28 base decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # Block 1: j=0, k=0..4 (2x1x5), rank 10
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
    
    # Block 2: j=1,2, k=0,1 (2x2x2 Strassen), rank 7
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 0, n, m, p)
    
    # Block 3: j=1,2, k=2,3 (2x2x2 Strassen), rank 7
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 2, n, m, p)
    
    # Block 4: j=1,2, k=4 (2x2x1 schoolbook), rank 4
    for i in range(n):
        for j in range(2):
            U[i*m + (1+j), col] = 1
            V[(1+j)*p + 4, col] = 1
            W[4*n + i, col] = 1
            col += 1
    
    assert col == 28, f"Column count mismatch: {col} != 28"
    return U, V, W

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification of tensor decomposition."""
    rank = U.shape[1]
    # Reshape to 3D tensors: U[i,j,r], V[j,k,r], W[k,i,r]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # For each (i,j,k), check sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1
    # Compute via einsum
    result = np.einsum('ijr,jkr,kr->ijk', U3, V3, W3)
    return np.allclose(result, 1.0, atol=1e-10)

def _try_remove_column(U, V, W, col_to_remove, n, m, p):
    """Try removing a single column and verify."""
    U_new = np.delete(U, col_to_remove, axis=1)
    V_new = np.delete(V, col_to_remove, axis=1)
    W_new = np.delete(W, col_to_remove, axis=1)
    
    if _verify_fast(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new
    return None

def _try_remove_columns(U, V, W, cols_to_remove, n, m, p):
    """Try removing multiple columns and verify."""
    U_new = np.delete(U, cols_to_remove, axis=1)
    V_new = np.delete(V, cols_to_remove, axis=1)
    W_new = np.delete(W, cols_to_remove, axis=1)
    
    if _verify_fast(U_new, V_new, W_new, n, m, p):
        return U_new, V_new, W_new
    return None

def _search_rank27(U, V, W, n, m, p, deadline):
    """Search for rank-27 decomposition from rank-28 base."""
    current_rank = U.shape[1]
    
    # Phase 1: Try removing single columns
    for col_to_remove in range(current_rank):
        if time.time() > deadline - 30:
            return None
        result = _try_remove_column(U, V, W, col_to_remove, n, m, p)
        if result is not None:
            return result
    
    # Phase 2: Try removing pairs of columns (more expensive but thorough)
    for col1 in range(current_rank):
        for col2 in range(col1 + 1, current_rank):
            if time.time() > deadline - 20:
                return None
            result = _try_remove_columns(U, V, W, [col1, col2], n, m, p)
            if result is not None:
                return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for safety
    
    # Build rank-28 decomposition
    U, V, W = _build_rank28()
    
    # Verify base decomposition
    if not _verify_fast(U, V, W, n, m, p):
        raise ValueError("Base decomposition failed verification")
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Search for rank-27
    if time.time() < deadline - 35:
        reduced = _search_rank27(U, V, W, n, m, p, deadline)
        if reduced is not None:
            best_U, best_V, best_W = reduced
            best_rank = best_U.shape[1]
    
    # Final verification
    if not _verify_fast(best_U, best_V, best_W, n, m, p):
        # Fall back to rank 28
        U, V, W = _build_rank28()
        best_U, best_V, best_W = U, V, W
        best_rank = 28
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

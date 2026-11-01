import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]
SEED_RANK = 28

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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # and check T == 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

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
    
    # Block 3: j=1,2, k=2,3,4 (2x2x3), rank 11
    # First 2x2x2 Strassen block for k=2,3
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 2, n, m, p)
    
    # Schoolbook for k=4 (4 terms)
    for i in range(2):
        for j in range(2):
            U[(i)*m + (1+j), col] = 1
            V[(1+j)*p + 4, col] = 1
            W[4*n + i, col] = 1
            col += 1
    
    assert col == 28, f"Column count mismatch: {col} != 28"
    return U, V, W

def _try_rank_reduction(U, V, W, target_rank, deadline, n, m, p):
    """
    Try to reduce rank by checking if any column can be removed.
    Returns (U, V, W) with reduced rank if successful, else None.
    """
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W
    
    # Try removing one column at a time
    for col_to_remove in range(current_rank):
        if time.time() > deadline - 10:
            return None
        
        # Create copies with column removed
        U_new = np.delete(U, col_to_remove, axis=1)
        V_new = np.delete(V, col_to_remove, axis=1)
        W_new = np.delete(W, col_to_remove, axis=1)
        
        if _verify_decomposition(U_new, V_new, W_new, n, m, p):
            return U_new, V_new, W_new
    
    return None

def _try_pairwise_reduction(U, V, W, target_rank, deadline, n, m, p):
    """
    Try to reduce rank by removing pairs of columns.
    Returns (U, V, W) with reduced rank if successful, else None.
    """
    current_rank = U.shape[1]
    
    if current_rank <= target_rank:
        return U, V, W
    
    for col1 in range(current_rank):
        if time.time() > deadline - 15:
            return None
        for col2 in range(col1 + 1, current_rank):
            if time.time() > deadline - 10:
                return None
            
            U_new = np.delete(U, [col1, col2], axis=1)
            V_new = np.delete(V, [col1, col2], axis=1)
            W_new = np.delete(W, [col1, col2], axis=1)
            
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    # Build rank-28 decomposition
    U, V, W = _build_rank28()
    
    # Verify base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Base decomposition failed verification")
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by removing single columns
    if time.time() < deadline - 20:
        reduced = _try_rank_reduction(U, V, W, best_rank - 1, deadline, n, m, p)
        if reduced is not None:
            best_U, best_V, best_W = reduced
            best_rank = best_U.shape[1]
    
    # Try pairwise column removal
    if time.time() < deadline - 15 and best_rank > 26:
        reduced = _try_pairwise_reduction(best_U, best_V, best_W, best_rank - 2, deadline, n, m, p)
        if reduced is not None:
            best_U, best_V, best_W = reduced
            best_rank = best_U.shape[1]
    
    # Try further single-column reductions after pairwise
    if time.time() < deadline - 10 and best_rank > 25:
        reduced = _try_rank_reduction(best_U, best_V, best_W, best_rank - 1, deadline, n, m, p)
        if reduced is not None:
            best_U, best_V, best_W = reduced
            best_rank = best_U.shape[1]
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fall back to rank 28
        U, V, W = _build_rank28()
        best_U, best_V, best_W = U, V, W
        best_rank = 28
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

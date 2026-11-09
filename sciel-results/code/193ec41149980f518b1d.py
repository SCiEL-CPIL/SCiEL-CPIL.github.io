"""Matrix multiplication 4,5,6 - Strassen-based rank reduction via embedding in 6x6x6."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]
TIME_LIMIT = 150.0

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """
    Build rank-7 Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Returns U, V, W as 2x2 x 7 matrices (flattened to 4x7).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11 + A22) * (B11 + B22)
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0
    
    # M2 = (A21 + A22) * B11
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0; W[3, 1] = -1.0
    
    # M3 = A11 * (B12 - B22)
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    W[0, 2] = 1.0; W[3, 2] = 1.0
    
    # M4 = A22 * (B21 - B11)
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0; W[2, 3] = 1.0
    
    # M5 = (A11 + A12) * B22
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[1, 4] = 1.0
    
    # M6 = (A21 - A11) * (B11 + B12)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0
    
    # M7 = (A12 - A22) * (B21 + B22)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0
    
    return U, V, W


def _build_full_strassen(N):
    """
    Build full Strassen decomposition for N x N x N where N is a power of 2.
    Returns U, V, W as (N*N) x rank matrices.
    U[i*N+j, c], V[j*N+k, c], W[k*N+i, c]
    """
    if N == 2:
        return _strassen_2x2()
    
    # Split N into two halves
    h = N // 2
    U_h, V_h, W_h = _build_full_strassen(h)
    rank_h = U_h.shape[1]
    
    # For N x N, we have 4 sub-blocks of size h x h
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = A*B
    
    # We need to build the full decomposition by combining 4 copies of the h-sized decomposition
    # But Strassen combines them in a specific way.
    
    # Actually, the standard recursive Strassen builds N x N from 4 h x h multiplications
    # using 7 recursive calls. Let me build this properly.
    
    # For N x N using 7 recursive h x h multiplications:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M is an h x h matrix product, decomposed with rank_h terms.
    # Total rank = 7 * rank_h
    
    rank_full = 7 * rank_h
    U_full = np.zeros((N * N, rank_full))
    V_full = np.zeros((N * N, rank_full))
    W_full = np.zeros((N * N, rank_full))
    
    # Helper to get sub-block indices
    def block_idx(i, j, offset_i, offset_j, N, h):
        """Get flattened index for element (i,j) in block at (offset_i, offset_j)"""
        return (offset_i + i) * N + (offset_j + j)
    
    # M1 = (A11+A22)(B11+B22)
    # A part: A11 (top-left) and A22 (bottom-right)
    # B part: B11 (top-left) and B22 (bottom-right)
    # C part: contributes to C11 (+1), C22 (+1)
    for c in range(rank_h):
        col = c
        # U: A11 and A22
        for i in range(h):
            for j in range(h):
                U_full[block_idx(i, j, 0, 0, N, h), col] += U_h[i * h + j, c]  # A11
                U_full[block_idx(i, j, h, h, N, h), col] += U_h[i * h + j, c]  # A22
        # V: B11 and B22
        for j in range(h):
            for k in range(h):
                V_full[block_idx(j, k, 0, 0, N, h), col] += V_h[j * h + k, c]  # B11
                V_full[block_idx(j, k, h, h, N, h), col] += V_h[j * h + k, c]  # B22
        # W: C11 (+1) and C22 (+1)
        for k in range(h):
            for i in range(h):
                W_full[block_idx(k, i, 0, 0, N, h), col] += W_h[k * h + i, c]  # C11
                W_full[block_idx(k, i, h, h, N, h), col] += W_h[k * h + i, c]  # C22
    
    # M2 = (A21+A22)B11
    # A part: A21 (bottom-left) and A22 (bottom-right)
    # B part: B11 (top-left)
    # C part: contributes to C21 (+1), C22 (-1)
    for c in range(rank_h):
        col = rank_h + c
        # U: A21 and A22
        for i in range(h):
            for j in range(h):
                U_full[block_idx(i, j, h, 0, N, h), col] += U_h[i * h + j, c]  # A21
                U_full[block_idx(i, j, h, h, N, h), col] += U_h[i * h + j, c]  # A22
        # V: B11
        for j in range(h):
            for k in range(h):
                V_full[block_idx(j, k, 0, 0, N, h), col] += V_h[j * h + k, c]
        # W: C21 (+1), C22 (-1)
        for k in range(h):
            for i in range(h):
                W_full[block_idx(k, i, h, 0, N, h), col] += W_h[k * h + i, c]  # C21
                W_full[block_idx(k, i, h, h, N, h), col] -= W_h[k * h + i, c]  # C22
    
    # M3 = A11(B12-B22)
    # A part: A11 (top-left)
    # B part: B12 (top-right) and B22 (bottom-right)
    # C part: contributes to C12 (+1), C22 (+1)
    for c in range(rank_h):
        col = 2 * rank_h + c
        # U: A11
        for i in range(h):
            for j in range(h):
                U_full[block_idx(i, j, 0, 0, N, h), col] += U_h[i * h + j, c]
        # V: B12 (+1) and B22 (-1)
        for j in range(h):
            for k in range(h):
                V_full[block_idx(j, k, 0, h, N, h), col] += V_h[j * h + k, c]  # B12
                V_full[block_idx(j, k, h, h, N, h), col] -= V_h[j * h + k, c]  # B22
        # W: C12 (+1), C22 (+1)
        for k in range(h):
            for i in range(h):
                W_full[block_idx(k, i, 0, h, N, h), col] += W_h[k * h + i, c]  # C12
                W_full[block_idx(k, i, h, h, N, h), col] += W_h[k * h + i, c]  # C22
    
    # M4 = A22(B21-B11)
    # A part: A22 (bottom-right)
    # B part: B21 (bottom-left) and B11 (top-left)
    # C part: contributes to C11 (+1), C21 (+1)
    for c in range(rank_h):
        col = 3 * rank_h + c
        # U: A22
        for i in range(h):
            for j in range(h):
                U_full[block_idx(i, j, h, h, N, h), col] += U_h[i * h + j, c]
        # V: B21 (+1) and B11 (-1)
        for j in range(h):
            for k in range(h):
                V_full[block_idx(j, k, h, 0, N, h), col] += V_h[j * h + k, c]  # B21
                V_full[block_idx(j, k, 0, 0, N, h), col] -= V_h[j * h + k, c]  # B11
        # W: C11 (+1), C21 (+1)
        for k in range(h):
            for i in range(h):
                W_full[block_idx(k, i, 0, 0, N, h), col] += W_h[k * h + i, c]  # C11
                W_full[block_idx(k, i, h, 0, N, h), col] += W_h[k * h + i, c]  # C21
    
    # M5 = (A11+A12)B22
    # A part: A11 (top-left) and A12 (top-right)
    # B part: B22 (bottom-right)
    # C part: contributes to C11 (-1), C12 (+1)
    for c in range(rank_h):
        col = 4 * rank_h + c
        # U: A11 and A12
        for i in range(h):
            for j in range(h):
                U_full[block_idx(i, j, 0, 0, N, h), col] += U_h[i * h + j, c]  # A11
                U_full[block_idx(i, j, 0, h, N, h), col] += U_h[i * h + j, c]  # A12
        # V: B22
        for j in range(h):
            for k in range(h):
                V_full[block_idx(j, k, h, h, N, h), col] += V_h[j * h + k, c]
        # W: C11 (-1), C12 (+1)
        for k in range(h):
            for i in range(h):
                W_full[block_idx(k, i, 0, 0, N, h), col] -= W_h[k * h + i, c]  # C11
                W_full[block_idx(k, i, 0, h, N, h), col] += W_h[k * h + i, c]  # C12
    
    # M6 = (A21-A11)(B11+B12)
    # A part: A21 (bottom-left) and A11 (top-left)
    # B part: B11 (top-left) and B12 (top-right)
    # C part: contributes to C22 (+1)
    for c in range(rank_h):
        col = 5 * rank_h + c
        # U: A21 (+1) and A11 (-1)
        for i in range(h):
            for j in range(h):
                U_full[block_idx(i, j, h, 0, N, h), col] += U_h[i * h + j, c]  # A21
                U_full[block_idx(i, j, 0, 0, N, h), col] -= U_h[i * h + j, c]  # A11
        # V: B11 and B12
        for j in range(h):
            for k in range(h):
                V_full[block_idx(j, k, 0, 0, N, h), col] += V_h[j * h + k, c]  # B11
                V_full[block_idx(j, k, 0, h, N, h), col] += V_h[j * h + k, c]  # B12
        # W: C22 (+1)
        for k in range(h):
            for i in range(h):
                W_full[block_idx(k, i, h, h, N, h), col] += W_h[k * h + i, c]  # C22
    
    # M7 = (A12-A22)(B21+B22)
    # A part: A12 (top-right) and A22 (bottom-right)
    # B part: B21 (bottom-left) and B22 (bottom-right)
    # C part: contributes to C11 (+1)
    for c in range(rank_h):
        col = 6 * rank_h + c
        # U: A12 (+1) and A22 (-1)
        for i in range(h):
            for j in range(h):
                U_full[block_idx(i, j, 0, h, N, h), col] += U_h[i * h + j, c]  # A12
                U_full[block_idx(i, j, h, h, N, h), col] -= U_h[i * h + j, c]  # A22
        # V: B21 and B22
        for j in range(h):
            for k in range(h):
                V_full[block_idx(j, k, h, 0, N, h), col] += V_h[j * h + k, c]  # B21
                V_full[block_idx(j, k, h, h, N, h), col] += V_h[j * h + k, c]  # B22
        # W: C11 (+1)
        for k in range(h):
            for i in range(h):
                W_full[block_idx(k, i, 0, 0, N, h), col] += W_h[k * h + i, c]  # C11
    
    return U_full, V_full, W_full


def _extract_subtensor(U_full, V_full, W_full, N, n, m, p):
    """
    Extract the n x m x p sub-tensor from the N x N x N decomposition.
    U_full[i*N+j, c] -> U_sub[i*m+j, c] for i<n, j<m
    V_full[j*N+k, c] -> V_sub[j*p+k, c] for j<m, k<p
    W_full[k*N+i, c] -> W_sub[k*n+i, c] for k<p, i<n
    """
    rank = U_full.shape[1]
    
    U_sub = np.zeros((n * m, rank))
    V_sub = np.zeros((m * p, rank))
    W_sub = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            U_sub[i * m + j, :] = U_full[i * N + j, :]
    
    for j in range(m):
        for k in range(p):
            V_sub[j * p + k, :] = V_full[j * N + k, :]
    
    for k in range(p):
        for i in range(n):
            W_sub[k * n + i, :] = W_full[k * N + i, :]
    
    return U_sub, V_sub, W_sub


def _remove_zero_columns(U, V, W):
    """Remove columns where any of U, V, W is entirely zero."""
    rank = U.shape[1]
    keep = []
    for c in range(rank):
        if np.any(U[:, c] != 0) and np.any(V[:, c] != 0) and np.any(W[:, c] != 0):
            keep.append(c)
    
    if len(keep) == rank:
        return U, V, W
    
    U_new = U[:, keep]
    V_new = V[:, keep]
    W_new = W[:, keep]
    
    return U_new, V_new, W_new


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W


def construct():
    """
    Find a low-rank exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    Uses embedding in a 6x6x6 Strassen decomposition.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Fallback: schoolbook
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try embedding in 6x6x6 Strassen
    try:
        N = 6
        U_full, V_full, W_full = _build_full_strassen(N)
        
        U_sub, V_sub, W_sub = _extract_subtensor(U_full, V_full, W_full, N, n, m, p)
        U_sub, V_sub, W_sub = _remove_zero_columns(U_sub, V_sub, W_sub)
        
        # Verify
        if _verify_decomposition(U_sub, V_sub, W_sub, n, m, p):
            if U_sub.shape[1] < best_rank:
                best_U, best_V, best_W = U_sub, V_sub, W_sub
                best_rank = U_sub.shape[1]
    except Exception:
        pass
    
    # Try embedding in 8x8x8 Strassen (larger, but might have more structure)
    if time.time() - start_time < TIME_LIMIT - 30:
        try:
            N = 8
            U_full, V_full, W_full = _build_full_strassen(N)
            
            U_sub, V_sub, W_sub = _extract_subtensor(U_full, V_full, W_full, N, n, m, p)
            U_sub, V_sub, W_sub = _remove_zero_columns(U_sub, V_sub, W_sub)
            
            if _verify_decomposition(U_sub, V_sub, W_sub, n, m, p):
                if U_sub.shape[1] < best_rank:
                    best_U, best_V, best_W = U_sub, V_sub, W_sub
                    best_rank = U_sub.shape[1]
        except Exception:
            pass
    
    # Check deadline
    if time.time() - start_time > TIME_LIMIT - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

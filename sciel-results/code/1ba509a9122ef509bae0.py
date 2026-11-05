import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

def _matmul_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i, j, k] = delta(i,k)*delta(j,i)
    under the convention C[i, k] = sum_j A[i, j] B[j, k]."""
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            T[i, j, i] = 1
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == T[i,j,k]."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', uc, vc, wc)
    expected = _matmul_tensor(n, m, p).astype(np.float64)
    return np.allclose(T, expected, atol=tol, rtol=0)

def _schoolbook_block(n, m, p, i0, j0, k0):
    """Schoolbook decomposition for a sub-block A[i0:i0+n, j0:j0+m] * B[j0:j0+m, k0:k0+p].
    Returns (U, V, W) where U has shape (n*m, n*m*p) but we only fill the sub-block rows/cols.
    Actually returns local matrices of shapes (n*m, n*m*p), (m*p, n*m*p), (p*n, n*m*p)."""
    r = n * m * p
    U = np.zeros((n * m, r), dtype=np.int64)
    V = np.zeros((m * p, r), dtype=np.int64)
    W = np.zeros((p * n, r), dtype=np.int64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _strassen_222():
    """Strassen decomposition for 2x2 * 2x2, rank 7.
    Returns U (4,7), V (4,7), W (4,7).
    U index: i*2+j. V index: j*2+k. W index: k*2+i.
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    # c0 = (A00+A11)(B00+B11) -> C00 += c0, C11 += c0
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # c1 = (A10+A11)B00 -> C10 += c1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    # c2 = A00(B01-B11) -> C01 += c2
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    # c3 = A11(B10-B00) -> C00 += c3, C10 += c3
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    # c4 = (A00+A01)B11 -> C00 -= c4, C01 += c4
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    # c5 = (A10-A00)(B00+B01) -> C11 += c5
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # c6 = (A01-A11)(B10+B11) -> C00 += c6, C11 -= c6
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = -1
    return U, V, W

def _place_block2(U_full, V_full, W_full, U_blk, V_blk, W_blk,
                  n_full, m_full, p_full,
                  n_b, m_b, p_b, i0, j0, k0, c0):
    """Place block at (i0,j0,k0) with dims (n_b,m_b,p_b) starting at column c0."""
    for i in range(n_b):
        for j in range(m_b):
            for c in range(U_blk.shape[1]):
                U_full[(i0 + i) * m_full + (j0 + j), c0 + c] += U_blk[i * m_b + j, c]
    for j in range(m_b):
        for k in range(p_b):
            for c in range(V_blk.shape[1]):
                V_full[(j0 + j) * p_full + (k0 + k), c0 + c] += V_blk[j * p_b + k, c]
    for k in range(p_b):
        for i in range(n_b):
            for c in range(W_blk.shape[1]):
                W_full[(k0 + k) * n_full + (i0 + i), c0 + c] += W_blk[k * n_b + i, c]

def _decompose_332():
    """Decompose 3x3 * 3x2 into rank 17.
    Split: 2x2x2 (Strassen, rank 7) + 2x1x2 (rank 4) + 1x2x2 (rank 4) + 1x1x2 (rank 2).
    n=3, m=3, p=2.
    i in {0,1,2}, j in {0,1,2}, k in {0,1}.
    """
    n, m, p = 3, 3, 2
    total_rank = 17
    U = np.zeros((n * m, total_rank), dtype=np.int64)
    V = np.zeros((m * p, total_rank), dtype=np.int64)
    W = np.zeros((p * n, total_rank), dtype=np.int64)
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2 Strassen
    U1, V1, W1 = _strassen_222()
    _place_block2(U, V, W, U1, V1, W1, n, m, p, 2, 2, 2, 0, 0, 0, 0)
    
    # Block 2: i in {0,1}, j=2, k in {0,1} -> 2x1x2 schoolbook (rank 4)
    U2, V2, W2 = _schoolbook_block(2, 1, 2, 0, 0, 0)
    _place_block2(U, V, W, U2, V2, W2, n, m, p, 2, 1, 2, 0, 2, 0, 7)
    
    # Block 3: i=2, j in {0,1}, k in {0,1} -> 1x2x2 schoolbook (rank 4)
    U3, V3, W3 = _schoolbook_block(1, 2, 2, 0, 0, 0)
    _place_block2(U, V, W, U3, V3, W3, n, m, p, 1, 2, 2, 2, 0, 0, 11)
    
    # Block 4: i=2, j=2, k in {0,1} -> 1x1x2 schoolbook (rank 2)
    U4, V4, W4 = _schoolbook_block(1, 1, 2, 0, 0, 0)
    _place_block2(U, V, W, U4, V4, W4, n, m, p, 1, 1, 2, 2, 2, 0, 15)
    
    return U, V, W

def _decompose_334():
    """Decompose 3x3 * 3x4 into rank 34 by combining two 3x3*3x2 decompositions."""
    n, m, p = 3, 3, 4
    total_rank = 34
    U = np.zeros((n * m, total_rank), dtype=np.int64)
    V = np.zeros((m * p, total_rank), dtype=np.int64)
    W = np.zeros((p * n, total_rank), dtype=np.int64)
    
    # First 3x3*3x2 for k in {0,1}
    U1, V1, W1 = _decompose_332()
    _place_block2(U, V, W, U1, V1, W1, n, m, p, 3, 3, 2, 0, 0, 0, 0)
    
    # Second 3x3*3x2 for k in {2,3}
    U2, V2, W2 = _decompose_332()
    _place_block2(U, V, W, U2, V2, W2, n, m, p, 3, 3, 2, 0, 0, 2, 17)
    
    return U, V, W

def _decompose_338():
    """Decompose 3x3 * 3x8 into rank 68 by combining two 3x3*3x4 decompositions."""
    n, m, p = 3, 3, 8
    total_rank = 68
    U = np.zeros((n * m, total_rank), dtype=np.int64)
    V = np.zeros((m * p, total_rank), dtype=np.int64)
    W = np.zeros((p * n, total_rank), dtype=np.int64)
    
    # First 3x3*3x4 for k in {0,1,2,3}
    U1, V1, W1 = _decompose_334()
    _place_block2(U, V, W, U1, V1, W1, n, m, p, 3, 3, 4, 0, 0, 0, 0)
    
    # Second 3x3*3x4 for k in {4,5,6,7}
    U2, V2, W2 = _decompose_334()
    _place_block2(U, V, W, U2, V2, W2, n, m, p, 3, 3, 4, 0, 0, 4, 34)
    
    return U, V, W

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition, rank n*m*p."""
    r = n * m * p
    U = np.zeros((n * m, r), dtype=np.int64)
    V = np.zeros((m * p, r), dtype=np.int64)
    W = np.zeros((p * n, r), dtype=np.int64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, time_budget=5.0):
    """
    Attempt to reduce the rank of the decomposition by merging terms or finding a lower rank representation.
    This is a placeholder for a more sophisticated search if time permits, but for now we just return the input
    if it's already at or below target, or try a simple greedy merge if possible.
    Given the complexity and time limits, we will primarily rely on the structured decompositions.
    """
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return U, V, W
    
    # A simple reduction strategy: if we have redundant terms, remove them.
    # However, the structured decompositions (Strassen based) are already optimized for their sub-blocks.
    # Merging across blocks is non-trivial and risky.
    # For now, we just return the current decomposition.
    return U, V, W

def construct():
    n, m, p = 3, 3, 8
    
    # Start time for this specific call to respect the 120s limit
    call_start = time.time()
    
    # Try the rank-68 decomposition first
    try:
        U, V, W = _decompose_338()
        if _verify(U, V, W, n, m, p):
            # Check if we can do better or if we are close to the limit
            # If we have time, we could try to reduce further, but 68 is already a good improvement over 72.
            # The theoretical lower bound is higher, but finding it is hard.
            # We return this valid decomposition.
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook rank 72
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

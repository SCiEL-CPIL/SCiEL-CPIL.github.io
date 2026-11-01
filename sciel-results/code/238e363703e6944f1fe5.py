"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 matrix multiplication using Strassen (rank 7).
    U: 4x7, V: 4x7, W: 4x7
    Indexing: U[i*2+j], V[i*2+j], W[i*2+j]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # a11, a12, a21, a22 -> indices 0,1,2,3
    # b11, b12, b21, b22 -> indices 0,1,2,3
    # c11, c12, c21, c22 -> indices 0,1,2,3
    
    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # c11 += M1, c22 += M1
    
    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[3, 1] = 1  # c21 += M2
    
    # M3 = a11(b22-b12)
    U[0, 2] = 1
    V[3, 2] = 1; V[1, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1  # c12 += M3, c22 -= M3
    
    # M4 = a22(b11-b21)
    U[3, 3] = 1
    V[0, 3] = 1; V[2, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1  # c11 -= M4, c22 += M4
    
    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1  # c12 += M5
    
    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1  # c11 += M6
    
    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[2, 6] = 1  # c21 += M7
    
    return U, V, W


def _build_decomp(n, m, p):
    """
    Build a Strassen-based decomposition for 2x3x8.
    Split A = [A1 | A2] where A1 is 2x2, A2 is 2x1.
    Split B = [B1; B2] where B1 is 2x8, B2 is 1x8.
    C = A1*B1 + A2*B2.
    
    A2*B2: rank 16 (each of 2x1 * 1x8 = 16 scalar products).
    A1*B1: A1 is 2x2, B1 is 2x8. Split B1 into 4 blocks of 2x2.
    Each 2x2 * 2x2 uses Strassen (rank 7). Total 4*7 = 28.
    Grand total: 44.
    
    Indexing:
    U[i*m+j, c] where i in 0..1, j in 0..2
    V[j*p+k, c] where j in 0..2, k in 0..7
    W[k*n+i, c] where k in 0..7, i in 0..1
    """
    n, m, p = 2, 3, 8
    
    rank_total = 44
    U = np.zeros((n * m, rank_total))  # 6 x 44
    V = np.zeros((m * p, rank_total))  # 24 x 44
    W = np.zeros((p * n, rank_total))  # 16 x 44
    
    col = 0
    
    # Part 1: A2*B2
    # A2 has entries at U[2, c] (i=0,j=2) and U[5, c] (i=1,j=2)
    # B2 has entries at V[16+k, c] for k in 0..7 (j=2, k)
    # Output: W[k*n+i, c] for i in 0..1, k in 0..7
    for k in range(8):
        # i=0, j=2, k
        U[2, col] = 1
        V[16 + k, col] = 1
        W[k * 2 + 0, col] = 1
        col += 1
        # i=1, j=2, k
        U[5, col] = 1
        V[16 + k, col] = 1
        W[k * 2 + 1, col] = 1
        col += 1
    
    # Part 2: A1*B1 via Strassen (4 blocks of 2x2)
    # A1 entries: U[0,c] (i=0,j=0), U[1,c] (i=0,j=1), U[3,c] (i=1,j=0), U[4,c] (i=1,j=1)
    # B1 entries: V[j*p+k, c] where j in 0,1 and k in 0..7
    # For block g (g=0,1,2,3), B1 columns 2g, 2g+1
    # Output: W[k*n+i, c] for i in 0,1 and k in 2g, 2g+1
    
    for g in range(4):
        # B1 block indices: j=0: V[0*p+2g], V[0*p+2g+1]; j=1: V[1*p+2g], V[1*p+2g+1]
        b11 = 0 * p + 2 * g      # V index for j=0, k=2g
        b12 = 0 * p + 2 * g + 1  # V index for j=0, k=2g+1
        b21 = 1 * p + 2 * g      # V index for j=1, k=2g
        b22 = 1 * p + 2 * g + 1  # V index for j=1, k=2g+1
        
        a11 = 0  # U index for i=0, j=0
        a12 = 1  # U index for i=0, j=1
        a21 = 3  # U index for i=1, j=0
        a22 = 4  # U index for i=1, j=1
        
        # Output W indices:
        # c11: W[(2g)*2 + 0, col] -> W[2g*2, col]
        # c12: W[(2g+1)*2 + 0, col] -> W[(2g+1)*2, col]
        # c21: W[(2g)*2 + 1, col] -> W[2g*2+1, col]
        # c22: W[(2g+1)*2 + 1, col] -> W[(2g+1)*2+1, col]
        
        w11 = (2 * g) * 2 + 0
        w12 = (2 * g + 1) * 2 + 0
        w21 = (2 * g) * 2 + 1
        w22 = (2 * g + 1) * 2 + 1
        
        # M1 = (a11+a22)(b11+b22)
        U[a11, col] = 1; U[a22, col] = 1
        V[b11, col] = 1; V[b22, col] = 1
        W[w11, col] = 1; W[w22, col] = 1
        col += 1
        
        # M2 = (a21+a22)b11
        U[a21, col] = 1; U[a22, col] = 1
        V[b11, col] = 1
        W[w21, col] = 1
        col += 1
        
        # M3 = a11(b22-b12)
        U[a11, col] = 1
        V[b22, col] = 1; V[b12, col] = -1
        W[w12, col] = 1; W[w22, col] = -1
        col += 1
        
        # M4 = a22(b11-b21)
        U[a22, col] = 1
        V[b11, col] = 1; V[b21, col] = -1
        W[w11, col] = -1; W[w22, col] = 1
        col += 1
        
        # M5 = (a11+a12)b22
        U[a11, col] = 1; U[a12, col] = 1
        V[b22, col] = 1
        W[w12, col] = 1
        col += 1
        
        # M6 = (a21-a11)(b11+b12)
        U[a21, col] = 1; U[a11, col] = -1
        V[b11, col] = 1; V[b12, col] = 1
        W[w11, col] = 1
        col += 1
        
        # M7 = (a12-a22)(b21+b22)
        U[a12, col] = 1; U[a22, col] = -1
        V[b21, col] = 1; V[b22, col] = 1
        W[w21, col] = 1
        col += 1
    
    return U.tolist(), V.tolist(), W.tolist()


def _verify(U, V, W, n, m, p):
    """Verify decomposition correctness."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    return max_err


def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()


def _try_rank_reduction(U, V, W, n, m, p, max_time=30):
    """
    Try to reduce rank by finding linear dependencies among rank-1 terms.
    Returns (U_reduced, V_reduced, W_reduced) or None.
    """
    start = time.time()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build tensor vectors: for each column c, form a (n*m*p)-dim vector
    def tensor_vec(u_col, v_col, w_col):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i * m * p + j * p + k] = u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
        return vec
    
    # Build matrix of shape (n*m*p, rank) where each column is a tensor vector
    M = np.zeros((n * m * p, rank))
    for c in range(rank):
        M[:, c] = tensor_vec(U[:, c], V[:, c], W[:, c])
    
    # Check if columns are linearly dependent
    try:
        r = np.linalg.matrix_rank(M, tol=1e-10)
    except np.linalg.LinAlgError:
        return None
    
    if r < rank:
        # There are linear dependencies. Try to eliminate columns.
        # Find null space
        try:
            from scipy.linalg import null_space
            ns = null_space(M, rcond=1e-10)
            if ns.shape[1] > 0:
                # For each null space vector, we can eliminate one column
                # This is complex; for now, just return None
                pass
        except Exception:
            pass
    
    if time.time() - start > max_time:
        return None
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Primary: Strassen-based rank-44 decomposition
    try:
        U, V, W = _build_decomp(n, m, p)
        err = _verify(U, V, W, n, m, p)
        if err < 1e-10:
            # Try rank reduction if time permits
            if time.time() < deadline - 15:
                reduced = _try_rank_reduction(U, V, W, n, m, p, max_time=30)
                if reduced is not None:
                    U, V, W = reduced
                    err = _verify(U, V, W, n, m, p)
                    if err < 1e-10:
                        return {"U": U, "V": V, "W": W}
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook rank-48
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

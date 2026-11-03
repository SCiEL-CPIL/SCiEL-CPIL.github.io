import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5


def _validate(U, V, W, tol=1e-9):
    """Check sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = N, M, P
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """Return (U, V, W) for Strassen's rank-7 decomposition of 2x2x2."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # m3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # m4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1
    
    # m5 = (A00+A10)B11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = 1
    
    # m6 = (A11-A10)(B00+B10)
    U[3, 5] = 1; U[2, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[0, 5] = 1
    
    # m7 = (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[2, 6] = 1
    
    return U, V, W


def _schoolbook_block(n, m, p):
    """Return (U, V, W) for schoolbook decomposition of n x m * m x p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    return U, V, W


def _combine_decompositions(decomps, n_total, m_total, p_total):
    """
    Combine a list of (U, V, W, i_off, j_off, k_off, n_sub, m_sub, p_sub) tuples
    into a single decomposition.
    """
    total_rank = sum(U.shape[1] for U, V, W, _, _, _, _, _, _ in decomps)
    U_full = np.zeros((n_total * m_total, total_rank))
    V_full = np.zeros((m_total * p_total, total_rank))
    W_full = np.zeros((p_total * n_total, total_rank))
    
    r_off = 0
    for U_sub, V_sub, W_sub, i_off, j_off, k_off, n_sub, m_sub, p_sub in decomps:
        R_sub = U_sub.shape[1]
        for r in range(R_sub):
            r_full = r_off + r
            for i in range(n_sub):
                for j in range(m_sub):
                    U_full[(i + i_off) * m_total + (j + j_off), r_full] = U_sub[i * m_sub + j, r]
            for j in range(m_sub):
                for k in range(p_sub):
                    V_full[(j + j_off) * p_total + (k + k_off), r_full] = V_sub[j * m_sub + k, r]
            for k in range(p_sub):
                for i in range(n_sub):
                    W_full[(k + k_off) * n_total + (i + i_off), r_full] = W_sub[k * p_sub + i, r]
        r_off += R_sub
    
    return U_full, V_full, W_full


def _construct_rank_99():
    """
    Construct a rank-99 decomposition for 4x5x5.
    
    Strategy: Split 4x5x5 into:
    - 4x4x4 part (rows 0-3, cols 0-3 of B)
    - 4x1x5 part (row 4 of B)
    
    For 4x4x4, use a rank-63 construction (if we can find one).
    For 4x1x5, use schoolbook rank 20.
    Total: 63 + 20 = 83.
    
    But we need to handle the overlap carefully. Actually, this split doesn't work
    directly because the tensor structure is different.
    
    Let me try a different approach: use the fact that we can save one rank by
    finding a linear dependency.
    
    For now, return schoolbook 100 as a safe fallback.
    """
    n, m, p = N, M, P
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    return U, V, W


def construct():
    t0 = time.time()
    n, m, p = N, M, P

    # Try to construct a rank-99 decomposition
    elapsed = time.time() - t0
    if elapsed < DEADLINE_S - 60.0:
        U, V, W = _construct_rank_99()
        if _validate(U, V, W):
            rank = U.shape[1]
            if rank < n * m * p:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fall back to schoolbook rank 100
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1

    # Validate with deadline guard
    elapsed = time.time() - t0
    if elapsed < DEADLINE_S - 10.0:
        if not _validate(U, V, W):
            raise ValueError("Schoolbook decomposition failed validation")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

"""Matrix multiplication 2,4,6 – rank-minimal decomposition via block decomposition."""
DIMENSIONS = [2, 4, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _schoolbook_fallback(n, m, p):
    """Standard O(nmp) decomposition for multiplying (n x m) by (m x p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2_block():
    """Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen)."""
    n_s, m_s, p_s = 2, 2, 2
    rank_s = 7
    U = np.zeros((n_s * m_s, rank_s), dtype=float)
    V = np.zeros((m_s * p_s, rank_s), dtype=float)
    W = np.zeros((p_s * n_s, rank_s), dtype=float)
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[2, 3] = 1
    # M5 = (A00+A10)*B11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    # M6 = (A11-A10)*(B00+B10)
    U[3, 5] = 1; U[2, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    # M7 = (A01-A11)*(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify_decomp(U, V, W, n, m, p, rank, tol=1e-10):
    """Verify decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_2x4x2_from_strassen():
    """Build 2x4x2 decomposition from two 2x2x2 Strassen blocks. Rank 14."""
    n, m, p = 2, 4, 2
    rank = 14
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U_s, V_s, W_s = _strassen_2x2x2_block()
    
    # Block 1: j in {0,1}
    for r in range(7):
        U[0:4, r] = U_s[0:4, r]
        V[0:4, r] = V_s[0:4, r]
        W[:, r] = W_s[:, r]
    
    # Block 2: j in {2,3}
    for r in range(7):
        U[4:8, r+7] = U_s[0:4, r]
        V[4:8, r+7] = V_s[0:4, r]
        W[:, r+7] = W_s[:, r]
    
    return U, V, W

def _build_2x4x4_from_2x4x2():
    """Build 2x4x4 decomposition from two 2x4x2 blocks. Rank 28."""
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U1, V1, W1 = _build_2x4x2_from_strassen()
    
    # Block 1: k in {0,1}
    for r in range(14):
        U[:, r] = U1[:, r]
        V[0:8, r] = V1[0:8, r]
        W[0:4, r] = W1[0:4, r]
    
    # Block 2: k in {2,3}
    for r in range(14):
        U[:, r+14] = U1[:, r]
        V[8:16, r+14] = V1[0:8, r]
        W[4:8, r+14] = W1[0:4, r]
    
    return U, V, W

def construct():
    """Construct rank-minimal exact tensor decomposition for 2x4 * 4x6 multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Build 2x4x4 (rank 28) and 2x4x2 (rank 14)
    U2, V2, W2 = _build_2x4x4_from_2x4x2()
    U1, V1, W1 = _build_2x4x2_from_strassen()
    
    # Verify sub-decompositions
    if not _verify_decomp(U1, V1, W1, 2, 4, 2, 14):
        U, V, W = _schoolbook_fallback(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if not _verify_decomp(U2, V2, W2, 2, 4, 4, 28):
        U, V, W = _schoolbook_fallback(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Combine: 2x4x6 = 2x4x4 (k=0..3) + 2x4x2 (k=4,5)
    total_rank = 42
    U_total = np.zeros((n * m, total_rank), dtype=float)
    V_total = np.zeros((m * p, total_rank), dtype=float)
    W_total = np.zeros((p * n, total_rank), dtype=float)
    
    # 2x4x4 part (k=0..3)
    U_total[:, :28] = U2
    for j in range(4):
        for k in range(4):
            V_total[j*6 + k, :28] = V2[j*4 + k, :]
    for k in range(4):
        for i in range(2):
            W_total[k*2 + i, :28] = W2[k*2 + i, :]
    
    # 2x4x2 part (k=4,5)
    U_total[:, 28:] = U1
    for j in range(4):
        for kl in range(2):
            V_total[j*6 + (kl+4), 28:] = V1[j*2 + kl, :]
    for kl in range(2):
        for i in range(2):
            W_total[(kl+4)*2 + i, 28:] = W1[kl*2 + i, :]
    
    # Final verification
    if not _verify_decomp(U_total, V_total, W_total, n, m, p, total_rank):
        U, V, W = _schoolbook_fallback(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

# EVOLVE-BLOCK-END

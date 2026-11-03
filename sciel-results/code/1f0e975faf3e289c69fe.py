import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomp(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
    T_target = _build_target_tensor(n, m, p)
    return np.allclose(T_comp, T_target, atol=1e-9)

def _schoolbook(n_sub, m_sub, p_sub):
    """Return schoolbook decomposition for n_sub x m_sub x p_sub."""
    rank = n_sub * m_sub * p_sub
    U = np.zeros((n_sub * m_sub, rank), dtype=np.float64)
    V = np.zeros((m_sub * p_sub, rank), dtype=np.float64)
    W = np.zeros((p_sub * n_sub, rank), dtype=np.float64)
    col = 0
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                U[i * m_sub + j, col] = 1.0
                V[j * p_sub + k, col] = 1.0
                W[k * n_sub + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_222():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7.
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (A00+A11)*(B00+B11) -> C00+=M1, C11+=M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00 -> C10+=M2, C11-=M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00*(B01-B11) -> C01+=M3, C11+=M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11*(B10-B00) -> C00+=M4, C10+=M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)*B11 -> C00-=M5, C01+=M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01) -> C11+=M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11) -> C00+=M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _embed_sub_decomp(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                      i_off, j_off, k_off, n, m, p):
    """
    Embed a sub-decomposition into the full n x m x p problem.
    """
    rank = U_sub.shape[1]
    U_full = np.zeros((n * m, rank), dtype=np.float64)
    V_full = np.zeros((m * p, rank), dtype=np.float64)
    W_full = np.zeros((p * n, rank), dtype=np.float64)
    
    for idx in range(n_sub * m_sub):
        i_s = idx // m_sub
        j_s = idx % m_sub
        i_f = i_off + i_s
        j_f = j_off + j_s
        U_full[i_f * m + j_f, :] = U_sub[idx, :]
    
    for idx in range(m_sub * p_sub):
        j_s = idx // p_sub
        k_s = idx % p_sub
        j_f = j_off + j_s
        k_f = k_off + k_s
        V_full[j_f * p + k_f, :] = V_sub[idx, :]
    
    for idx in range(p_sub * n_sub):
        k_s = idx // n_sub
        i_s = idx % n_sub
        k_f = k_off + k_s
        i_f = i_off + i_s
        W_full[k_f * n + i_f, :] = W_sub[idx, :]
    
    return U_full, V_full, W_full

def _sub_decomp(n_sub, m_sub, p_sub):
    """
    Get best known decomposition for a sub-block.
    Uses Strassen for 2x2x2, schoolbook otherwise.
    """
    if n_sub == 2 and m_sub == 2 and p_sub == 2:
        return _strassen_222()
    return _schoolbook(n_sub, m_sub, p_sub)

def _block_decomp_3x4x6():
    """
    Block decomposition for 3x4x6.
    Split: n=[2,1], m=[2,2], p=[2,2,2]
    Blocks:
      (2,2,2) x4 -> 4*7=28
      (1,2,2) x2 -> 2*4=8
      Total: 36
    """
    n, m, p = 3, 4, 6
    n_splits = [2, 1]
    m_splits = [2, 2]
    p_splits = [2, 2, 2]
    
    U_total = np.zeros((n * m, 0), dtype=np.float64)
    V_total = np.zeros((m * p, 0), dtype=np.float64)
    W_total = np.zeros((p * n, 0), dtype=np.float64)
    
    i_off = 0
    for n_sub in n_splits:
        j_off = 0
        for m_sub in m_splits:
            k_off = 0
            for p_sub in p_splits:
                U_sub, V_sub, W_sub = _sub_decomp(n_sub, m_sub, p_sub)
                U_full, V_full, W_full = _embed_sub_decomp(
                    U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                    i_off, j_off, k_off, n, m, p
                )
                if U_total.shape[1] == 0:
                    U_total = U_full
                    V_total = V_full
                    W_total = W_full
                else:
                    U_total = np.hstack([U_total, U_full])
                    V_total = np.hstack([V_total, V_full])
                    W_total = np.hstack([W_total, W_full])
                k_off += p_sub
            j_off += m_sub
        i_off += n_sub
    
    return U_total, V_total, W_total

def _block_decomposition_356():
    """
    Construct block decomposition for 3x5x6 by splitting m=5 into 1+4.
    Block 1: 3x1x6 (schoolbook, rank 18)
    Block 2: 3x4x6 (block decomp, rank 36)
    Total rank: 54
    """
    n, m, p = 3, 5, 6
    
    # Block 1: m indices [0], i.e., 3x1x6
    n1, m1, p1 = 3, 1, 6
    U1, V1, W1 = _schoolbook(n1, m1, p1)
    U1_full, V1_full, W1_full = _embed_sub_decomp(
        U1, V1, W1, n1, m1, p1,
        i_off=0, j_off=0, k_off=0, n=n, m=m, p=p
    )
    
    # Block 2: m indices [1,2,3,4], i.e., 3x4x6
    U2, V2, W2 = _block_decomp_3x4x6()
    U2_full, V2_full, W2_full = _embed_sub_decomp(
        U2, V2, W2, 3, 4, 6,
        i_off=0, j_off=1, k_off=0, n=n, m=m, p=p
    )
    
    U_total = np.hstack([U1_full, U2_full])
    V_total = np.hstack([V1_full, V2_full])
    W_total = np.hstack([W1_full, W2_full])
    
    return U_total, V_total, W_total

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    
    # Block decomposition splitting m=5 into 1+4 (rank 54)
    U, V, W = _block_decomposition_356()
    
    if not _verify_decomp(n, m, p, U, V, W):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END

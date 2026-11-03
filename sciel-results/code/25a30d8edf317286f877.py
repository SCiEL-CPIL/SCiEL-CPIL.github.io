import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _get_strassen_2x2x2():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U row i*2+j, V row j*2+k, W row k*2+i
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # m4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = -1; W[3, 4] = 1
    
    # m6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1
    
    # m7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[2, 6] = 1
    
    return U, V, W

def _build_decomposition_84(n, m, p):
    """
    Builds the 84-rank decomposition for 3x5x6.
    Partition: n={2,1}, m={2,2,1}, p={2,2,2}
    """
    sU, sV, sW = _get_strassen_2x2x2()
    
    # Partition indices
    # i: {0,1} top, {2} bottom
    # j: {0,1}, {2,3}, {4}
    # k: {0,1}, {2,3}, {4,5}
    
    n_blocks = [(0, 2), (2, 1)]
    m_blocks = [(0, 2), (2, 2), (4, 1)]
    p_blocks = [(0, 2), (2, 2), (4, 2)]
    
    # Calculate total rank
    total_rank = 0
    for n_start, n_size in n_blocks:
        for m_start, m_size in m_blocks:
            for p_start, p_size in p_blocks:
                if n_size == 2 and m_size == 2 and p_size == 2:
                    total_rank += 7
                else:
                    total_rank += n_size * m_size * p_size
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col = 0
    
    for n_start, n_size in n_blocks:
        for m_start, m_size in m_blocks:
            for p_start, p_size in p_blocks:
                if n_size == 2 and m_size == 2 and p_size == 2:
                    # Strassen block
                    for c in range(7):
                        # U: rows (i, j) where i in [n_start, n_start+2), j in [m_start, m_start+2)
                        # Local i, j in {0,1}
                        # U[local_i * 2 + local_j, c]
                        
                        # Map local to global
                        for li in range(2):
                            for lj in range(2):
                                gi = n_start + li
                                gj = m_start + lj
                                U[gi * m + gj, col] = sU[li * 2 + lj, c]
                        
                        for lj in range(2):
                            for lk in range(2):
                                gj = m_start + lj
                                gk = p_start + lk
                                V[gj * p + gk, col] = sV[lj * 2 + lk, c]
                        
                        for lk in range(2):
                            for li in range(2):
                                gk = p_start + lk
                                gi = n_start + li
                                W[gk * n + gi, col] = sW[lk * 2 + li, c]
                        
                        col += 1
                else:
                    # Schoolbook block
                    for li in range(n_size):
                        for lj in range(m_size):
                            for lk in range(p_size):
                                gi = n_start + li
                                gj = m_start + lj
                                gk = p_start + lk
                                U[gi * m + gj, col] = 1.0
                                V[gj * p + gk, col] = 1.0
                                W[gk * n + gi, col] = 1.0
                                col += 1
    
    if col != total_rank:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        total_rank = col
        
    return U, V, W, total_rank

def _verify(U, V, W, n, m, p):
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125

    # Try the 84-rank decomposition
    try:
        U, V, W, rank = _build_decomposition_84(n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback: schoolbook
    rank_school = n * m * p
    U_fb = np.zeros((n * m, rank_school), dtype=np.float64)
    V_fb = np.zeros((m * p, rank_school), dtype=np.float64)
    W_fb = np.zeros((p * n, rank_school), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END

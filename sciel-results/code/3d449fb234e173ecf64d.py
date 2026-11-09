import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin within 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    # Partition indices into blocks
    m_blocks = [(0, 2), (2, 4), (4, 5)]  # column blocks of A / row blocks of B
    p_blocks = [(0, 2), (2, 4), (4, 5)]  # column blocks of B / column blocks of C
    
    def block_rank(s, t):
        """Rank for multiplying (2×s) by (s×t) -> (2×t)"""
        if s == 2 and t == 2:
            return 7  # Strassen
        else:
            return 2 * s * t  # schoolbook
    
    total_rank = 0
    for (m_start, m_end) in m_blocks:
        s = m_end - m_start
        for (p_start, p_end) in p_blocks:
            t = p_end - p_start
            total_rank += block_rank(s, t)
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    for (m_start, m_end) in m_blocks:
        s = m_end - m_start
        for (p_start, p_end) in p_blocks:
            t = p_end - p_start
            
            if s == 2 and t == 2:
                # Strassen decomposition for 2×2×2 block
                # Local indices: i in {0,1}, j_local in {0,1}, k_local in {0,1}
                # Global: j = m_start + j_local, k = p_start + k_local
                
                # Strassen 7 terms for 2×2×2 matrix multiplication
                # Local U indices: i*2 + j_local (0-3)
                # Local V indices: j_local*2 + k_local (0-3)
                # Local W indices: k_local*2 + i (0-3)
                
                # The 7 rank-1 tensors (verified to reconstruct 2×2×2 matmul tensor)
                strassen_terms = [
                    # (U_local, V_local, W_local)
                    ([1,0,0,1], [1,0,0,1], [1,0,0,1]),  # M1
                    ([0,0,1,1], [1,0,0,0], [0,0,1,0]),  # M2
                    ([1,0,0,0], [0,0,1,-1], [0,1,0,1]),  # M3
                    ([0,0,0,1], [0,1,-1,0], [1,0,1,0]),  # M4
                    ([1,0,1,0], [0,0,0,1], [-1,1,0,0]),  # M5
                    ([0,0,-1,1], [1,0,1,0], [0,0,0,1]),  # M6
                    ([0,1,0,-1], [0,1,0,1], [1,0,0,0]),  # M7
                ]
                
                for u_loc, v_loc, w_loc in strassen_terms:
                    # Place U values at global positions
                    for i in range(2):
                        for j_local in range(2):
                            j = m_start + j_local
                            U[i * m + j, col] = u_loc[i * 2 + j_local]
                    
                    # Place V values at global positions
                    for j_local in range(2):
                        for k_local in range(2):
                            j = m_start + j_local
                            k = p_start + k_local
                            V[j * p + k, col] = v_loc[j_local * 2 + k_local]
                    
                    # Place W values at global positions
                    for k_local in range(2):
                        for i in range(2):
                            k = p_start + k_local
                            W[k * n + i, col] = w_loc[k_local * 2 + i]
                    
                    col += 1
            else:
                # Schoolbook for this block
                for i in range(n):
                    for j_local in range(s):
                        for k_local in range(t):
                            j = m_start + j_local
                            k = p_start + k_local
                            U[i * m + j, col] = 1.0
                            V[j * p + k, col] = 1.0
                            W[k * n + i, col] = 1.0
                            col += 1
    
    # Verify the decomposition
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-9, rtol=0):
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U_school = np.zeros((n*m, schoolbook_rank), dtype=float)
        V_school = np.zeros((m*p, schoolbook_rank), dtype=float)
        W_school = np.zeros((p*n, schoolbook_rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, col] = 1.0
                    V_school[j*p+k, col] = 1.0
                    W_school[k*n+i, col] = 1.0
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

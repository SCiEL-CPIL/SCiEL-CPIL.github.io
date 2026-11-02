"""Matrix multiplication 2,4,8 - Optimized Strassen-based rank reduction."""
DIMENSIONS = [2, 4, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _make_strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices:
    U[i*2+j, c]
    V[j*2+k, c]
    W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10 += M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[0, 1] = 1
    
    # M3: A00(B01-B11) -> C01 += M3, C11 -= M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = -1
    
    # M4: A11(B10-B00) -> C00 -= M4, C10 += M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 0] -= 1; W[0, 1] += 1
    
    # M5: (A00+A01)B11 -> C01 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C00 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C11 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1
    
    return U, V, W

def _make_decomposition(n, m, p):
    """Build full decomposition for n x m x p using 2x2 Strassen blocks."""
    n_blocks = n // 2
    m_blocks = m // 2
    p_blocks = p // 2
    
    rank = n_blocks * m_blocks * p_blocks * 7
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    su, sv, sw = _make_strassen_2x2x2()
    
    col = 0
    for ib in range(n_blocks):
        for jb in range(m_blocks):
            for kb in range(p_blocks):
                for c in range(7):
                    for i_off in range(2):
                        for j_off in range(2):
                            u_val = su[i_off * 2 + j_off, c]
                            if u_val != 0:
                                i_global = ib * 2 + i_off
                                j_global = jb * 2 + j_off
                                U[i_global * m + j_global, col] += u_val
                    for j_off in range(2):
                        for k_off in range(2):
                            v_val = sv[j_off * 2 + k_off, c]
                            if v_val != 0:
                                j_global = jb * 2 + j_off
                                k_global = kb * 2 + k_off
                                V[j_global * p + k_global, col] += v_val
                    for k_off in range(2):
                        for i_off in range(2):
                            w_val = sw[k_off * 2 + i_off, c]
                            if w_val != 0:
                                k_global = kb * 2 + k_off
                                i_global = ib * 2 + i_off
                                W[k_global * n + i_global, col] += w_val
                    col += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    try:
        U, V, W = _make_decomposition(n, m, p)
        rank = len(U[0])
        
        # Verify correctness
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid and time.time() < deadline:
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook (rank 64)
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

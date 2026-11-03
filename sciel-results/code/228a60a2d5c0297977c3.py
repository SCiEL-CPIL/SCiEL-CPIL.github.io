import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _get_strassen_222():
    """
    Returns (U, V, W) for 2x2x2 Strassen multiplication.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indexing:
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Term 0: M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1
    U[3][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    W[0][0] += 1
    W[3][0] += 1

    # Term 1: M2 = (A10 + A11)B00
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[2][1] += 1
    W[3][1] += -1

    # Term 2: M3 = A00(B01 - B11)
    U[0][2] = 1
    V[1][2] = 1
    V[3][2] = -1
    W[2][2] += 1
    W[3][2] += 1

    # Term 3: M4 = A11(B10 - B00)
    U[3][3] = 1
    V[2][3] = 1
    V[0][3] = -1
    W[0][3] += -1
    W[2][3] += 1

    # Term 4: M5 = (A00 + A01)B11
    U[0][4] = 1
    U[1][4] = 1
    V[3][4] = 1
    W[0][4] += 1
    W[2][4] += 1

    # Term 5: M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[1][5] = 1
    W[3][5] += 1

    # Term 6: M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1
    U[3][6] = -1
    V[2][6] = 1
    V[3][6] = 1
    W[0][6] += -1

    return U, V, W

def _build_rank71():
    n, m, p = 3, 5, 5
    
    su, sv, sw = _get_strassen_222()
    
    # Partitions
    i_blocks = [(0, 1), (2,)]
    j_blocks = [(0, 1), (2, 3), (4,)]
    k_blocks = [(0, 1), (2, 3), (4,)]
    
    total_rank = 0
    for i_b in i_blocks:
        for j_b in j_blocks:
            for k_b in k_blocks:
                if len(i_b)==2 and len(j_b)==2 and len(k_b)==2:
                    total_rank += 7
                else:
                    total_rank += len(i_b)*len(j_b)*len(k_b)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i_b in i_blocks:
        for j_b in j_blocks:
            for k_b in k_blocks:
                if len(i_b)==2 and len(j_b)==2 and len(k_b)==2:
                    # Strassen
                    for c in range(7):
                        for li in range(2):
                            gi = i_b[li]
                            for lj in range(2):
                                gj = j_b[lj]
                                U[gi*m + gj][col+c] = su[li*2+lj][c]
                        for lj in range(2):
                            gj = j_b[lj]
                            for lk in range(2):
                                gk = k_b[lk]
                                V[gj*p + gk][col+c] = sv[lj*2+lk][c]
                        for lk in range(2):
                            gk = k_b[lk]
                            for li in range(2):
                                gi = i_b[li]
                                W[gk*n + gi][col+c] = sw[lk*2+li][c]
                    col += 7
                else:
                    # Schoolbook
                    for gi in i_b:
                        for gj in j_b:
                            for gk in k_b:
                                U[gi*m + gj][col] = 1
                                V[gj*p + gk][col] = 1
                                W[gk*n + gi][col] = 1
                                col += 1
                                
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, rank):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i*m + j]
                v_val = V_np[j*p + k]
                w_val = W_np[k*n + i]
                s = np.sum(u_val * v_val * w_val)
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def _schoolbook_factors(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    try:
        U, V, W = _build_rank71()
        rank = len(W[0])
        if _verify_decomposition(U, V, W, n, m, p, rank):
            return {"U": U, "V": V, "W": W}
    except Exception as e:
        pass
        
    U, V, W = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

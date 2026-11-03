import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _get_strassen_222():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1
    
    return U, V, W

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

def _build_rank71():
    n, m, p = 3, 5, 5
    
    su, sv, sw = _get_strassen_222()
    
    # Partitions
    # i_blocks: indices for rows of A (and C)
    # j_blocks: indices for cols of A (and rows of B)
    # k_blocks: indices for cols of B (and cols of C)
    
    i_blocks = [(0,), (1, 2)]
    j_blocks = [(0, 1), (2, 3), (4,)]
    k_blocks = [(0, 1), (2, 3), (4,)]
    
    total_rank = 0
    for i_b in i_blocks:
        for j_b in j_blocks:
            for k_b in k_blocks:
                ni = len(i_b)
                nj = len(j_b)
                nk = len(k_b)
                if ni == 2 and nj == 2 and nk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * nj * nk
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i_b in i_blocks:
        for j_b in j_blocks:
            for k_b in k_blocks:
                ni = len(i_b)
                nj = len(j_b)
                nk = len(k_b)
                
                if ni == 2 and nj == 2 and nk == 2:
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
                    for li in range(ni):
                        gi = i_b[li]
                        for lj in range(nj):
                            gj = j_b[lj]
                            for lk in range(nk):
                                gk = k_b[lk]
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
                s = np.sum(W_np[k*n + i] * U_np[i*m + j] * V_np[j*p + k])
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    try:
        if time.time() < deadline - 10:
            U, V, W = _build_rank71()
            if _verify_decomposition(U, V, W, n, m, p, 71):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
        
    U, V, W = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

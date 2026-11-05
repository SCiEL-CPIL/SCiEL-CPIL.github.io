"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return U, V, W for 2x2 matrix multiplication (rank 7).
    U shape (4,7), V shape (4,7), W shape (4,7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen's 7 products for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    V[0][0] = 1; V[3][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[3][2] = -1
    V[2][3] = 1; V[0][3] = -1
    V[3][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[2][6] = 1; V[3][6] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][2] = 1; W[1][4] = 1
    W[2][1] = 1; W[2][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    return U, V, W

def _schoolbook(ni, mj, pk):
    """Schoolbook multiplication for ni x mj * mj x pk."""
    rank = ni * mj * pk
    U = [[0]*rank for _ in range(ni * mj)]
    V = [[0]*rank for _ in range(mj * pk)]
    W = [[0]*rank for _ in range(pk * ni)]
    col = 0
    for a in range(ni):
        for b in range(mj):
            for c in range(pk):
                U[a * mj + b][col] = 1
                V[b * pk + c][col] = 1
                W[c * ni + a][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    start_time = time.time()
    deadline = start_time + 110
    
    # Block decomposition:
    # A (3x4): rows [2,1], cols [2,2]
    # B (4x5): rows [2,2], cols [2,2,1]
    # C (3x5): rows [2,1], cols [2,2,1]
    
    n_sizes = [2, 1]
    m_sizes = [2, 2]
    p_sizes = [2, 2, 1]
    
    def get_block(ni, mj, pk):
        if ni == 2 and mj == 2 and pk == 2:
            return _strassen_2x2()
        else:
            return _schoolbook(ni, mj, pk)
    
    # Build full U, V, W
    total_rank = 0
    for i in range(len(n_sizes)):
        for j in range(len(m_sizes)):
            for k in range(len(p_sizes)):
                ni, mj, pk = n_sizes[i], m_sizes[j], p_sizes[k]
                _, _, _ = get_block(ni, mj, pk)
                total_rank += ni * mj * pk if not (ni==2 and mj==2 and pk==2) else 7
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(len(n_sizes)):
        for j in range(len(m_sizes)):
            for k in range(len(p_sizes)):
                ni, mj, pk = n_sizes[i], m_sizes[j], p_sizes[k]
                U_sub, V_sub, W_sub = get_block(ni, mj, pk)
                r = len(U_sub[0])
                
                for ii in range(ni):
                    for jj in range(mj):
                        u_row = (i * n_sizes[i] + ii) * m + (j * m_sizes[j] + jj)
                        u_sub_row = ii * mj + jj
                        for rr in range(r):
                            U[u_row][col + rr] = U_sub[u_sub_row][rr]
                
                for jj in range(mj):
                    for kk in range(pk):
                        v_row = (j * m_sizes[j] + jj) * p + (k * p_sizes[k] + kk)
                        v_sub_row = jj * pk + kk
                        for rr in range(r):
                            V[v_row][col + rr] = V_sub[v_sub_row][rr]
                
                for kk in range(pk):
                    for ii in range(ni):
                        w_row = (k * p_sizes[k] + kk) * n + (i * n_sizes[i] + ii)
                        w_sub_row = kk * ni + ii
                        for rr in range(r):
                            W[w_row][col + rr] = W_sub[w_sub_row][rr]
                
                col += r
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for r in range(total_rank):
        term_A = np.zeros(n)
        term_B = np.zeros(p)
        for i in range(n):
            for j in range(m):
                term_A[i] += A[i, j] * U[i * m + j][r]
        for k in range(p):
            for j in range(m):
                term_B[k] += B[j, k] * V[j * p + k][r]
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W[k * n + i][r] * term_A[i] * term_B[k]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

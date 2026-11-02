"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2_2x2():
    """
    Return the Strassen decomposition for 2x2 * 2x2 -> 2x2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    U[i*2+j] = A[i,j], V[j*2+k] = B[j,k], W[k*2+i] = C[i,k]
    W[0]=C[0,0], W[1]=C[1,0], W[2]=C[0,1], W[3]=C[1,1]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a11+a22)(b11+b22) -> c11, c22
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a21+a22)b11 -> c21
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # M3 = a11(b12-b22) -> c12
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # M4 = a22(b21-b11) -> c11, c21
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (a11+a12)b22 -> c11(-1), c12
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (a21-a11)(b11+b12) -> c22
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a12-a22)(b21+b22) -> c11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _block_strassen_4x4_4x8():
    """
    Build the block Strassen decomposition for (4,4,8).
    """
    n, m, p = 4, 4, 8
    U_s, V_s, W_s = _strassen_2x2_2x2()
    
    total_rank = 112
    U = np.zeros((16, total_rank))
    V = np.zeros((32, total_rank))
    W = np.zeros((32, total_rank))
    
    def embed_strassen_2x2_2x4(U_off, V_off, W_off, col_start):
        # Left half of 2x4 (cols 0-1)
        for c in range(7):
            for r in range(4):
                if U_s[r, c] != 0:
                    U[U_off + r, col_start + c] += U_s[r, c]
                if V_s[r, c] != 0:
                    V[V_off + r, col_start + c] += V_s[r, c]
                if W_s[r, c] != 0:
                    W[W_off + r, col_start + c] += W_s[r, c]
        # Right half of 2x4 (cols 2-3)
        for c in range(7):
            for r in range(4):
                if U_s[r, c] != 0:
                    U[U_off + r, col_start + 7 + c] += U_s[r, c]
                if V_s[r, c] != 0:
                    V[V_off + 4 + r, col_start + 7 + c] += V_s[r, c]
                if W_s[r, c] != 0:
                    W[W_off + 4 + r, col_start + 7 + c] += W_s[r, c]
    
    # A11*B11
    embed_strassen_2x2_2x4(0, 0, 0, 0)
    # A12*B21
    embed_strassen_2x2_2x4(4, 16, 0, 14)
    # A11*B12
    embed_strassen_2x2_2x4(0, 8, 8, 28)
    # A12*B22
    embed_strassen_2x2_2x4(4, 24, 8, 42)
    # A21*B11
    embed_strassen_2x2_2x4(8, 0, 16, 56)
    # A22*B21
    embed_strassen_2x2_2x4(12, 16, 16, 70)
    # A21*B12
    embed_strassen_2x2_2x4(8, 8, 24, 84)
    # A22*B22
    embed_strassen_2x2_2x4(12, 24, 24, 98)
    
    return U, V, W

def construct():
    start_time = time.time()
    
    try:
        U, V, W = _block_strassen_4x4_4x8()
        U_list = [[int(U[i][j]) if U[i][j] == int(U[i][j]) else float(U[i][j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
        V_list = [[int(V[i][j]) if V[i][j] == int(V[i][j]) else float(V[i][j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
        W_list = [[int(W[i][j]) if W[i][j] == int(W[i][j]) else float(W[i][j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
        return {"U": U_list, "V": V_list, "W": W_list}
    except:
        n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END

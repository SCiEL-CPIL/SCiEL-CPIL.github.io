"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Block decomposition: split middle dimension (4) into two groups of 2
    # A = [A1 | A2] (each 2x2), B = [B1; B2] (each 2x3)
    # C = A1*B1 + A2*B2
    # Each term is 2x2 * 2x3 = 12 scalar mults, total 24
    rank = 24
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for block in range(2):
        j_off = block * 2
        for i in range(n):
            for j in range(2):
                for k in range(p):
                    U[i*m + (j_off + j), col] = 1
                    V[(j_off + j)*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
    
    # Verify
    def verify(U_mat, V_mat, W_mat, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_mat[i*m+j, :r], V_mat[j*p+k, :r] * W_mat[k*n+i, :r])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, rank):
        # Fall back to full schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END

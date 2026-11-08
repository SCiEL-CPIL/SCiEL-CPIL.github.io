import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def build_rank33():
        rank = 33
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        col = 0
        
        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + (j_off + 0)][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 0) * p + (k_off + 0)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[(k_off + 0) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P2 = (A10 + A11) B00
            U[1 * m + (j_off + 0)][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 0) * p + (k_off + 0)][c] += 1
            W[(k_off + 0) * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            
            # P3 = A00 (B01 - B11)
            U[0 * m + (j_off + 0)][c] += 1
            V[(j_off + 0) * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 0)][c] += 1
            V[(j_off + 0) * p + (k_off + 0)][c] -= 1
            W[(k_off + 0) * n + 0][c] += 1
            W[(k_off + 0) * n + 1][c] += 1
            c += 1
            
            # P5 = (A00 + A01) B11
            U[0 * m + (j_off + 0)][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[(k_off + 0) * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            
            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + (j_off + 0)][c] += 1
            U[0 * m + (j_off + 0)][c] -= 1
            V[(j_off + 0) * p + (k_off + 0)][c] += 1
            V[(j_off + 0) * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + (k_off + 0)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[(k_off + 0) * n + 0][c] += 1
            c += 1
            
            return c

        # 4 Strassen blocks
        col = add_strassen(0, 0, col)  # J1 x K1
        col = add_strassen(0, 2, col)  # J1 x K2
        col = add_strassen(2, 0, col)  # J2 x K1
        col = add_strassen(2, 2, col)  # J2 x K2
        
        # Boundary terms (Rank 1 each)
        
        # J1 x K3 (j in 0,1; k=4)
        for i in range(n):
            for j in [0, 1]:
                U[i * m + j][col] = 1
        for j in [0, 1]:
            V[j * p + 4][col] = 1
        for i in range(n):
            W[4 * n + i][col] = 1
        col += 1
        
        # J2 x K3 (j in 2,3; k=4)
        for i in range(n):
            for j in [2, 3]:
                U[i * m + j][col] = 1
        for j in [2, 3]:
            V[j * p + 4][col] = 1
        for i in range(n):
            W[4 * n + i][col] = 1
        col += 1
        
        # J3 x K1 (j=4; k in 0,1)
        for i in range(n):
            U[i * m + 4][col] = 1
        for k in [0, 1]:
            V[4 * p + k][col] = 1
        for i in range(n):
            for k in [0, 1]:
                W[k * n + i][col] = 1
        col += 1
        
        # J3 x K2 (j=4; k in 2,3)
        for i in range(n):
            U[i * m + 4][col] = 1
        for k in [2, 3]:
            V[4 * p + k][col] = 1
        for i in range(n):
            for k in [2, 3]:
                W[k * n + i][col] = 1
        col += 1
        
        # J3 x K3 (j=4; k=4)
        for i in range(n):
            U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        for i in range(n):
            W[4 * n + i][col] = 1
        col += 1
        
        return U, V, W

    U33, V33, W33 = build_rank33()
    
    if validate(U33, V33, W33):
        return {
            "U": [[float(x) for x in row] for row in U33],
            "V": [[float(x) for x in row] for row in V33],
            "W": [[float(x) for x in row] for row in W33]
        }
    else:
        # Fallback to rank 50
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
# EVOLVE-BLOCK-END

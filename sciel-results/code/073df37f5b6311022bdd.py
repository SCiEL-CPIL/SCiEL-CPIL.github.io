"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 5
    # Schoolbook rank is 50.
    # We'll use a known decomposition for 2x5x5 with rank 46.
    # The decomposition is based on the following:
    # - 4 blocks of 2x2x2 (each rank 7) -> 28
    # - 1 block of 2x4x1 (rank 8) -> 8
    # - 1 block of 2x1x5 (rank 10) -> 10
    # Total = 46.
    
    # U is (n*m) x rank = 10 x 46
    # V is (m*p) x rank = 25 x 46
    # W is (p*n) x rank = 10 x 46
    
    U = np.zeros((10, 46))
    V = np.zeros((25, 46))
    W = np.zeros((10, 46))
    
    col = 0
    
    # 4 blocks of 2x2x2 (Strassen)
    # Each block uses 7 products.
    # Block 1: A[:, 0:2] * B[0:2, 0:2]
    # Block 2: A[:, 0:2] * B[0:2, 2:4]
    # Block 3: A[:, 2:4] * B[2:4, 0:2]
    # Block 4: A[:, 2:4] * B[2:4, 2:4]
    
    # Strassen for 2x2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each 2x2x2 block, we add 7 columns.
    # We'll write a helper to add a Strassen block.
    
    def add_strassen_block(U, V, W, col, i1, i2, j1, j2, k1, k2):
        # i1, i2: row indices for A (and C)
        # j1, j2: col indices for A, row indices for B
        # k1, k2: col indices for B (and C)
        # We assume the block is 2x2x2.
        
        # M1 = (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
        U[i1 * 5 + j1][col] = 1
        U[i2 * 5 + j2][col] = 1
        V[j1 * 5 + k1][col] = 1
        V[j2 * 5 + k2][col] = 1
        W[k1 * 2 + i1][col] = 1
        W[k1 * 2 + i2][col] = 1
        W[k2 * 2 + i1][col] = 1
        W[k2 * 2 + i2][col] = 1
        col += 1
        
        # M2 = (A[i2,j1] + A[i2,j2]) * B[j1,k1]
        U[i2 * 5 + j1][col] = 1
        U[i2 * 5 + j2][col] = 1
        V[j1 * 5 + k1][col] = 1
        W[k1 * 2 + i2][col] = 1
        col += 1
        
        # M3 = A[i1,j1] * (B[j1,k2] - B[j2,k2])
        U[i1 * 5 + j1][col] = 1
        V[j1 * 5 + k2][col] = 1
        V[j2 * 5 + k2][col] = -1
        W[k2 * 2 + i1][col] = 1
        col += 1
        
        # M4 = A[i2,j2] * (B[j2,k1] - B[j1,k1])
        U[i2 * 5 + j2][col] = 1
        V[j2 * 5 + k1][col] = 1
        V[j1 * 5 + k1][col] = -1
        W[k1 * 2 + i2][col] = 1
        col += 1
        
        # M5 = (A[i1,j1] + A[i1,j2]) * B[j2,k2]
        U[i1 * 5 + j1][col] = 1
        U[i1 * 5 + j2][col] = 1
        V[j2 * 5 + k2][col] = 1
        W[k2 * 2 + i1][col] = -1
        col += 1
        
        # M6 = (A[i2,j1] - A[i1,j1]) * (B[j1,k1] + B[j1,k2])
        U[i2 * 5 + j1][col] = 1
        U[i1 * 5 + j1][col] = -1
        V[j1 * 5 + k1][col] = 1
        V[j1 * 5 + k2][col] = 1
        W[k1 * 2 + i2][col] = 1
        W[k2 * 2 + i2][col] = 1
        col += 1
        
        # M7 = (A[i1,j2] - A[i2,j2]) * (B[j2,k1] + B[j2,k2])
        U[i1 * 5 + j2][col] = 1
        U[i2 * 5 + j2][col] = -1
        V[j2 * 5 + k1][col] = 1
        V[j2 * 5 + k2][col] = 1
        W[k1 * 2 + i1][col] = 1
        W[k2 * 2 + i1][col] = 1
        col += 1
        
        return col
    
    # Block 1: i1=0, i2=1, j1=0, j2=1, k1=0, k2=1
    col = add_strassen_block(U, V, W, col, 0, 1, 0, 1, 0, 1)
    
    # Block 2: i1=0, i2=1, j1=0, j2=1, k1=2, k2=3
    col = add_strassen_block(U, V, W, col, 0, 1, 0, 1, 2, 3)
    
    # Block 3: i1=0, i2=1, j1=2, j2=3, k1=0, k2=1
    col = add_strassen_block(U, V, W, col, 0, 1, 2, 3, 0, 1)
    
    # Block 4: i1=0, i2=1, j1=2, j2=3, k1=2, k2=3
    col = add_strassen_block(U, V, W, col, 0, 1, 2, 3, 2, 3)
    
    # Block 5: A[:, 0:4] * B[0:4, 4] -> C[:, 4]
    # For each j in 0..3, product A[:, j] * B[j, 4]
    for j in range(4):
        U[0 * 5 + j][col] = 1
        U[1 * 5 + j][col] = 1
        V[j * 5 + 4][col] = 1
        W[4 * 2 + 0][col] = 1
        W[4 * 2 + 1][col] = 1
        col += 1
    
    # Block 6: A[:, 4] * B[4, 0:5] -> C[:, 0:5]
    # For each k in 0..4, product A[:, 4] * B[4, k]
    for k in range(5):
        U[0 * 5 + 4][col] = 1
        U[1 * 5 + 4][col] = 1
        V[4 * 5 + k][col] = 1
        W[k * 2 + 0][col] = 1
        W[k * 2 + 1][col] = 1
        col += 1
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

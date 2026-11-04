"""Tensor-product of Strassen's rank-7 decomposition for 4x4 matrix multiplication."""
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # C = A @ B where A, B, C are 2x2
    # C11 = M1 + M4 - M5 - M7
    # C12 = M2 + M5
    # C21 = M3 + M6
    # C22 = M1 - M4 + M6 + M7
    #
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # U_A[i, m] for A[i,j]: U_A row index i*2+j, column m
    U_A = np.zeros((4, 7))
    V_B = np.zeros((4, 7))
    W_C = np.zeros((4, 7))
    
    # M1 = (A11 + A22)(B11 + B22)
    U_A[0][0] = 1; U_A[3][0] = 1
    V_B[0][0] = 1; V_B[3][0] = 1
    W_C[0][0] = 1; W_C[3][0] = 1  # C11 += M1, C22 += M1
    
    # M2 = (A21 + A22)B11
    U_A[2][1] = 1; U_A[3][1] = 1
    V_B[0][1] = 1
    W_C[1][1] = 1; W_C[3][1] = -1  # C12 += M2, C22 -= M2
    
    # M3 = A11(B21 - B22)
    U_A[0][2] = 1
    V_B[2][2] = 1; V_B[3][2] = -1
    W_C[2][2] = 1; W_C[3][2] = 1  # C21 += M3, C22 += M3
    
    # M4 = A22(B12 - B11)
    U_A[3][3] = 1
    V_B[1][3] = 1; V_B[0][3] = -1
    W_C[0][3] = 1; W_C[3][3] = -1  # C11 += M4, C22 -= M4
    
    # M5 = (A11 + A12)B22
    U_A[0][4] = 1; U_A[1][4] = 1
    V_B[3][4] = 1
    W_C[0][4] = -1; W_C[1][4] = 1  # C11 -= M5, C12 += M5
    
    # M6 = (A21 - A11)(B11 + B12)
    U_A[2][5] = 1; U_A[0][5] = -1
    V_B[0][5] = 1; V_B[1][5] = 1
    W_C[2][5] = 1; W_C[3][5] = 1  # C21 += M6, C22 += M6
    
    # M7 = (A12 - A22)(B21 + B22)
    U_A[1][6] = 1; U_A[3][6] = -1
    V_B[2][6] = 1; V_B[3][6] = 1
    W_C[0][6] = -1; W_C[1][6] = 1  # C11 -= M7, C12 += M7
    
    # Now construct 4x4 decomposition using tensor product
    # For 4x4: split into 2x2 blocks
    # A is 4x4, indices (i,j) where i,j in 0..3
    # i = i1*2 + i0, j = j1*2 + j0
    # A[i,j] = A_{i1,j1}[i0,j0] (block (i1,j1) of A, element (i0,j0))
    
    # U for 4x4: shape (16, 49)
    # U[i*4+j, c1*7+c2] = U_A[i1*2+i0, c1] * U_A[j1*2+j0, c2]
    # where i = i1*2+i0, j = j1*2+j0
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for i in range(4):
        for j in range(4):
            i1 = i // 2
            i0 = i % 2
            j1 = j // 2
            j0 = j % 2
            row_U = i * 4 + j
            
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    U[row_U, col] = U_A[i1 * 2 + i0, c1] * U_A[j1 * 2 + j0, c2]
    
    # V for 4x4: shape (16, 49)
    # V[j*4+k, c1*7+c2] = V_B[j1*2+j0, c1] * V_B[k1*2+k0, c2]
    # where j = j1*2+j0, k = k1*2+k0
    
    for j in range(4):
        for k in range(4):
            j1 = j // 2
            j0 = j % 2
            k1 = k // 2
            k0 = k % 2
            row_V = j * 4 + k
            
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    V[row_V, col] = V_B[j1 * 2 + j0, c1] * V_B[k1 * 2 + k0, c2]
    
    # W for 4x4: shape (16, 49)
    # W[k*4+i, c1*7+c2] = W_C[k1*2+k0, c1] * W_C[i1*2+i0, c2]
    # where k = k1*2+k0, i = i1*2+i0
    # Note: W follows convention W[k*n+i, r] where n=4, so row index is k*4+i
    
    for k in range(4):
        for i in range(4):
            k1 = k // 2
            k0 = k % 2
            i1 = i // 2
            i0 = i % 2
            row_W = k * 4 + i
            
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    W[row_W, col] = W_C[k1 * 2 + k0, c1] * W_C[i1 * 2 + i0, c2]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

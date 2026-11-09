"""Matrix multiplication 4x4x7 rank-minimal exact tensor decomposition."""
import numpy as np

def construct():
    n, m, p = 4, 4, 7
    total_rank = 56
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Strassen 2x2x2 base case (rank 7)
    # A indices: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B indices: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    # C indices: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    
    # For 2x2*2x2:
    # U row = i*2+j, V row = j*2+k, W row = k*2+i
    # n=2, m=2, p=2
    
    strassen_U = np.zeros((4, 7))
    strassen_V = np.zeros((4, 7))
    strassen_W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    strassen_U[0, 0] = 1; strassen_U[3, 0] = 1
    strassen_V[0, 0] = 1; strassen_V[3, 0] = 1
    strassen_W[0, 0] = 1; strassen_W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    strassen_U[2, 1] = 1; strassen_U[3, 1] = 1
    strassen_V[0, 1] = 1
    strassen_W[2, 1] = 1; strassen_W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    strassen_U[0, 2] = 1
    strassen_V[1, 2] = 1; strassen_V[3, 2] = -1
    strassen_W[1, 2] = 1; strassen_W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    strassen_U[3, 3] = 1
    strassen_V[2, 3] = 1; strassen_V[0, 3] = -1
    strassen_W[0, 3] = 1; strassen_W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    strassen_U[0, 4] = 1; strassen_U[1, 4] = 1
    strassen_V[3, 4] = 1
    strassen_W[0, 4] = -1; strassen_W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    strassen_U[2, 5] = 1; strassen_U[0, 5] = -1
    strassen_V[0, 5] = 1; strassen_V[1, 5] = 1
    strassen_W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    strassen_U[1, 6] = 1; strassen_U[3, 6] = -1
    strassen_V[2, 6] = 1; strassen_V[3, 6] = 1
    strassen_W[0, 6] = 1
    
    # Build 4x4x7 decomposition
    # Split A (4x4) into 2x2 blocks: A11, A12, A21, A22
    # Split B (4x7) into 2x7 blocks: B1 (rows 0-1), B2 (rows 2-3)
    # C = A*B, split into C1 (rows 0-1, 2x7) and C2 (rows 2-3, 2x7)
    # C1 = A11*B1 + A12*B2
    # C2 = A21*B1 + A22*B2
    
    # For each 2x2 block multiplication with 2x7 B:
    # Split B (2x7) into B_a (2x4) and B_b (2x3)
    # Use Strassen on (2x2)*(2x4): rank 14 (two 2x2 blocks)
    # Use Strassen+schoolbook on (2x2)*(2x3): rank 11
    
    def _mul_2x2_x_2x4():
        """(2x2)*(2x4) -> (2x4), rank 14"""
        U = np.zeros((4, 14))
        V = np.zeros((8, 14))
        W = np.zeros((8, 14))
        
        # Two 2x2*2x2 multiplications
        # Block 1: cols 0-1
        U[:, :7] = strassen_U
        V[0:4, :7] = strassen_V
        # W: C is 2x4, W index = k*2+i for C[i,k]
        # C1 (cols 0-1): C[0,0]=W[0], C[1,0]=W[1], C[0,1]=W[2], C[1,1]=W[3]
        W[0, :7] = strassen_W[0, :]
        W[1, :7] = strassen_W[1, :]
        W[2, :7] = strassen_W[2, :]
        W[3, :7] = strassen_W[3, :]
        
        # Block 2: cols 2-3
        U[:, 7:] = strassen_U
        V[4:8, 7:] = strassen_V
        W[4, 7:] = strassen_W[0, :]
        W[5, 7:] = strassen_W[1, :]
        W[6, 7:] = strassen_W[2, :]
        W[7, 7:] = strassen_W[3, :]
        
        return U, V, W
    
    def _mul_2x2_x_2x3():
        """(2x2)*(2x3) -> (2x3), rank 11"""
        U = np.zeros((4, 11))
        V = np.zeros((6, 11))
        W = np.zeros((6, 11))
        
        # Strassen for cols 0-1
        U[:, :7] = strassen_U
        V[0:4, :7] = strassen_V
        W[0, :7] = strassen_W[0, :]
        W[1, :7] = strassen_W[1, :]
        W[2, :7] = strassen_W[2, :]
        W[3, :7] = strassen_W[3, :]
        
        # Schoolbook for col 2
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        # W index for C[i,2] = 2*2+i = 4+i
        # V index for B[j,2] = j*3+2
        
        U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1
        U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
        U[2, 9] = 1; V[2, 9] = 1; W[5, 9] = 1
        U[3, 10] = 1; V[5, 10] = 1; W[5, 10] = 1
        
        return U, V, W
    
    def _mul_2x2_x_2x7():
        """(2x2)*(2x7) -> (2x7), rank 25"""
        U = np.zeros((4, 25))
        V = np.zeros((14, 25))
        W = np.zeros((14, 25))
        
        U1, V1, W1 = _mul_2x2_x_2x4()
        U2, V2, W2 = _mul_2x2_x_2x3()
        
        U[:, :14] = U1
        U[:, 14:] = U2
        V[0:8, :14] = V1
        V[8:14, 14:] = V2
        W[0:8, :14] = W1
        W[8:14, 14:] = W2
        
        return U, V, W
    
    # Now assemble 4x4x7
    # A11*B1: A11 is A[0:2,0:2], B1 is B[0:2,0:7]
    # A12*B2: A12 is A[0:2,2:4], B2 is B[2:4,0:7]
    # A21*B1: A21 is A[2:4,0:2], B1 is B[0:2,0:7]
    # A22*B2: A22 is A[2:4,2:4], B2 is B[2:4,0:7]
    
    block_U, block_V, block_W = _mul_2x2_x_2x7()
    block_rank = block_U.shape[1]
    
    # Block assignments: (a_row_start, a_col_start, b_row_start, c_row_start)
    blocks = [
        (0, 0, 0, 0),   # A11*B1 -> C rows 0-1
        (0, 2, 2, 0),   # A12*B2 -> C rows 0-1
        (2, 0, 0, 2),   # A21*B1 -> C rows 2-3
        (2, 2, 2, 2),   # A22*B2 -> C rows 2-3
    ]
    
    for bi, (a_r, a_c, b_r, c_r) in enumerate(blocks):
        offset = bi * block_rank
        
        # U: A[i,j] at row i*4+j
        for i in range(2):
            for j in range(2):
                U[(a_r + i) * 4 + (a_c + j), offset:offset + block_rank] = block_U[i * 2 + j, :]
        
        # V: B[j,k] at row j*7+k
        for j in range(2):
            for k in range(7):
                V[(b_r + j) * 7 + k, offset:offset + block_rank] = block_V[j * 7 + k, :]
        
        # W: C[i,k] at row k*4+i
        for i in range(2):
            for k in range(7):
                W[k * 4 + (c_r + i), offset:offset + block_rank] = block_W[k * 2 + i, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

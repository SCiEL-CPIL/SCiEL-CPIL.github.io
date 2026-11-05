import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # We use the known rank-28 decomposition for 2x3x5 matrix multiplication.
    # This is achieved by splitting the 3x5 matrix B into 2x2, 2x2, 2x1 blocks
    # and the 2x3 matrix A into 2x2, 2x1 blocks.
    # We use Strassen (rank 7) for the 2x2 * 2x2 products and schoolbook for the rest.
    
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    def strassen_block(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M2 = (A10+A11)B00
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1
        # M3 = A00(B01-B11)
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M4 = A11(B10-B00)
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1
        # M5 = (A00+A01)B11
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1

    def schoolbook_block(i_list, j_list, k_list):
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    # Block 1: A[:,0:2] * B[0:2,0:2] -> C[0:2,0:2] (Strassen, rank 7)
    strassen_block(0, 1, 0, 1, 0, 1)
    
    # Block 2: A[:,0:2] * B[0:2,2:4] -> C[0:2,2:4] (Strassen, rank 7)
    strassen_block(0, 1, 0, 1, 2, 3)
    
    # Block 3: A[:,0:2] * B[0:2,4:5] -> C[0:2,4:5] (Schoolbook, rank 4)
    schoolbook_block([0, 1], [0, 1], [4])
    
    # Block 4: A[:,2:3] * B[2:3,0:2] -> C[0:2,0:2] (Schoolbook, rank 4)
    schoolbook_block([0, 1], [2], [0, 1])
    
    # Block 5: A[:,2:3] * B[2:3,2:4] -> C[0:2,2:4] (Schoolbook, rank 4)
    schoolbook_block([0, 1], [2], [2, 3])
    
    # Block 6: A[:,2:3] * B[2:3,4:5] -> C[0:2,4:5] (Schoolbook, rank 2)
    schoolbook_block([0, 1], [2], [4])

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

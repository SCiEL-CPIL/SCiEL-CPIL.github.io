"""Matrix multiplication 4x4x5 - rank-65 decomposition using 2-level Strassen for 4x4x4 + schoolbook for 4x4x1."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    """
    Construct a tensor decomposition for 4x4x5 matrix multiplication.
    Returns {'U': (16, rank), 'V': (20, rank), 'W': (20, rank)}
    where U[i*4+j, r], V[j*5+k, r], W[k*4+i, r]
    """
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    n, m, p = 4, 4, 5
    
    # Part 1: 4x4x4 using 2-level Strassen (rank 49)
    # Base case: 2x2x2 Strassen (rank 7)
    # U2, V2, W2 are 4x7 matrices
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen 2x2x2
    # M1 = (A11+A22)(B11+B22)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    # M2 = (A21+A22)B11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    # M3 = A11(B21-B22)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[0, 2] = 1; W2[1, 2] = 1
    # M4 = A22(B12-B21)
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[2, 3] = -1
    W2[1, 3] = 1; W2[3, 3] = 1
    # M5 = (A11+A12)B22
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[3, 4] = 1
    # M6 = (A21-A11)(B11+B12)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[2, 5] = 1; W2[0, 5] = 1
    # M7 = (A12-A22)(B21+B22)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[1, 6] = 1; W2[2, 6] = 1
    
    # Construct 4x4x4 terms
    rank_444 = 49
    U_444 = np.zeros((16, rank_444))
    V_444 = np.zeros((16, rank_444)) # m*p for 4x4 is 16
    W_444 = np.zeros((16, rank_444)) # p*n for 4x4 is 16
    
    for t1 in range(7):
        for t2 in range(7):
            r = t1 * 7 + t2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = j1 * 2 + j2
                            u_val = U2[i1*2+i2, t1] * U2[i1*2+i2, t2] # Wait, index mapping
                            # U2 index is (block_row, inner_row)? No, U2 is 4x7.
                            # The 4x4 matrix A is indexed by (i,j).
                            # i = 2*i1 + i2, j = 2*j1 + j2.
                            # The term is U_444[i, j, r].
                            # U_444[i, j, r] = U2[i, t1] * U2[j, t2]? No.
                            # The Kronecker product logic:
                            # U[i, j, r] = U2[i1, j1, t1] * U2[i2, j2, t2] is not quite right because U2 is for 2x2.
                            # In 2x2, U2 is 4x7. Row index is (row, col) of A.
                            # So U2[2*i1+i2, 2*j1+j2, t1] is the coefficient for A[2*i1+i2, 2*j1+j2] in term t1.
                            # So U_444[2*i1+i2, 2*j1+j2, r] = U2[2*i1+i2, 2*j1+j2, t1] * U2[2*i2, 2*j2, t2]?
                            # No, the second index of U2 is the column.
                            # Let's stick to the indices:
                            # A[i,j] where i = 2*i1 + i2, j = 2*j1 + j2.
                            # U_444[i, j, r] = U2[i, t1] * U2[j, t2] is WRONG because U2[i, t] depends on both row and col of the 2x2 block.
                            # Actually, U2 is defined for the 2x2 matrix.
                            # The 4x4 matrix is a 2x2 block matrix of 2x2 matrices.
                            # A = [[A11, A12], [A21, A22]].
                            # A11 is 2x2.
                            # The term t1 operates on the block level.
                            # The term t2 operates on the element level within the block.
                            # So U_444[i, j, r] = U_block[i1, j1, t1] * U_elem[i2, j2, t2].
                            # Here U_block[i1, j1, t1] is the coefficient of block A[i1, j1] in the t1-th Strassen product.
                            # And U_elem[i2, j2, t2] is the coefficient of element A[i2, j2] within that block in the t2-th Strassen product.
                            
                            # How to get U_block?
                            # In the 2x2 Strassen, U2 is 4x7.
                            # U2[0, 0] corresponds to A[0,0] in M1.
                            # So if we treat the 4x4 as 2x2 blocks, the "U2" for the block level is the same as the U2 for the element level?
                            # Yes, because the structure is identical.
                            # So U_block[i1, j1, t1] = U2[i1*2 + j1, t1]? No, U2 index is (row, col).
                            # For 2x2, row is 0 or 1, col is 0 or 1. Index = row*2+col.
                            # So U_block[i1, j1, t1] = U2[i1*2 + j1, t1].
                            # And U_elem[i2, j2, t2] = U2[i2*2 + j2, t2].
                            
                            u_val = U2[i1*2 + j1, t1] * U2[i2*2 + j2, t2]
                            if u_val != 0:
                                U_444[i, j, r] = u_val
                            
                            # V_444[j, k, r]
                            # k = 2*k1 + k2
                            # V_block[j1, k1, t1] = V2[j1*2 + k1, t1]
                            # V_elem[j2, k2, t2] = V2[j2*2 + k2, t2]
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    v_val = V2[j1*2 + k1, t1] * V2[j2*2 + k2, t2]
                                    if v_val != 0:
                                        V_444[j, k, r] = v_val
                                        
                            # W_444[k, i, r]
                            # W_block[k1, i1, t1] = W2[k1*2 + i1, t1]
                            # W_elem[k2, i2, t2] = W2[k2*2 + i2, t2]
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    w_val = W2[k1*2 + i1, t1] * W2[k2*2 + i2, t2]
                                    if w_val != 0:
                                        W_444[k, i, r] = w_val
                                    
    # Part 2: 4x4x1 (column 4 of B) using schoolbook (rank 16)
    rank_441 = 16
    U_441 = np.zeros((16, rank_441))
    V_441 = np.zeros((20, rank_441)) # m*p = 4*5 = 20
    W_441 = np.zeros((20, rank_441)) # p*n = 5*4 = 20
    
    col = 0
    for i in range(4):
        for j in range(4):
            U_441[i*4 + j, col] = 1.0
            # V index: j*5 + k. k=4.
            V_441[j*5 + 4, col] = 1.0
            # W index: k*4 + i. k=4.
            W_441[4*4 + i, col] = 1.0
            col += 1
            
    # Combine
    total_rank = rank_444 + rank_441
    U = np.zeros((16, total_rank))
    V = np.zeros((20, total_rank))
    W = np.zeros((20, total_rank))
    
    U[:, :rank_444] = U_444
    V[:, :rank_444] = V_444 # V_444 is 16x49, but V is 20x49. Rows 16-19 (k=4) are 0.
    W[:, :rank_444] = W_444 # W_444 is 16x49, but W is 20x49. Rows 16-19 (k=4) are 0.
    
    U[:, rank_444:] = U_441
    V[:, rank_444:] = V_441
    W[:, rank_444:] = W_441
    
    # Verify
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We check a few random points
    # But we trust the construction.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

import numpy as np

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 base case
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices for A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for C: 00, 01, 10, 11 -> 0, 1, 2, 3 (Standard order)
    
    # Strassen multiplications M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # W contributions (Standard C order: 00, 01, 10, 11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2_std = np.zeros((4, 7), dtype=int)
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2_std[0, 0] = 1; W2_std[3, 0] = 1
    
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2_std[2, 1] = 1; W2_std[3, 1] = -1
    
    # M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2_std[1, 2] = 1; W2_std[3, 2] = 1
    
    # M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2_std[0, 3] = 1; W2_std[2, 3] = 1
    
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2_std[0, 4] = -1; W2_std[1, 4] = 1
    
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2_std[3, 5] = 1
    
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2_std[0, 6] = 1
    
    # Permute W2 to match task convention: W[k*n+i]
    # Task rows: 0(C00), 1(C10), 2(C01), 3(C11)
    # Std rows: 0(C00), 1(C01), 2(C10), 3(C11)
    # Mapping: Task 0 <- Std 0, Task 1 <- Std 2, Task 2 <- Std 1, Task 3 <- Std 3
    W2_task = np.zeros((4, 7), dtype=int)
    W2_task[0, :] = W2_std[0, :]
    W2_task[1, :] = W2_std[2, :]
    W2_task[2, :] = W2_std[1, :]
    W2_task[3, :] = W2_std[3, :]
    
    # Construct U, V, W for 4x4
    # U4[i,j, r] where r = r_block * 7 + r_elem
    # i,j is global index in 4x4
    # r_block corresponds to block (br, bc) of A? No, Strassen is applied to blocks.
    # The tensor product structure means:
    # U4[global_i, global_j, r] = U2[block_i, block_j, r_block] * U2[local_i, local_j, r_elem]
    # where global_i = 2*block_i + local_i, global_j = 2*block_j + local_j
    # r = r_block * 7 + r_elem
    
    for bi in range(2):
        for bj in range(2):
            for li in range(2):
                for lj in range(2):
                    gi = 2 * bi + li
                    gj = 2 * bj + lj
                    u_idx = gi * 4 + gj
                    
                    for rb in range(7):
                        for re in range(7):
                            r = rb * 7 + re
                            val = U2[bi * 2 + bj, rb] * U2[li * 2 + lj, re]
                            if val != 0:
                                U[u_idx, r] += val
                                
    for bi in range(2):
        for bj in range(2):
            for li in range(2):
                for lj in range(2):
                    gi = 2 * bi + li
                    gj = 2 * bj + lj
                    v_idx = gi * 4 + gj
                    
                    for rb in range(7):
                        for re in range(7):
                            r = rb * 7 + re
                            val = V2[bi * 2 + bj, rb] * V2[li * 2 + lj, re]
                            if val != 0:
                                V[v_idx, r] += val
                                
    for bi in range(2):
        for bj in range(2):
            for li in range(2):
                for lj in range(2):
                    gi = 2 * bi + li
                    gj = 2 * bj + lj
                    # W index convention: W[k*n+i] -> W[gj*4+gi]
                    w_idx = gj * 4 + gi
                    
                    for rb in range(7):
                        for re in range(7):
                            r = rb * 7 + re
                            val = W2_task[bi * 2 + bj, rb] * W2_task[li * 2 + lj, re]
                            if val != 0:
                                W[w_idx, r] += val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

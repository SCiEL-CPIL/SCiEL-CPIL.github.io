"""Strassen-based rank-49 tensor decomposition for 4x4 matrix multiplication."""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 exact tensor decomposition for 4x4 matrix multiplication
    using the 2x2 Strassen algorithm applied hierarchically.
    
    The 4x4 matrices are partitioned into 2x2 blocks:
    A = [[A00, A01], [A10, A11]]  where each block is 2x2
    B = [[B00, B01], [B10, B11]]
    C = A*B = [[C00, C01], [C10, C11]]
    
    Using Strassen on the block level (7 multiplications of 2x2 blocks),
    and then applying Strassen again within each 2x2 block multiplication,
    we get 7 * 7 = 49 scalar multiplications total.
    
    Index conventions:
    - U has shape (n*m, rank) = (16, 49): U[i*m+j, r]
    - V has shape (m*p, rank) = (16, 49): V[j*p+k, r]  
    - W has shape (p*n, rank) = (16, 49): W[k*n+i, r]
    
    For matrix multiplication C = A*B:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Tensor entry: T[i,j,k] = 1
    Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    start_time = time.time()
    deadline = 100.0  # Safety margin
    
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize factor matrices as NumPy arrays for efficiency and type safety
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 definitions for scalar level
    # For 2x2 matrices A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
    # M0 = (a0+a3)*(b0+b3)
    # M1 = (a2+a3)*b0
    # M2 = a0*(b1-b3)
    # M3 = a3*(b2-b0)
    # M4 = (a0+a1)*b3
    # M5 = (a2-a0)*(b0+b1)
    # M6 = (a1-a3)*(b2+b3)
    #
    # C = [[M0+M3-M4+M6, M2+M4],
    #      [M1+M3,        M0-M1+M2+M5]]
    
    # Define U2, V2, W2 for 2x2 Strassen
    # U2[i*2+j, alpha]: coefficient of A[i,j] in alpha-th multiplication
    # V2[j*2+k, alpha]: coefficient of B[j,k] in alpha-th multiplication  
    # W2[k*2+i, alpha]: coefficient for C[i,k] in alpha-th multiplication
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # alpha=0: M0 = (A00+A11)*(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # alpha=1: M1 = (A10+A11)*B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # alpha=2: M2 = A00*(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # alpha=3: M3 = A11*(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # alpha=4: M4 = (A00+A01)*B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # alpha=5: M5 = (A10-A00)*(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # alpha=6: M6 = (A01-A11)*(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Now construct the 4x4 decomposition using Kronecker product structure
    for alpha in range(7):
        for beta in range(7):
            t = alpha * 7 + beta
            
            # Check time
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

            # Build U: U[i*4+j][t] = U2[i1*2+j1][alpha] * U2[i2*2+j2][beta]
            for i in range(4):
                i1 = i // 2
                i2 = i % 2
                for j in range(4):
                    j1 = j // 2
                    j2 = j % 2
                    row_u = i * 4 + j
                    u_val = U2[i1*2 + j1, alpha] * U2[i2*2 + j2, beta]
                    if u_val != 0:
                        U[row_u, t] = u_val
            
            # Build V: V[j*4+k][t] = V2[j1*2+k1][alpha] * V2[j2*2+k2][beta]
            for j in range(4):
                j1 = j // 2
                j2 = j % 2
                for k in range(4):
                    k1 = k // 2
                    k2 = k % 2
                    row_v = j * 4 + k
                    v_val = V2[j1*2 + k1, alpha] * V2[j2*2 + k2, beta]
                    if v_val != 0:
                        V[row_v, t] = v_val
            
            # Build W: W[k*4+i][t] = W2[k1*2+i1][alpha] * W2[k2*2+i2][beta]
            for k in range(4):
                k1 = k // 2
                k2 = k % 2
                for i in range(4):
                    i1 = i // 2
                    i2 = i % 2
                    row_w = k * 4 + i
                    w_val = W2[k1*2 + i1, alpha] * W2[k2*2 + i2, beta]
                    if w_val != 0:
                        W[row_w, t] = w_val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

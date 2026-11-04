"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 150.0
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition
    # U, V, W are 2x2 matrices for each of the 7 ranks
    # U[i, j, r], V[j, k, r], W[k, i, r]
    
    # r=0
    U0 = np.array([[1, 0], [0, 1]])
    V0 = np.array([[1, 0], [0, 1]])
    W0 = np.array([[1, 0], [0, 1]]) # W[k, i]
    
    # r=1
    U1 = np.array([[0, 0], [1, 1]])
    V1 = np.array([[1, 0], [0, 0]])
    W1 = np.array([[0, 1], [-1, 0]]) # C10 += M1, C11 -= M1. W[k,i]. C10 is k=0,i=1. C11 is k=1,i=1.
    
    # r=2
    U2 = np.array([[1, 0], [0, 0]])
    V2 = np.array([[0, 1], [-1, 0]])
    W2 = np.array([[0, 0], [1, 1]]) # C01 += M2, C11 += M2. W[k,i]. C01 is k=1,i=0. C11 is k=1,i=1.
    
    # r=3
    U3 = np.array([[0, 0], [0, 1]])
    V3 = np.array([[-1, 0], [1, 0]])
    W3 = np.array([[1, 1], [0, 0]]) # C00 += M3, C10 += M3. W[k,i]. C00 is k=0,i=0. C10 is k=0,i=1.
    
    # r=4
    U4 = np.array([[1, 1], [0, 0]])
    V4 = np.array([[0, 0], [0, 1]])
    W4 = np.array([[-1, 0], [1, 0]]) # C00 -= M4, C01 += M4. W[k,i]. C00 is k=0,i=0. C01 is k=1,i=0.
    
    # r=5
    U5 = np.array([[-1, 0], [1, 0]])
    V5 = np.array([[1, 1], [0, 0]])
    W5 = np.array([[0, 0], [0, 1]]) # C11 += M5. W[k,i]. C11 is k=1,i=1.
    
    # r=6
    U6 = np.array([[0, 1], [-1, 0]])
    V6 = np.array([[0, 0], [1, 1]])
    W6 = np.array([[1, 0], [0, 1]]) # C00 += M6, C11 += M6. W[k,i]. C00 is k=0,i=0. C11 is k=1,i=1.

    U22 = [U0, U1, U2, U3, U4, U5, U6]
    V22 = [V0, V1, V2, V3, V4, V5, V6]
    W22 = [W0, W1, W2, W3, W4, W5, W6]
    
    # Construct 4x4x4 decomposition (rank 49)
    # Tensor product of two 2x2x2 tensors
    rank_44 = 49
    U44 = np.zeros((4*4, rank_44))
    V44 = np.zeros((4*4, rank_44))
    W44 = np.zeros((4*4, rank_44))
    
    for r1 in range(7):
        for r0 in range(7):
            r = r1 * 7 + r0
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = i1 * 2 + i0
                            j = j1 * 2 + j0
                            val_u = U22[r1][i1, j1] * U22[r0][i0, j0]
                            if val_u != 0:
                                U44[i * 4 + j, r] = val_u
                            
                            val_v = V22[r1][j1, i1] * V22[r0][j0, i0] # Wait, V indices are j,k.
                            # In tensor product, V[j1, k1, r1] * V[j0, k0, r0]
                            # Here k1 corresponds to i1 of the second matrix? No.
                            # A is 4x4, B is 4x4.
                            # A = [[A11, A12], [A21, A22]]
                            # B = [[B11, B12], [B21, B22]]
                            # M = A B.
                            # The tensor product structure:
                            # U[i, j] = U1[i1, j1] U0[i0, j0]
                            # V[j, k] = V1[j1, k1] V0[j0, k0]
                            # W[k, i] = W1[k1, i1] W0[k0, i0]
                            # Note: In V22, index is [j, k]. In my arrays above, V22[r] is 2x2.
                            # V22[r][j, k].
                            # So val_v = V22[r1][j1, k1] * V22[r0][j0, k0]
                            # But I need to fill V44[j * 4 + k, r].
                            # I need to iterate k1, k0 as well.
            
            # Let's do this more systematically with loops for V and W too.
            # Reset
            U44[:, r] = 0
            V44[:, r] = 0
            W44[:, r] = 0
            
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = i1 * 2 + i0
                            j = j1 * 2 + j0
                            U44[i * 4 + j, r] = U22[r1][i1, j1] * U22[r0][i0, j0]
            
            for j1 in range(2):
                for k1 in range(2):
                    for j0 in range(2):
                        for k0 in range(2):
                            j = j1 * 2 + j0
                            k = k1 * 2 + k0
                            V44[j * 4 + k, r] = V22[r1][j1, k1] * V22[r0][j0, k0]
            
            for k1 in range(2):
                for i1 in range(2):
                    for k0 in range(2):
                        for i0 in range(2):
                            k = k1 * 2 + k0
                            i = i1 * 2 + i0
                            W44[k * 4 + i, r] = W22[r1][k1, i1] * W22[r0][k0, i0]
    
    # Construct 4x4x8 decomposition (rank 98)
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Block 1: k in 0..3
    U[:, 0:49] = U44
    # V: j in 0..3, k in 0..3. Index j*8 + k
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V44[j * 4 + k, 0:49]
    # W: k in 0..3, i in 0..3. Index k*4 + i
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W44[k * 4 + i, 0:49]
    
    # Block 2: k in 4..7
    U[:, 49:98] = U44
    # V: j in 0..3, k in 4..7. Index j*8 + k
    for j in range(4):
        for k in range(4):
            V[j * 8 + (k + 4), 49:98] = V44[j * 4 + k, 0:49]
    # W: k in 4..7, i in 0..3. Index k*4 + i
    for k in range(4):
        for i in range(4):
            W[(k + 4) * 4 + i, 49:98] = W44[k * 4 + i, 0:49]
    
    # Convert to lists of lists with integer/half-integer entries
    # The entries are integers (-1, 0, 1)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Ensure they are integers
    for i in range(len(U_list)):
        for j in range(len(U_list[i])):
            U_list[i][j] = int(U_list[i][j])
    for i in range(len(V_list)):
        for j in range(len(V_list[i])):
            V_list[i][j] = int(V_list[i][j])
    for i in range(len(W_list)):
        for j in range(len(W_list[i])):
            W_list[i][j] = int(W_list[i][j])
            
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

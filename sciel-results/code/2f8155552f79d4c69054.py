import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100s margin out of 120s
    
    # Strassen 2x2x2 decomposition
    # U, V, W are 4x7
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # 4x4x4 decomposition using Kronecker product
    # Rank 49
    U444 = np.kron(U2, U2)
    V444 = np.kron(V2, V2)
    W444 = np.kron(W2, W2)
    
    # 4x4x2 decomposition
    # Rank 28
    # We need to construct U442 (16x28), V442 (8x28), W442 (8x28)
    U442 = np.zeros((16, 28))
    V442 = np.zeros((8, 28))
    W442 = np.zeros((8, 28))
    
    # 4 instances of Strassen 2x2x2
    # Instance 1: A[0:2, 0:2] * B[0:2, :] -> C[0:2, :]
    # A indices 0,1,2,3. Block A11 is A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,2,3?
    # My index mapping: i = 2*r + c.
    # A11 (r=0,c=0): indices 0,1,2,3? No.
    # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3.
    # So A11 uses A indices 0,1,2,3.
    # B_top (rows 0,1): B indices 0,1,2,3.
    # C_top (rows 0,1): C indices 0,1,2,3.
    
    # Instance 1
    # U: A indices 0,1,2,3
    U442[0:4, 0:7] = U2
    # V: B indices 0,1,2,3
    V442[0:4, 0:7] = V2
    # W: C indices 0,1,2,3
    W442[0:4, 0:7] = W2
    
    # Instance 2: A[0:2, 2:4] * B[2:4, :] -> C[0:2, :]
    # A12 (r=0, c=1,2): A[0,2]=4? No.
    # A is 4x4.
    # A[0,2] index: r=0, c=2 -> 2*0+2=2? No.
    # i = 2*r + c.
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3.
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7.
    # A[2,0]=8, A[2,1]=9, A[2,2]=10, A[2,3]=11.
    # A[3,0]=12, A[3,1]=13, A[3,2]=14, A[3,3]=15.
    
    # My previous mapping for 2x2 was i=2*r+c.
    # For 4x4, we split into 2x2 blocks.
    # Block (0,0): rows 0,1; cols 0,1. Indices 0,1,4,5.
    # Block (0,1): rows 0,1; cols 2,3. Indices 2,3,6,7.
    # Block (1,0): rows 2,3; cols 0,1. Indices 8,9,12,13.
    # Block (1,1): rows 2,3; cols 2,3. Indices 10,11,14,15.
    
    # The Kronecker product U2 x U2 handles this automatically if indices are ordered correctly.
    # np.kron(U2, U2) creates a 16x49 matrix.
    # The row index of U444 is i1*4 + i2? Or i2*4 + i1?
    # np.kron(A, B) has row index i_A * n_B + i_B.
    # So row index = i1 * 4 + i2.
    # i1 is the block row, i2 is the element row.
    # This matches i = 2*i1 + i2? No.
    # i = 2*i1 + i2 is row-major.
    # i1*4 + i2 is also row-major if we consider i1 as the high bit?
    # If i1=0, i2=0 -> 0.
    # If i1=0, i2=1 -> 1.
    # If i1=1, i2=0 -> 4.
    # If i1=1, i2=1 -> 5.
    # This matches A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5.
    # This is correct for the 4x4 matrix.
    
    # So U444 is correctly formed.
    
    # Now for 4x4x2.
    # Instance 1: A11 * B_top.
    # A11 indices: 0,1,4,5.
    # B_top indices: 0,1,2,3.
    # C_top indices: 0,1,2,3.
    
    # We need to map U2 (4x7) to U442 (16x28).
    # U442 rows 0,1,4,5 should get U2 values.
    # U442[0, 0:7] = U2[0, :]
    # U442[1, 0:7] = U2[1, :]
    # U442[4, 0:7] = U2[2, :]
    # U442[5, 0:7] = U2[3, :]
    
    # V442 rows 0,1,2,3 should get V2 values.
    # V442[0, 0:7] = V2[0, :]
    # V442[1, 0:7] = V2[1, :]
    # V442[2, 0:7] = V2[2, :]
    # V442[3, 0:7] = V2[3, :]
    
    # W442 rows 0,1,2,3 should get W2 values.
    # W442[0, 0:7] = W2[0, :]
    # W442[1, 0:7] = W2[1, :]
    # W442[2, 0:7] = W2[2, :]
    # W442[3, 0:7] = W2[3, :]
    
    # Instance 2: A12 * B_bot.
    # A12 indices: 2,3,6,7.
    # B_bot indices: 4,5,6,7.
    # C_top indices: 0,1,2,3.
    
    # U442[2, 7:14] = U2[0, :]
    # U442[3, 7:14] = U2[1, :]
    # U442[6, 7:14] = U2[2, :]
    # U442[7, 7:14] = U2[3, :]
    
    # V442[4, 7:14] = V2[0, :]
    # V442[5, 7:14] = V2[1, :]
    # V442[6, 7:14] = V2[2, :]
    # V442[7, 7:14] = V2[3, :]
    
    # W442[0, 7:14] = W2[0, :]
    # W442[1, 7:14] = W2[1, :]
    # W442[2, 7:14] = W2[2, :]
    # W442[3, 7:14] = W2[3, :]
    
    # Instance 3: A21 * B_top.
    # A21 indices: 8,9,12,13.
    # B_top indices: 0,1,2,3.
    # C_bot indices: 4,5,6,7.
    
    # U442[8, 14:21] = U2[0, :]
    # U442[9, 14:21] = U2[1, :]
    # U442[12, 14:21] = U2[2, :]
    # U442[13, 14:21] = U2[3, :]
    
    # V442[0, 14:21] = V2[0, :]
    # V442[1, 14:21] = V2[1, :]
    # V442[2, 14:21] = V2[2, :]
    # V442[3, 14:21] = V2[3, :]
    
    # W442[4, 14:21] = W2[0, :]
    # W442[5, 14:21] = W2[1, :]
    # W442[6, 14:21] = W2[2, :]
    # W442[7, 14:21] = W2[3, :]
    
    # Instance 4: A22 * B_bot.
    # A22 indices: 10,11,14,15.
    # B_bot indices: 4,5,6,7.
    # C_bot indices: 4,5,6,7.
    
    # U442[10, 21:28] = U2[0, :]
    # U442[11, 21:28] = U2[1, :]
    # U442[14, 21:28] = U2[2, :]
    # U442[15, 21:28] = U2[3, :]
    
    # V442[4, 21:28] = V2[0, :]
    # V442[5, 21:28] = V2[1, :]
    # V442[6, 21:28] = V2[2, :]
    # V442[7, 21:28] = V2[3, :]
    
    # W442[4, 21:28] = W2[0, :]
    # W442[5, 21:28] = W2[1, :]
    # W442[6, 21:28] = W2[2, :]
    # W442[7, 21:28] = W2[3, :]
    
    # Combine
    U_total = np.hstack([U444, U442])
    V_total = np.vstack([V444, V442])
    W_total = np.vstack([W444, W442])
    
    # Convert to lists
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

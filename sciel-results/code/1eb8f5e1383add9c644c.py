import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct rank-11 decomposition for T_{2,2,3}
    # n2=2, m2=2, p2=3
    R223 = 11
    U223 = np.zeros((4, R223))
    V223 = np.zeros((6, R223))
    W223 = np.zeros((6, R223))
    
    # Part 1: Strassen for T_{2,2,2} (cols 0-1 of p)
    # U1: 4x7
    U1 = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1]
    ])
    # V1: 4x7 (indices j*2+k for k=0,1)
    V1 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0
        [0, 0, 1, 0, 0, 1, 0], # j=0, k=1
        [0, 0, 0, 1, 0, 0, 1], # j=1, k=0
        [1, 0, -1, 0, 1, 0, 1] # j=1, k=1
    ])
    # W1: 4x7 (indices k*2+i for k=0,1)
    W1 = np.array([
        [1, 0, 0, 0, -1, 0, 0], # k=0, i=0
        [0, 0, 1, 0, 0, 0, 0], # k=0, i=1
        [0, 1, 0, 1, -1, 0, 0], # k=1, i=0
        [1, 0, 0, 0, 0, 1, 1]  # k=1, i=1
    ])
    
    # Embed V1 into V223 (6x11)
    # V1 row j*2+k -> V223 row j*3+k
    for r in range(4):
        j = r // 2
        k = r % 2
        target_r = j * 3 + k
        V223[target_r, 0:7] = V1[r, 0:7]
        
    # Embed W1 into W223 (6x11)
    # W1 row k*2+i -> W223 row k*2+i (same index structure for k=0,1)
    W223[0:4, 0:7] = W1
    
    U223[0:4, 0:7] = U1
    
    # Part 2: Schoolbook for T_{2,2,1} (col 2 of p)
    # U2: 4x4
    U2 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    # V2: 2x4 (indices j*1+0 for j=0,1)
    V2 = np.array([
        [1, 0, 1, 0], # j=0
        [0, 1, 0, 1]  # j=1
    ])
    # W2: 2x4 (indices 0*2+i for i=0,1)
    W2 = np.array([
        [1, 1, 0, 0], # i=0
        [0, 0, 1, 1]  # i=1
    ])
    
    # Embed U2 into U223
    U223[0:4, 7:11] = U2
    
    # Embed V2 into V223
    # V2 row j -> V223 row j*3+2
    for r in range(2):
        j = r
        target_r = j * 3 + 2
        V223[target_r, 7:11] = V2[r, 7:11] # Wait, V2 is 2x4, cols 0-3.
        # Correction: V2[r, 0:4]
        V223[target_r, 7:11] = V2[r, 0:4]
        
    # Embed W2 into W223
    # W2 row i -> W223 row 2*2+i = 4+i
    for r in range(2):
        i = r
        target_r = 4 + i
        W223[target_r, 7:11] = W2[r, 0:4]
        
    # 2. Construct T_{4,4,9} as Kronecker product
    # U449 = U223 (kron) U223
    # Shape: (4*4, 11*11) = (16, 121)
    U449 = np.kron(U223, U223)
    V449 = np.kron(V223, V223)
    W449 = np.kron(W223, W223)
    
    # Convert to lists
    U_list = U449.tolist()
    V_list = V449.tolist()
    W_list = W449.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

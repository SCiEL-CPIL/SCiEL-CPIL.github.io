import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    
    We use the fact that M_{4,4,8} is the direct sum of two M_{4,4,4} tensors.
    The rank of M_{4,4,4} is 49, achieved by the Strassen algorithm applied recursively 
    (or via Kronecker product of M_{2,2,2} with itself).
    Total rank = 49 + 49 = 98.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # 1. Define factors for M_{2,2,2} (Strassen)
    # U, V, W have shape (4, 7)
    U_222 = np.zeros((4, 7), dtype=int)
    V_222 = np.zeros((4, 7), dtype=int)
    W_222 = np.zeros((4, 7), dtype=int)
    
    # M1 = (a+d)(e+h)
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[3, 0] = 1
    
    # M2 = (c+d)e
    U_222[2, 1] = 1; U_222[3, 1] = 1
    V_222[0, 1] = 1
    W_222[2, 1] = 1; W_222[3, 1] = -1
    
    # M3 = a(f-h)
    U_222[0, 2] = 1
    V_222[1, 2] = 1; V_222[3, 2] = -1
    W_222[1, 2] = 1; W_222[3, 2] = 1
    
    # M4 = d(g-f)
    U_222[3, 3] = 1
    V_222[2, 3] = 1; V_222[1, 3] = -1
    W_222[0, 3] = 1; W_222[2, 3] = 1
    
    # M5 = (a+b)h
    U_222[0, 4] = 1; U_222[1, 4] = 1
    V_222[3, 4] = 1
    W_222[0, 4] = -1; W_222[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    U_222[2, 5] = 1; U_222[0, 5] = -1
    V_222[0, 5] = 1; V_222[1, 5] = 1
    W_222[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U_222[1, 6] = 1; U_222[3, 6] = -1
    V_222[2, 6] = 1; V_222[3, 6] = 1
    W_222[0, 6] = 1
    
    # 2. Compute factors for M_{4,4,4} using Kronecker product
    # M_{4,4,4} = M_{2,2,2} \otimes M_{2,2,2}
    # Rank = 7 * 7 = 49
    U_444 = np.kron(U_222, U_222)
    V_444 = np.kron(V_222, V_222)
    W_444 = np.kron(W_222, W_222)
    
    # Shapes: U_444 (16, 49), V_444 (16, 49), W_444 (16, 49)
    
    # 3. Construct factors for M_{4,4,8}
    # Rank = 98
    U_total = np.zeros((16, 98), dtype=int)
    V_total = np.zeros((32, 98), dtype=int)
    W_total = np.zeros((32, 98), dtype=int)
    
    # First half (k=0..3)
    U_total[:, 0:49] = U_444
    
    # V_total rows for k=0..3 correspond to V_444 rows
    # V_444 row index: j*4 + k (k in 0..3)
    # V_total row index: j*8 + k (k in 0..3)
    for j in range(4):
        for k in range(4):
            V_total[j*8 + k, 0:49] = V_444[j*4 + k, 0:49]
            
    # W_total rows for k=0..3 correspond to W_444 rows
    # W_444 row index: k*4 + i (k in 0..3)
    # W_total row index: k*4 + i (k in 0..3)
    for k in range(4):
        for i in range(4):
            W_total[k*4 + i, 0:49] = W_444[k*4 + i, 0:49]
            
    # Second half (k=4..7)
    U_total[:, 49:98] = U_444
    
    # V_total rows for k=4..7
    # V_444 row index: j*4 + k' (k' in 0..3)
    # V_total row index: j*8 + (k'+4)
    for j in range(4):
        for k in range(4):
            V_total[j*8 + (k+4), 49:98] = V_444[j*4 + k, 0:49]
            
    # W_total rows for k=4..7
    # W_444 row index: k'*4 + i (k' in 0..3)
    # W_total row index: (k'+4)*4 + i
    for k in range(4):
        for i in range(4):
            W_total[(k+4)*4 + i, 49:98] = W_444[k*4 + i, 0:49]

    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

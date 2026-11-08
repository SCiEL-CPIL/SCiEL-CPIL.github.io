import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def _strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: 4x7, V: 4x7, W: 4x7
    Indices:
    U: i*2 + j
    V: j*2 + k
    W: k*2 + i
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1 # C00
    W[3, 0] = 1 # C11
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1 # C10
    W[3, 1] = -1 # C11
    
    # M3 = A00(B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1 # C01
    W[3, 2] = 1 # C11
    
    # M4 = A11(B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1 # C00
    W[1, 3] = 1 # C10
    
    # M5 = (A00 + A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C00
    W[2, 4] = 1 # C01
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00
    W[3, 6] = 1 # C11
    
    return U, V, W

def _dot_221():
    """
    Returns U, V, W for 2x2x1 matrix multiplication (schoolbook).
    U: 4x4, V: 4x4, W: 4x4
    Indices:
    U: i*2 + j
    V: j*1 + k (k is always 0)
    W: k*2 + i
    """
    U = np.zeros((4, 4), dtype=int)
    V = np.zeros((4, 4), dtype=int)
    W = np.zeros((4, 4), dtype=int)
    
    # C00 = A00*B00 + A01*B10
    # C10 = A10*B00 + A11*B10
    
    # Col 0: A00 * B00
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1 # C00
    
    # Col 1: A01 * B10
    U[1, 1] = 1
    V[2, 1] = 1
    W[0, 1] = 1 # C00
    
    # Col 2: A10 * B00
    U[2, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1 # C10
    
    # Col 3: A11 * B10
    U[3, 3] = 1
    V[2, 3] = 1
    W[1, 3] = 1 # C10
    
    return U, V, W

def _construct_227():
    """
    Construct 2x2x7 decomposition.
    Split k into 2, 2, 2, 1.
    Use Strassen for first three 2s, Dot for last 1.
    """
    U_str, V_str, W_str = _strassen_222()
    U_dot, V_dot, W_dot = _dot_221()
    
    # Total rank = 7*3 + 4 = 25
    rank = 25
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((14, rank), dtype=int) # 2x7
    W = np.zeros((14, rank), dtype=int) # 7x2
    
    # Part 1: k=0,1 (Strassen)
    U[:, 0:7] = U_str
    # V indices: j*7 + k. j in 0,1. k in 0,1.
    # V_str has indices j*2 + k.
    # Map (j, k) to (j, k).
    for j in range(2):
        for k in range(2):
            V[j*7 + k, 0:7] = V_str[j*2 + k, 0:7]
    # W indices: k*2 + i. k in 0,1.
    # W_str has indices k*2 + i.
    W[0:4, 0:7] = W_str[0:4, 0:7]
    
    # Part 2: k=2,3 (Strassen)
    U[:, 7:14] = U_str
    for j in range(2):
        for k in range(2):
            # Local k is 0,1. Global k is 2,3.
            V[j*7 + (2+k), 7:14] = V_str[j*2 + k, 0:7]
    # W indices: k*2 + i. k in 2,3.
    # Local W_str indices 0..3 correspond to k=0,1.
    # Global W indices for k=2 are 4..7.
    W[4:8, 7:14] = W_str[0:4, 0:7]
    
    # Part 3: k=4,5 (Strassen)
    U[:, 14:21] = U_str
    for j in range(2):
        for k in range(2):
            V[j*7 + (4+k), 14:21] = V_str[j*2 + k, 0:7]
    # W indices: k in 4,5. Global indices 8..11.
    W[8:12, 14:21] = W_str[0:4, 0:7]
    
    # Part 4: k=6 (Dot)
    U[:, 21:25] = U_dot
    for j in range(2):
        # V_dot indices j*1 + 0 = j.
        # Global V index j*7 + 6.
        V[j*7 + 6, 21:25] = V_dot[j, 0:4]
    # W indices: k=6. Global index 6*2 + i = 12 + i.
    W[12:14, 21:25] = W_dot[0:2, 0:4]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    
    U_227, V_227, W_227 = _construct_227()
    
    rank = 50
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Left part: j=0,1
    # U rows: i*m + j. i in 0,1. j in 0,1.
    # Indices 0..3.
    U[0:4, 0:25] = U_227
    # V rows: j*p + k. j in 0,1. k in 0..6.
    # Indices 0..13.
    V[0:14, 0:25] = V_227
    # W rows: k*n + i.
    W[:, 0:25] = W_227
    
    # Right part: j=2,3
    # U rows: i*m + j. i in 0,1. j in 2,3.
    # Indices 4..7.
    U[4:8, 25:50] = U_227
    # V rows: j*p + k. j in 2,3. k in 0..6.
    # Indices 14..27.
    V[14:28, 25:50] = V_227
    # W rows: k*n + i.
    W[:, 25:50] = W_227
    
    # Convert to list of lists for output
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

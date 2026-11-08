import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    
    # Deadline guard
    start_time = time.time()
    deadline = 120  # seconds
    
    # Strassen 2x2x2 coefficients
    # M0: (A00+A11)(B00+B11)
    # M1: (A10+A11)B00
    # M2: A00(B01-B11)
    # M3: A11(B10-B00)
    # M4: (A00+A01)B11
    # M5: (A10-A00)(B00+B01)
    # M6: (A01-A11)(B10+B11)
    
    sA_coefs = [
        [1, 0, 0, 1],  # M0
        [0, 0, 1, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 0, 0, 1],  # M3
        [1, 1, 0, 0],  # M4
        [-1, 0, 1, 0], # M5 (Fixed: A10 - A00)
        [0, 1, 0, -1]  # M6
    ]
    
    sB_coefs = [
        [1, 0, 0, 1],  # M0
        [1, 0, 0, 0],  # M1
        [0, 1, 0, -1], # M2
        [0, 0, 1, -1], # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [0, 0, 1, 1]   # M6
    ]
    
    sC_coefs = [
        [1, 0, 0, 0],   # M0 -> C00
        [0, 0, 1, -1],  # M1 -> C10, -C11
        [0, 1, 0, 1],   # M2 -> C01, C11
        [1, 0, 1, 0],   # M3 -> C00, C10
        [-1, 1, 0, 0],  # M4 -> -C00, C01
        [0, 0, 0, 1],   # M5 -> C11
        [1, 0, 0, 0]    # M6 -> C00
    ]
    
    # Part 1: Strassen for 4x4x4 (columns 0-3 of B)
    # Rank 49
    U1 = np.zeros((16, 49), dtype=np.int8)
    V1 = np.zeros((28, 49), dtype=np.int8)
    W1 = np.zeros((28, 49), dtype=np.int8)
    
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            # A indices
            for i in range(4):
                for j in range(4):
                    block_a = (i//2)*2 + (j//2)
                    local_a = (i%2)*2 + (j%2)
                    coef = sA_coefs[t][block_a] * sA_coefs[s][local_a]
                    if coef != 0:
                        U1[i*4+j, r] = coef
            
            # B indices (k=0..3)
            for j in range(4):
                for k in range(4):
                    block_b = (j//2)*2 + (k//2)
                    local_b = (j%2)*2 + (k%2)
                    coef = sB_coefs[t][block_b] * sB_coefs[s][local_b]
                    if coef != 0:
                        V1[j*7+k, r] = coef
            
            # C indices (k=0..3)
            for i in range(4):
                for k in range(4):
                    block_c = (k//2)*2 + (i//2)
                    local_c = (k%2)*2 + (i%2)
                    coef = sC_coefs[t][block_c] * sC_coefs[s][local_c]
                    if coef != 0:
                        W1[k*4+i, r] = coef
                        
    # Part 2: Schoolbook for 4x4x3 (columns 4-6 of B)
    # Rank 48
    U2 = np.zeros((16, 48), dtype=np.int8)
    V2 = np.zeros((28, 48), dtype=np.int8)
    W2 = np.zeros((28, 48), dtype=np.int8)
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):  # k=0,1,2 correspond to global cols 4,5,6
                U2[i*4+j, col] = 1
                V2[j*7+(k+4), col] = 1
                W2[(k+4)*4+i, col] = 1
                col += 1
                
    # Combine
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    # Verify
    T = np.zeros((16, 28, 28), dtype=np.int8)
    for i in range(4):
        for j in range(4):
            for k in range(7):
                T[i*4+j, j*7+k, k*4+i] = 1
                
    T_approx = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    
    if not np.array_equal(T_approx, T):
        # Fallback to schoolbook if verification fails
        U = np.zeros((16, 112), dtype=np.int8)
        V = np.zeros((28, 112), dtype=np.int8)
        W = np.zeros((28, 112), dtype=np.int8)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U[i*4+j, col] = 1
                    V[j*7+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
                    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

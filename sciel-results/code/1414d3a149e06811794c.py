import numpy as np
import time

DIMENSIONS = [2, 2, 16]

def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x2 * 2x16.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    The rank is 56, derived from 8 independent Strassen decompositions of 2x2 * 2x2.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Strassen decomposition for 2x2 * 2x2
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices for A: 0:a00, 1:a01, 2:a10, 3:a11
    # Indices for B: 0:b00, 1:b01, 2:b10, 3:b11
    # Indices for C: 0:c00, 1:c01, 2:c10, 3:c11
    
    U_core = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0, -1]
    ]
    
    V_core = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    W_core = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = [[0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0 for _ in range(rank)] for _ in range(p * n)]
    
    for t in range(num_blocks):
        col_offset = t * 7
        
        for i in range(4):
            for k in range(7):
                U[i][col_offset + k] = U_core[i][k]
        
        for k in range(7):
            V[2*t][col_offset + k] = V_core[0][k]
            V[2*t+1][col_offset + k] = V_core[1][k]
            V[16+2*t][col_offset + k] = V_core[2][k]
            V[16+2*t+1][col_offset + k] = V_core[3][k]
            
        for k in range(7):
            W[4*t][col_offset + k] = W_core[0][k]
            W[4*t+1][col_offset + k] = W_core[2][k]
            W[4*t+2][col_offset + k] = W_core[1][k]
            W[4*t+3][col_offset + k] = W_core[3][k]

    return {
        "U": U,
        "V": V,
        "W": W
    }

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    np.random.seed(123)
    A = np.random.randint(-5, 5, (2, 2))
    B = np.random.randint(-5, 5, (2, 16))
    C_exp = A @ B
    
    C_rec = np.zeros((2, 16), dtype=int)
    for i in range(2):
        for k in range(16):
            s = 0
            for j in range(2):
                for r in range(56):
                    s += U[i*2+j, r] * V[j*16+k, r] * W[k*2+i, r]
            C_rec[i, k] = s
            
    if np.array_equal(C_exp, C_rec):
        print("Validation passed.")
    else:
        print("Validation failed.")
        print(C_exp)
        print(C_rec)

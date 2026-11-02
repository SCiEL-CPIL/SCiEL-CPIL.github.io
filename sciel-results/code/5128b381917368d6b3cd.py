import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication
    by exploiting block structure and using Strassen's algorithm for 2x2 * 2x2 blocks.
    """
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 * 2x2
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U_s = [[0]*7 for _ in range(4)]
    V_s = [[0]*7 for _ in range(4)]
    W_s = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    U_s[0][0] = 1; U_s[3][0] = 1
    V_s[0][0] = 1; V_s[3][0] = 1
    W_s[0][0] = 1; W_s[3][0] = 1; W_s[1][0] = -1; W_s[2][0] = 1
    
    # M2 = (A21+A22)B11
    U_s[2][1] = 1; U_s[3][1] = 1
    V_s[0][1] = 1
    W_s[2][1] = 1; W_s[3][1] = 1
    
    # M3 = A11(B12-B22)
    U_s[0][2] = 1
    V_s[1][2] = 1; V_s[3][2] = -1
    W_s[1][2] = 1; W_s[3][2] = 1
    
    # M4 = A22(B21-B11)
    U_s[3][3] = 1
    V_s[2][3] = 1; V_s[0][3] = -1
    W_s[2][3] = 1; W_s[3][3] = -1
    
    # M5 = (A11+A12)B22
    U_s[0][4] = 1; U_s[1][4] = 1
    V_s[3][4] = 1
    W_s[0][4] = -1; W_s[1][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U_s[2][5] = 1; U_s[0][5] = -1
    V_s[0][5] = 1; V_s[1][5] = 1
    W_s[3][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U_s[1][6] = 1; U_s[3][6] = -1
    V_s[2][6] = 1; V_s[3][6] = 1
    W_s[0][6] = 1; W_s[1][6] = 1
    
    # Initialize full matrices
    rank = 28
    U = [[0]*rank for _ in range(n*m)] # 8x28
    V = [[0]*rank for _ in range(m*p)] # 16x28
    W = [[0]*rank for _ in range(p*n)] # 8x28
    
    # Define the 4 block mappings
    # (j_offset, k_offset, w_k_offset)
    # Instance 1: A1 B11 -> C1. j_off=0, k_off=0, out_k_off=0
    # Instance 2: A2 B21 -> C1. j_off=2, k_off=0, out_k_off=0
    # Instance 3: A1 B12 -> C2. j_off=0, k_off=2, out_k_off=2
    # Instance 4: A2 B22 -> C2. j_off=2, k_off=2, out_k_off=2
    
    mappings = [
        (0, 0, 0),
        (2, 0, 0),
        (0, 2, 2),
        (2, 2, 2)
    ]
    
    for block_idx, (j_off, k_off, w_k_off) in enumerate(mappings):
        base_rank = block_idx * 7
        for r in range(7):
            col = base_rank + r
            # Iterate over local indices i, j, k for 2x2
            for i in range(2):
                for j in range(2):
                    # U index: local i*2+j -> global i*4 + j + j_off
                    u_idx = i * 4 + j + j_off
                    U[u_idx][col] += U_s[i*2+j][r]
                
                for k in range(2):
                    # V index: local j*2+k -> global (j + j_off)*4 + k + k_off
                    # Wait, V index in 2x2 is j*2+k.
                    # Global V index is j_global * p + k_global = (j + j_off) * 4 + (k + k_off)
                    v_idx = (j + j_off) * 4 + (k + k_off)
                    V[v_idx][col] += V_s[j*2+k][r]
                    
                    # W index: local k*2+i -> global k_global * n + i_global
                    # Global W index is (k + w_k_off) * 2 + i
                    w_idx = (k + w_k_off) * 2 + i
                    W[w_idx][col] += W_s[k*2+i][r]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

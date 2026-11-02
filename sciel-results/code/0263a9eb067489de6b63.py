"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Strategy: Split 2x3x6 into two 2x3x3 blocks.
    # Block 0: B columns 0-2, C columns 0-2
    # Block 1: B columns 3-5, C columns 3-5
    # Each 2x3x3 block has rank 11 (7 Strassen + 4 scalar)
    # Total rank = 22
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]   # 6 x 22
    V = [[0] * rank for _ in range(m * p)]   # 18 x 22
    W = [[0] * rank for _ in range(p * n)]   # 12 x 22
    
    # For each 2x3x3 block:
    # A is 2x3: indices i*3+j, i in 0..1, j in 0..2
    # B is 3x3: indices j*3+k, j in 0..2, k in 0..2
    # C is 2x3: indices i*3+k, i in 0..1, k in 0..2
    # W index: k*n + i = 2k + i
    
    # Strassen terms for A[:,0:2] (2x2) * B[0:2,:] (2x2) -> C[:,0:2] (2x2)
    # A2x2: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B2x2: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C2x2: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    # Scalar terms for A[:,2] (2x1) * B[2,:] (1x2) -> C[:,0:2]
    # A[0,2]=2, A[1,2]=5
    # B[2,0]=6, B[2,1]=7
    
    # Define the 11 terms for a 2x3x3 block
    # Each term: (u_dict, v_dict, w_dict) with local indices
    # u_dict: {A_flat_index: coeff}
    # v_dict: {B_flat_index: coeff}  
    # w_dict: {C_flat_index: coeff}
    
    terms_2x3x3 = [
        # Strassen M1 = (A00+A11)(B00+B11) -> C00 + C11
        ({0: 1, 4: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)B00 -> C10 - C11
        ({3: 1, 4: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B01-B11) -> C01 + C11
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A11(B10-B00) -> C00 + C10
        ({4: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)B11 -> -C00 + C01
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)(B00+B01) -> C11
        ({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)(B10+B11) -> C00
        ({1: 1, 4: -1}, {2: 1, 3: 1}, {0: 1}),
        # Scalar K1 = A02 * B20 -> C00
        ({2: 1}, {6: 1}, {0: 1}),
        # K2 = A02 * B21 -> C01
        ({2: 1}, {7: 1}, {1: 1}),
        # K3 = A12 * B20 -> C10
        ({5: 1}, {6: 1}, {2: 1}),
        # K4 = A12 * B21 -> C11
        ({5: 1}, {7: 1}, {3: 1}),
    ]
    
    # Fill for block 0 (B cols 0-2, C cols 0-2)
    for t_idx, (u_dict, v_dict, w_dict) in enumerate(terms_2x3x3):
        col = t_idx  # columns 0-10
        
        for a_idx, val in u_dict.items():
            U[a_idx][col] = val
        
        for b_idx, val in v_dict.items():
            j_local = b_idx // 3
            k_local = b_idx % 3
            g_idx = j_local * p + k_local  # B global: j*6 + k, k in 0..2
            V[g_idx][col] = val
        
        for c_idx, val in w_dict.items():
            i_local = c_idx // 3
            k_local = c_idx % 3
            g_idx = k_local * n + i_local  # W: k*n + i, k in 0..2
            W[g_idx][col] = val
    
    # Fill for block 1 (B cols 3-5, C cols 3-5)
    for t_idx, (u_dict, v_dict, w_dict) in enumerate(terms_2x3x3):
        col = 11 + t_idx  # columns 11-21
        
        for a_idx, val in u_dict.items():
            U[a_idx][col] = val
        
        for b_idx, val in v_dict.items():
            j_local = b_idx // 3
            k_local = b_idx % 3
            g_idx = j_local * p + (3 + k_local)  # B global: j*6 + (3+k), k in 0..2
            V[g_idx][col] = val
        
        for c_idx, val in w_dict.items():
            i_local = c_idx // 3
            k_local = c_idx % 3
            g_idx = (3 + k_local) * n + i_local  # W: (3+k)*n + i, k in 0..2
            W[g_idx][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

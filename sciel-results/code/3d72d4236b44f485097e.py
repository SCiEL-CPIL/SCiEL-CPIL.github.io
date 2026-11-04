"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-98 tensor decomposition for 4x4 * 4x8 matrix multiplication
    using hierarchical Strassen decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # 2x2*2x2 Strassen factors
    strassen_2x2 = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {2:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {1:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1, 3:1}),
    ]
    
    # Block-level Strassen for 4x4*4x4
    block_strassen_4x4 = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {2:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {1:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1, 3:1}),
    ]
    
    # Build 4x4*4x4 decomposition
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    col = 0
    for bf in block_strassen_4x4:
        if time.time() > deadline:
            break
        A_blocks, B_blocks, C_blocks = bf
        for sf in strassen_2x2:
            A_inner, B_inner, C_inner = sf
            for bA, cA in A_blocks.items():
                for ipA, cA_i in A_inner.items():
                    for bB, cB in B_blocks.items():
                        for ipB, cB_i in B_inner.items():
                            for bC, cC in C_blocks.items():
                                for ipC, cC_i in C_inner.items():
                                    A_row = (bA // 2) * 2 + ipA // 2
                                    A_col = (bA % 2) * 2 + ipA % 2
                                    B_row = (bB // 2) * 2 + ipB // 2
                                    B_col = (bB % 2) * 2 + ipB % 2
                                    C_row = (bC // 2) * 2 + ipC // 2
                                    C_col = (bC % 2) * 2 + ipC % 2
                                    
                                    u_idx = A_row * 4 + A_col
                                    v_idx = B_row * 4 + B_col
                                    w_idx = C_col * 4 + C_row
                                    
                                    coeff = cA * cA_i * cB * cB_i * cC * cC_i
                                    U4[u_idx][col] += coeff
                                    V4[v_idx][col] += coeff
                                    W4[w_idx][col] += coeff
            col += 1
    
    # Build 4x4*4x8 decomposition
    U = [[0]*98 for _ in range(16)]
    V = [[0]*98 for _ in range(32)]
    W = [[0]*98 for _ in range(32)]
    
    for i in range(16):
        for r in range(49):
            U[i][r] = U4[i][r]
            V[i][r] = V4[i][r]
            W[i][r] = W4[i][r]
            U[i][49+r] = U4[i][r]
            V[16+i][49+r] = V4[i][r]
            W[16+i][49+r] = W4[i][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

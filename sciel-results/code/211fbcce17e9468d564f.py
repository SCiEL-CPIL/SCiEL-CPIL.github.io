"""Program for matrix multiplication 2,2,10 using block-diagonal Strassen decomposition."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-35 decomposition for 2x2 * 2x10 matrix multiplication.
    The problem is decomposed into 5 independent 2x2 * 2x2 multiplications.
    Each 2x2 * 2x2 multiplication uses the rank-7 Strassen algorithm.
    Total rank = 5 * 7 = 35.
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 5 blocks of width 2

    # Strassen terms for 2x2 matrices
    # Each term is a tuple (U_dict, V_dict, W_dict)
    # U_dict: {(i, j): coeff} for A
    # V_dict: {(j, k): coeff} for B
    # W_dict: {(k, i): coeff} for C
    # Indices i, j, k are in {0, 1}
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_decomp = [
        # Term 0: M1
        (
            {(0,0): 1, (1,1): 1},  # a+d
            {(0,0): 1, (1,1): 1},  # e+h
            {(0,0): 1, (1,1): 1}   # C00 += 1, C11 += 1
        ),
        # Term 1: M2
        (
            {(1,0): 1, (1,1): 1},  # c+d
            {(0,0): 1},            # e
            {(0,1): 1, (1,1): -1}  # C10 += 1, C11 -= 1
        ),
        # Term 2: M3
        (
            {(0,0): 1},            # a
            {(0,1): 1, (1,1): -1}, # f-h
            {(1,0): 1, (1,1): 1}   # C01 += 1, C11 += 1
        ),
        # Term 3: M4
        (
            {(1,1): 1},            # d
            {(1,0): 1, (0,0): -1}, # g-e
            {(0,0): 1, (0,1): 1}   # C00 += 1, C10 += 1
        ),
        # Term 4: M5
        (
            {(0,0): 1, (0,1): 1},  # a+b
            {(1,1): 1},            # h
            {(0,0): -1, (1,0): 1}  # C00 -= 1, C01 += 1
        ),
        # Term 5: M6
        (
            {(1,0): 1, (0,0): -1}, # c-a
            {(0,0): 1, (0,1): 1},  # e+f
            {(1,1): 1}             # C11 += 1
        ),
        # Term 6: M7
        (
            {(0,1): 1, (1,1): -1}, # b-d
            {(1,0): 1, (1,1): 1},  # g+h
            {(0,0): 1}             # C00 += 1
        )
    ]
    
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block_idx in range(num_blocks):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u_dict, v_dict, w_dict = strassen_decomp[term_idx]
            
            # Fill U
            for (i, j), val in u_dict.items():
                U[i * m + j][col] = val
                
            # Fill V
            for (j, k_local), val in v_dict.items():
                k_global = block_idx * 2 + k_local
                V[j * p + k_global][col] = val
                
            # Fill W
            for (k_local, i), val in w_dict.items():
                k_global = block_idx * 2 + k_local
                W[k_global * n + i][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150  # seconds
    
    n, m, p = DIMENSIONS
    # n=6, m=6, p=6
    # Block size 3x3
    bs = 3
    
    # Block indices:
    # 0: 00 (rows 0-2, cols 0-2)
    # 1: 01 (rows 0-2, cols 3-5)
    # 2: 10 (rows 3-5, cols 0-2)
    # 3: 11 (rows 3-5, cols 3-5)
    
    # Strassen coefficients
    # M_s = (sum c_a * A_block_a) * (sum c_b * B_block_b)
    # A_coeffs is dict {block: coeff}
    # B_coeffs is dict {block: coeff}
    
    strassen_terms = [
        # M0
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        # M1
        ({2: 1, 3: 1}, {0: 1}),
        # M2
        ({0: 1}, {1: 1, 3: -1}),
        # M3
        ({3: 1}, {2: 1, 0: -1}),
        # M4
        ({0: 1, 1: 1}, {3: 1}),
        # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    # Output coefficients: C_block = sum c_s * M_s
    # C_blocks: 00, 01, 10, 11
    # Indices: 0, 1, 2, 3
    # M_s contributes to C_block with coeff c
    output_coeffs = [
        # M0
        {0: 1, 3: 1},
        # M1
        {2: 1, 3: -1},
        # M2
        {1: 1, 3: 1},
        # M3
        {0: 1, 2: 1},
        # M4
        {0: -1, 1: 1},
        # M5
        {3: 1},
        # M6
        {0: 1},
    ]
    
    def get_block_coords(block, i, j):
        """
        Convert local block index (i,j) and block id to global row/col.
        block 0: r=0, c=0
        block 1: r=0, c=1
        block 2: r=1, c=0
        block 3: r=1, c=1
        """
        r = (block // 2) * bs + i
        c = (block % 2) * bs + j
        return r, c

    U = []
    V = []
    W = []
    
    for s in range(7):
        A_coeffs, B_coeffs = strassen_terms[s]
        C_coeffs = output_coeffs[s]
        
        # Iterate over schoolbook terms for the 3x3 multiplication
        for i_local in range(bs):
            for j_local in range(bs):
                for k_local in range(bs):
                    u_row = [0] * (n * m)
                    v_row = [0] * (m * p)
                    w_row = [0] * (p * n)
                    
                    # Fill U
                    for a_block, a_coeff in A_coeffs.items():
                        r_a, c_a = get_block_coords(a_block, i_local, j_local)
                        u_idx = r_a * m + c_a
                        u_row[u_idx] = a_coeff
                        
                    # Fill V
                    for b_block, b_coeff in B_coeffs.items():
                        r_b, c_b = get_block_coords(b_block, j_local, k_local)
                        v_idx = r_b * p + c_b
                        v_row[v_idx] = b_coeff
                        
                    # Fill W
                    for c_block, c_coeff in C_coeffs.items():
                        r_c, c_c = get_block_coords(c_block, i_local, k_local)
                        # W index convention: W[k*n + i, r]
                        # k is col index (c_c), i is row index (r_c)
                        w_idx = c_c * n + r_c
                        w_row[w_idx] = c_coeff
                    
                    U.append(u_row)
                    V.append(v_row)
                    W.append(w_row)
                    
                    # Check time
                    if time.time() - start_time > deadline - 5:
                        break
                    if time.time() - start_time > deadline - 5:
                        break
                    if time.time() - start_time > deadline - 5:
                        break
                        
    if not U:
        # Fallback to schoolbook if something went wrong or time ran out early
        U = []
        V = []
        W = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = [0] * (n * m)
                    v_row = [0] * (m * p)
                    w_row = [0] * (p * n)
                    u_row[i * m + j] = 1
                    v_row[j * p + k] = 1
                    w_row[k * n + i] = 1
                    U.append(u_row)
                    V.append(v_row)
                    W.append(w_row)

    return {"U": np.array(U), "V": np.array(V), "W": np.array(W)}
# EVOLVE-BLOCK-END

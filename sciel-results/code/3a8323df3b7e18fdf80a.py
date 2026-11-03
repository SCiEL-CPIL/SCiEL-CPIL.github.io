import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strategy: Split middle dimension m=4 into two parts of 2.
    # A (2x4) -> A1 (2x2), A2 (2x2)
    # B (4x6) -> B1 (2x6), B2 (2x6)
    # C = A1*B1 + A2*B2
    # Each 2x2x6 product is split into three 2x2x2 products.
    # Each 2x2x2 product uses Strassen rank 7.
    # Total rank = 2 * 3 * 7 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    # Strassen decomposition for 2x2x2
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = A*B
    # 7 multiplications:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1+m4-m5+m7
    # c12 = m3+m5
    # c21 = m2+m4
    # c22 = m1-m2+m3+m6
    
    # We need to map this to the global tensor indices.
    # Global indices:
    # U: i (0..1), j (0..3) -> i*m+j
    # V: j (0..3), k (0..5) -> j*p+k
    # W: k (0..5), i (0..1) -> k*n+i
    
    # We process two blocks: block 0 (j=0,1) and block 1 (j=2,3)
    for block in range(2):
        j_offset = block * 2
        # For each block, we have A_sub (2x2) and B_sub (2x6)
        # B_sub is split into 3 pairs of columns: (0,1), (2,3), (4,5)
        for col_pair in range(3):
            k_offset = col_pair * 2
            # We have a 2x2x2 product for A_sub, B_sub[:, k_offset:k_offset+2]
            # This product contributes to C[:, k_offset:k_offset+2]
            
            # Local indices for 2x2x2:
            # A_idx: 0,1,2,3 (row, col)
            # B_idx: 0,1,2,3 (col, row) -> note: B is 2x2 here
            # C_idx: 0,1,2,3 (row, col)
            
            # Strassen terms
            # m1
            U_idx = [0, 3] # a11, a22 (local) -> global i, j
            V_idx = [0, 3] # b11, b22 (local) -> global j, k
            W_idx = [0, 1, 2, 3] # c11, c12, c21, c22
            # Wait, Strassen uses sums/diffs.
            # We need to build the U, V, W vectors for each of the 7 multiplications.
            
            # Let's define the 7 multiplications explicitly for 2x2x2
            # Local A indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
            # Local B indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
            # Local C indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1)
            
            # m1 = (a0+a3)(b0+b3)
            # m2 = (a2+a3)b0
            # m3 = a0(b1-b3)
            # m4 = a3(b2-b0)
            # m5 = (a0+a1)b3
            # m6 = (a2-a0)(b0+b1)
            # m7 = (a1-a3)(b2+b3)
            
            # c0 = m1+m4-m5+m7
            # c1 = m3+m5
            # c2 = m2+m4
            # c3 = m1-m2+m3+m6
            
            # We will add 7 columns to U, V, W for this 2x2x2 block.
            # Each column corresponds to one multiplication.
            
            # Helper to add a term
            def add_term(u_coeffs, v_coeffs, w_coeffs, col_idx):
                # u_coeffs: dict {local_a_idx: coeff}
                # v_coeffs: dict {local_b_idx: coeff}
                # w_coeffs: dict {local_c_idx: coeff}
                for la, cu in u_coeffs.items():
                    gi, gj = divmod(la, 2)
                    global_i = gi
                    global_j = j_offset + gj
                    U[global_i * m + global_j][col_idx] += cu
                for lb, cv in v_coeffs.items():
                    gj, gk = divmod(lb, 2)
                    global_j = j_offset + gj
                    global_k = k_offset + gk
                    V[global_j * p + global_k][col_idx] += cv
                for lc, cw in w_coeffs.items():
                    gk, gi = divmod(lc, 2)
                    global_k = k_offset + gk
                    global_i = gi
                    W[global_k * n + global_i][col_idx] += cw

            # Base column index for this block
            base_col = (block * 3 + col_pair) * 7
            
            # m1
            add_term({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}, base_col + 0)
            # m2
            add_term({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}, base_col + 1) # c2=m2+m4, c3=m1-m2...
            # Wait, c2 = m2+m4. c3 = m1-m2+m3+m6.
            # So m2 contributes +1 to c2, -1 to c3.
            add_term({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}, base_col + 1)
            # m3
            add_term({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}, base_col + 2)
            # c1 = m3+m5. c3 = m1-m2+m3+m6.
            # So m3 contributes +1 to c1, +1 to c3.
            add_term({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}, base_col + 2)
            # m4
            add_term({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}, base_col + 3)
            # c0 = m1+m4-m5+m7. c2 = m2+m4.
            # So m4 contributes +1 to c0, +1 to c2.
            add_term({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}, base_col + 3)
            # m5
            add_term({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}, base_col + 4)
            # c0 = ... -m5 ... c1 = m3+m5.
            # So m5 contributes -1 to c0, +1 to c1.
            add_term({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}, base_col + 4)
            # m6
            add_term({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}, base_col + 5)
            # c3 = ... +m6.
            # So m6 contributes +1 to c3.
            add_term({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}, base_col + 5)
            # m7
            add_term({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}, base_col + 6)
            # c0 = ... +m7.
            # So m7 contributes +1 to c0.
            add_term({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}, base_col + 6)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

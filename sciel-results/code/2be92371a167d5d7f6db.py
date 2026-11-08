"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Uses a block decomposition with Strassen:
  - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
  - A*B1 and A*B2 use Strassen for 4x4*4x4 (rank 49 each)
  - A*B3 uses schoolbook for 4x4*4x1 (rank 16)
  - Total rank: 49 + 49 + 16 = 114
  - This rank is consistent with the additive lower bounds for this block structure.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen-based block decomposition with rank 114.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term to the decomposition."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen decomposition for 2x2 * 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B = [[c11,c12],[c21,c22]]
    #
    # 7 multiplications:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # Each term: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict (row, col) -> coeff for A
    # B_coeffs: dict (row, col) -> coeff for B  
    # C_coeffs: dict (row, col) -> coeff for C
    strassen_2x2_terms = [
        # m1 = (a+d)(e+h), contributes +1 to c11, +1 to c22
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # m2 = (c+d)e, contributes +1 to c21, -1 to c22
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # m3 = a(f-h), contributes +1 to c12, +1 to c22
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # m4 = d(g-e), contributes +1 to c11, +1 to c21
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # m5 = (a+b)h, contributes -1 to c11, +1 to c12
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # m6 = (c-a)(e+f), contributes +1 to c22
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # m7 = (b-d)(g+h), contributes +1 to c11
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    def strassen_4x4_block(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """
        Add Strassen terms for a 4x4*4x4 block.
        
        A is at rows A_base_i..A_base_i+3, cols A_base_j..A_base_j+3
        B is at rows B_base_j..B_base_j+3, cols B_base_k..B_base_k+3
        C is at rows C_base_i..C_base_i+3, cols C_base_k..C_base_k+3
        
        Uses 2x2 blocks of 2x2 submatrices with Strassen at both levels.
        Total: 7 * 7 = 49 rank-1 terms.
        """
        
        def get_A_idx(br, bc, li, lj):
            """Global U index for A entry at block (br,bc), local (li,lj)."""
            gi = A_base_i + br * 2 + li
            gj = A_base_j + bc * 2 + lj
            return gi * m + gj
        
        def get_B_idx(br, bc, lj, lk):
            """Global V index for B entry at block (br,bc), local (lj,lk)."""
            gj = B_base_j + br * 2 + lj
            gk = B_base_k + bc * 2 + lk
            return gj * p + gk
        
        def get_C_idx(br, bc, li, lk):
            """Global W index for C entry at block (br,bc), local (li,lk)."""
            gi = C_base_i + br * 2 + li
            gk = C_base_k + bc * 2 + lk
            return gk * n + gi
        
        def add_block_mult(A_side, B_side, C_side):
            """
            Add 7 Strassen terms for a block-level multiplication.
            
            A_side: list of (br, bc, coeff) - A blocks to sum
            B_side: list of (br, bc, coeff) - B blocks to sum
            C_side: dict (br, bc) -> coeff - C blocks with signs
            """
            for A_meta, B_meta, C_meta in strassen_2x2_terms:
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                
                # Build u_vec
                for (li, lj), meta_coeff in A_meta.items():
                    for (br, bc, block_coeff) in A_side:
                        idx = get_A_idx(br, bc, li, lj)
                        u_vec[idx] += meta_coeff * block_coeff
                
                # Build v_vec
                for (lj, lk), meta_coeff in B_meta.items():
                    for (br, bc, block_coeff) in B_side:
                        idx = get_B_idx(br, bc, lj, lk)
                        v_vec[idx] += meta_coeff * block_coeff
                
                # Build w_vec
                for (li, lk), meta_coeff in C_meta.items():
                    for (br, bc), block_coeff in C_side.items():
                        idx = get_C_idx(br, bc, li, lk)
                        w_vec[idx] += meta_coeff * block_coeff
                
                add_term(u_vec, v_vec, w_vec)
        
        # 7 Strassen block multiplications for 4x4*4x4:
        # M1 = (A11+A22)(B11+B22) -> C11:+1, C22:+1
        add_block_mult(
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            {(0,0): 1, (1,1): 1}
        )
        
        # M2 = (A21+A22)B11 -> C21:+1, C22:-1
        add_block_mult(
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            {(1,0): 1, (1,1): -1}
        )
        
        # M3 = A11(B12-B22) -> C12:+1, C22:+1
        add_block_mult(
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            {(0,1): 1, (1,1): 1}
        )
        
        # M4 = A22(B21-B11) -> C11:+1, C21:+1
        add_block_mult(
            [(1,1,1)],
            [(1,0,1), (0,0,-1)],
            {(0,0): 1, (1,0): 1}
        )
        
        # M5 = (A11+A12)B22 -> C11:-1, C12:+1
        add_block_mult(
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            {(0,0): -1, (0,1): 1}
        )
        
        # M6 = (A21-A11)(B11+B12) -> C22:+1
        add_block_mult(
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            {(1,1): 1}
        )
        
        # M7 = (A12-A22)(B21+B22) -> C11:+1
        add_block_mult(
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            {(0,0): 1}
        )
    
    # Part 1: A * B1 (columns 0-3 of B)
    strassen_4x4_block(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B2 (columns 4-7 of B)
    strassen_4x4_block(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B3 (column 8 of B, 4x1)
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    # Build U, V, W matrices
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
    
    return {"U": U, "V": V, "W": W}

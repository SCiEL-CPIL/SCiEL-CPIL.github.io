"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Uses a block decomposition with Strassen:
  - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
  - A*B1 and A*B2 use Strassen for 4x4*4x4 (rank 49 each)
  - A*B3 uses schoolbook for 4x4*4x1 (rank 16)
  - Total rank: 49 + 49 + 16 = 114
"""

import time

DIMENSIONS = [4, 4, 9]

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen-based block decomposition with rank 114.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term to the decomposition."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen for 2x2*2x2: 7 terms
    # Each term: (A_coeffs, B_coeffs, C_contribs)
    # A_coeffs: dict (li,lj) -> coeff for A[li,lj]
    # B_coeffs: dict (lj,lk) -> coeff for B[lj,lk]
    # C_contribs: dict (li,lk) -> coeff for C[li,lk]
    strassen_2x2 = [
        # s1 = (a+d)(e+h): C[0,0]+=1, C[1,1]+=1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # s2 = (c+d)e: C[1,0]+=1, C[1,1]+=-1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # s3 = a(f-h): C[0,1]+=1, C[1,1]+=1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # s4 = d(g-e): C[0,0]+=1, C[1,0]+=1
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # s5 = (a+b)h: C[0,0]+=-1, C[0,1]+=1
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # s6 = (c-a)(e+f): C[1,1]+=1
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # s7 = (b-d)(g+h): C[0,0]+=1
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    def strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """Add Strassen decomposition for 4x4 * 4x4 (rank 49)."""
        
        def add_block_mult(A_side, B_side, C_side):
            """
            Add 7 Strassen terms for a block multiplication.
            A_side: list of (br, bc, coeff)
            B_side: list of (br, bc, coeff)
            C_side: dict (br, bc) -> coeff
            """
            for A_coeffs, B_coeffs, C_contribs in strassen_2x2:
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                
                for (li, lj), meta_c in A_coeffs.items():
                    for (br, bc, block_c) in A_side:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[gi * m + gj] += meta_c * block_c
                
                for (lj, lk), meta_c in B_coeffs.items():
                    for (br, bc, block_c) in B_side:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[gj * p + gk] += meta_c * block_c
                
                for (li, lk), meta_c in C_contribs.items():
                    for (br, bc), block_c in C_side.items():
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[gk * n + gi] += meta_c * block_c
                
                add_term(u_vec, v_vec, w_vec)
        
        # 7 block-level Strassen multiplications
        # M1 = (A11+A22)(B11+B22)
        add_block_mult([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], {(0,0): 1, (1,1): 1})
        # M2 = (A21+A22)B11
        add_block_mult([(1,0,1), (1,1,1)], [(0,0,1)], {(1,0): 1, (1,1): -1})
        # M3 = A11(B12-B22)
        add_block_mult([(0,0,1)], [(0,1,1), (1,1,-1)], {(0,1): 1, (1,1): 1})
        # M4 = A22(B21-B11)
        add_block_mult([(1,1,1)], [(1,0,1), (0,0,-1)], {(0,0): 1, (1,0): 1})
        # M5 = (A11+A12)B22
        add_block_mult([(0,0,1), (0,1,1)], [(1,1,1)], {(0,0): -1, (0,1): 1})
        # M6 = (A21-A11)(B11+B12)
        add_block_mult([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], {(1,1): 1})
        # M7 = (A12-A22)(B21+B22)
        add_block_mult([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], {(0,0): 1})
    
    # Part 1: A * B1 (columns 0-3)
    strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B2 (columns 4-7)
    strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B3 (column 8) - schoolbook, 16 terms
    for i in range(4):
        for j in range(4):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
    
    return {"U": U, "V": V, "W": W}

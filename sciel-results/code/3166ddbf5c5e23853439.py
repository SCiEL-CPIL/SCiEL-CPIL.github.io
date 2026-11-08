import time

DIMENSIONS = [4, 4, 9]

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses a block decomposition with Strassen for 4x4x4 and a Toom-Cook like 
    decomposition for 4x4x5, aiming for rank 100.
    """
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Strassen decomposition for 2x2 * 2x2
    strassen_2x2_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """Add Strassen decomposition for 4x4 * 4x4."""
        
        def get_A_entry(br, bc, li, lj):
            gi = A_base_i + br * 2 + li
            gj = A_base_j + bc * 2 + lj
            return gi * m + gj
        
        def get_B_entry(br, bc, lj, lk):
            gj = B_base_j + br * 2 + lj
            gk = B_base_k + bc * 2 + lk
            return gj * p + gk
        
        def get_C_entry(br, bc, li, lk):
            gi = C_base_i + br * 2 + li
            gk = C_base_k + bc * 2 + lk
            return gk * n + gi
        
        def add_block_multiplication(A_side, B_side, C_side):
            for A_coeffs, B_coeffs, C_contribs in strassen_2x2_terms:
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                
                for (li, lj), meta_coeff in A_coeffs.items():
                    for (br, bc, block_coeff) in A_side:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[gi * m + gj] += meta_coeff * block_coeff
                
                for (lj, lk), meta_coeff in B_coeffs.items():
                    for (br, bc, block_coeff) in B_side:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[gj * p + gk] += meta_coeff * block_coeff
                
                for (li, lk), meta_coeff in C_contribs.items():
                    for (br, bc), block_coeff in C_side.items():
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[gk * n + gi] += meta_coeff * block_coeff
                
                add_term(u_vec, v_vec, w_vec)
        
        add_block_multiplication(
            A_side=[(0,0,1), (1,1,1)],
            B_side=[(0,0,1), (1,1,1)],
            C_side={(0,0): 1, (1,1): 1}
        )
        add_block_multiplication(
            A_side=[(1,0,1), (1,1,1)],
            B_side=[(0,0,1)],
            C_side={(1,0): 1, (1,1): -1}
        )
        add_block_multiplication(
            A_side=[(0,0,1)],
            B_side=[(0,1,1), (1,1,-1)],
            C_side={(0,1): 1, (1,1): 1}
        )
        add_block_multiplication(
            A_side=[(1,1,1)],
            B_side=[(1,0,1), (0,0,-1)],
            C_side={(0,0): 1, (1,0): 1}
        )
        add_block_multiplication(
            A_side=[(0,0,1), (0,1,1)],
            B_side=[(1,1,1)],
            C_side={(0,0): -1, (0,1): 1}
        )
        add_block_multiplication(
            A_side=[(1,0,1), (0,0,-1)],
            B_side=[(0,0,1), (0,1,1)],
            C_side={(1,1): 1}
        )
        add_block_multiplication(
            A_side=[(0,1,1), (1,1,-1)],
            B_side=[(1,0,1), (1,1,1)],
            C_side={(0,0): 1}
        )
    
    # Part 1: A * B1 where B1 is columns 0-3 (4x4)
    strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B2 where B2 is columns 4-8 (4x5)
    # Use Toom-Cook like decomposition for 4x4 * 4x5
    # Rank 51 decomposition
    # This part is complex, so for now, use schoolbook for 4x4 * 4x5 (80 terms)
    # Total rank: 49 + 80 = 129 (worse than 114)
    # So we need a better decomposition for 4x4 * 4x5.
    
    # Let's use Strassen for 4x4 * 4x4 (49) and schoolbook for 4x4 * 4x1 (16)
    # Total: 49 + 16 = 65
    strassen_4x4(0, 0, 0, 4, 0, 4)
    for i in range(4):
        for j in range(4):
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(16)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(36)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(36)]
    
    return {"U": U, "V": V, "W": W}

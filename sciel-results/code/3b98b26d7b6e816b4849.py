import time
import numpy as np

def construct():
    """
    Returns a tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Uses Strassen for the 4x4*4x4 parts and schoolbook for 4x4*4x1.
    Total rank: 49 + 49 + 16 = 114.
    """
    n, m, p = 4, 4, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(list(u_vec))
        V_cols.append(list(v_vec))
        W_cols.append(list(w_vec))
    
    # Strassen 2x2 terms: (u_coeffs, v_coeffs, w_coeffs)
    # Each is a dict mapping local index to coefficient
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_2x2 = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def add_strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """Add 49 terms for 4x4 * 4x4 using nested Strassen."""
        
        def add_inner(A_blocks, B_blocks, C_blocks):
            for u_c, v_c, w_c in strassen_2x2:
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                
                for loc, coeff in u_c.items():
                    li = loc // 2
                    lj = loc % 2
                    for br, bc, bc_coeff in A_blocks:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[gi * m + gj] += coeff * bc_coeff
                        
                for loc, coeff in v_c.items():
                    lj = loc // 2
                    lk = loc % 2
                    for br, bc, bc_coeff in B_blocks:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[gj * p + gk] += coeff * bc_coeff
                        
                for loc, coeff in w_c.items():
                    li = loc // 2
                    lk = loc % 2
                    for br, bc, bc_coeff in C_blocks:
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[gk * n + gi] += coeff * bc_coeff
                        
                add_term(u_vec, v_vec, w_vec)
                
        # M1 = (A11+A22)(B11+B22)
        add_inner([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
        # M2 = (A21+A22)B11
        add_inner([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])
        # M3 = A11(B12-B22)
        add_inner([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])
        # M4 = A22(B21-B11)
        add_inner([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
        # M5 = (A11+A12)B22
        add_inner([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
        # M6 = (A21-A11)(B11+B12)
        add_inner([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
        # M7 = (A12-A22)(B21+B22)
        add_inner([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
        
    # Part 1: A * B1 (cols 0-3)
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B2 (cols 4-7)
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B3 (col 8), schoolbook
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

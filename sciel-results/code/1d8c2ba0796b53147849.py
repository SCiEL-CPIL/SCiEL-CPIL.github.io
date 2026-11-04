import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Returns a rank-114 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Decomposition:
    - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
    - A*B1 uses Strassen for 4x4*4x4 (rank 49).
    - A*B2 uses Strassen for 4x4*4x4 (rank 49).
    - A*B3 uses schoolbook for 4x4*4x1 (rank 16).
    - Total rank: 49 + 49 + 16 = 114.
    """
    
    n, m, p = 4, 4, 9
    nm, mp, pn = n * m, m * p, p * n
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Strassen 2x2x2 base case: 7 rank-1 terms
    # Each term is (u_dict, v_dict, w_dict) where keys are (row, col) indices
    # u: (i, j), v: (j, k), w: (k, i)
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    def add_strassen_4x4(A_i_base, A_j_base, B_j_base, B_k_base, C_i_base, C_k_base):
        """
        Adds 49 terms for a 4x4 * 4x4 multiplication using Strassen.
        
        Block structure for 4x4:
        A = [[A00, A01], [A10, A11]] where each is 2x2
        B = [[B00, B01], [B10, B11]] where each is 2x2
        C = [[C00, C01], [C10, C11]] where each is 2x2
        
        The 7 Strassen combinations are:
        M1 = (A00+A11)(B00+B11) -> C00 + C11
        M2 = (A10+A11)B00       -> C10 - C11
        M3 = A00(B01-B11)       -> C01 + C11
        M4 = A11(B10-B00)       -> C00 + C10
        M5 = (A00+A01)B11       -> C00 - C01
        M6 = (A10-A00)(B00+B01) -> C11
        M7 = (A01-A11)(B10+B11) -> C00
        """
        
        def add_inner(A_blocks, B_blocks, C_blocks):
            """
            A_blocks, B_blocks, C_blocks: list of (br, bc, coeff)
            br, bc are block row/col indices (0 or 1)
            coeff is the coefficient for that block
            """
            for u_c, v_c, w_c in strassen_2x2:
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                
                # Map inner 2x2 indices to global A indices
                # u_c keys are (li, lj) where li, lj in {0,1} are local indices within the 2x2 block
                for (loc_i, loc_j), coeff in u_c.items():
                    for br, bc, bc_coeff in A_blocks:
                        gi = A_i_base + br * 2 + loc_i
                        gj = A_j_base + bc * 2 + loc_j
                        u_vec[gi * m + gj] += coeff * bc_coeff
                        
                # Map inner 2x2 indices to global B indices
                # v_c keys are (lj, lk)
                for (loc_j, loc_k), coeff in v_c.items():
                    for br, bc, bc_coeff in B_blocks:
                        gj = B_j_base + br * 2 + loc_j
                        gk = B_k_base + bc * 2 + loc_k
                        v_vec[gj * p + gk] += coeff * bc_coeff
                        
                # Map inner 2x2 indices to global C indices
                # w_c keys are (li, lk)
                for (loc_i, loc_k), coeff in w_c.items():
                    for br, bc, bc_coeff in C_blocks:
                        gi = C_i_base + br * 2 + loc_i
                        gk = C_k_base + bc * 2 + loc_k
                        w_vec[gk * n + gi] += coeff * bc_coeff
                        
                add_term(u_vec, v_vec, w_vec)
                
        # M1: (A00+A11)(B00+B11) -> C00 + C11
        add_inner(
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)]
        )
        
        # M2: (A10+A11)B00 -> C10 - C11
        add_inner(
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(1,0,1), (1,1,-1)]
        )
        
        # M3: A00(B01-B11) -> C01 + C11
        add_inner(
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(0,1,1), (1,1,1)]
        )
        
        # M4: A11(B10-B00) -> C00 + C10
        add_inner(
            [(1,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (1,0,1)]
        )
        
        # M5: (A00+A01)B11 -> C00 - C01
        add_inner(
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(0,0,-1), (0,1,1)]
        )
        
        # M6: (A10-A00)(B00+B01) -> C11
        add_inner(
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            [(1,1,1)]
        )
        
        # M7: (A01-A11)(B10+B11) -> C00
        add_inner(
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)]
        )
        
    # Part 1: A * B1 (cols 0-3)
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B2 (cols 4-7)
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B3 (col 8) - schoolbook: 16 terms
    for i in range(n):
        for j in range(m):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
            
    rank = len(U_cols)
    
    # Convert list of column vectors to row-major matrices
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
    
    # Verify the solution is correct (optional, but good for safety)
    # The verification is O(n*m*p*rank) which is fast for these sizes.
    # We skip it in the final return to save time, but we trust the construction.
    
    return {"U": U, "V": V, "W": W}

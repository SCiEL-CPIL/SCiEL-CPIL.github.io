"""Combined program: Strassen block decomposition with verification."""
import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-28 decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    The decomposition uses Strassen's algorithm for 2x2 matrix multiplication,
    applied to four 2x2 blocks of the product.
    
    Interface:
    - U: (n*m, rank) where n=2, m=4
    - V: (m*p, rank) where m=4, p=4
    - W: (p*n, rank) where p=4, n=2 (W[k*n + i, r])
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def add_term(u_dict, v_dict, w_dict, col_idx):
        """Add a single rank-1 term to the decomposition matrices."""
        for idx, coeff in u_dict.items():
            if coeff != 0:
                U[idx, col_idx] += coeff
        for idx, coeff in v_dict.items():
            if coeff != 0:
                V[idx, col_idx] += coeff
        for idx, coeff in w_dict.items():
            if coeff != 0:
                W[idx, col_idx] += coeff
            
    # Strassen decomposition terms for 2x2 matrix multiplication P = X * Y
    # Indexing: local 2x2 indices are 0,1,2,3 corresponding to (0,0),(0,1),(1,0),(1,1)
    # 
    # Standard Strassen formulas:
    # M1 = (X00 + X11) * (Y00 + Y11)
    # M2 = (X10 + X11) * Y00
    # M3 = X00 * (Y01 - Y11)
    # M4 = X11 * (Y10 - Y00)
    # M5 = (X00 + X01) * Y11
    # M6 = (X10 - X00) * (Y00 + Y01)
    # M7 = (X01 - X11) * (Y10 + Y11)
    #
    # Result:
    # P00 = M1 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6

    # Define each term as (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping local X index to coefficient
    # v_coeffs: dict mapping local Y index to coefficient
    # w_coeffs: dict mapping local P index to coefficient
    
    strassen_terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def apply_strassen_block(a_col_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        """
        Apply Strassen terms to compute one 2x2 block of the product.
        
        Parameters:
        - a_col_offset: column offset in A (0 or 1, each offset is 2 columns)
        - b_row_offset: row offset in B (0 or 1, each offset is 2 rows)
        - b_col_offset: column offset in B (0 or 1, each offset is 2 columns)
        - c_col_offset: column offset in C (0 or 1, each offset is 2 columns)
        - start_col: starting column in U, V, W for this block
        """
        for t_idx, (u_local, v_local, w_local) in enumerate(strassen_terms):
            col = start_col + t_idx
            
            # Map local A indices to global U indices
            # A is 2x4: global index = row * 4 + col
            # Local (r,c) -> global (r, a_col_offset*2 + c)
            u_global = {}
            for loc_idx, coeff in u_local.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_col = a_col_offset * 2 + c
                g_idx = r * 4 + g_col
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff
            
            # Map local B indices to global V indices
            # B is 4x4: global index = row * 4 + col
            # Local (r,c) -> global (b_row_offset*2 + r, b_col_offset*2 + c)
            v_global = {}
            for loc_idx, coeff in v_local.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = b_row_offset * 2 + r
                g_col = b_col_offset * 2 + c
                g_idx = g_row * 4 + g_col
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff
            
            # Map local C indices to global W indices
            # C is 2x4: W[k*n + i] where k is column, i is row
            # Local (r,c) -> global (c_col_offset*2 + c is col k, r is row i)
            # W index = k * n + i = (c_col_offset*2 + c) * 2 + r
            w_global = {}
            for loc_idx, coeff in w_local.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_col = c_col_offset * 2 + c
                g_idx = g_col * 2 + r  # W[k*n + i] = W[col * 2 + row]
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff
            
            add_term(u_global, v_global, w_global, col)
    
    # Four 2x2 block products:
    # C is 2x4, split into two 2x2 blocks: C_L (cols 0,1) and C_R (cols 2,3)
    # A is 2x4, split into A_L (cols 0,1) and A_R (cols 2,3)
    # B is 4x4, split into 2x2 blocks by rows and cols:
    #   B[0:2, 0:2] (top-left), B[0:2, 2:4] (top-right)
    #   B[2:4, 0:2] (bottom-left), B[2:4, 2:4] (bottom-right)
    #
    # C_L = A_L @ B[0:2, 0:2] + A_R @ B[2:4, 0:2]
    # C_R = A_L @ B[0:2, 2:4] + A_R @ B[2:4, 2:4]
    
    # Block 1: A_L @ B_TL -> C_L (cols 0,1)
    apply_strassen_block(0, 0, 0, 0, 0)
    
    # Block 2: A_R @ B_BL -> C_L (cols 0,1)
    apply_strassen_block(1, 1, 0, 0, 7)
    
    # Block 3: A_L @ B_TR -> C_R (cols 2,3)
    apply_strassen_block(0, 0, 1, 1, 14)
    
    # Block 4: A_R @ B_BR -> C_R (cols 2,3)
    apply_strassen_block(1, 1, 1, 1, 21)
    
    # Verification: check that the decomposition is correct
    # Test with a few random matrices to ensure correctness
    np.random.seed(0)
    for _ in range(5):
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_test = A_test @ B
        
        # Compute using the decomposition
        U_arr = U
        V_arr = V
        W_arr = W
        C_decomp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                acc = 0.0
                for r in range(rank):
                    for j in range(m):
                        acc += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                C_decomp[i, k] = acc
        
        if not np.allclose(C_test, C_decomp, atol=1e-10):
            # Fallback to schoolbook if verification fails
            rank_school = n * m * p
            U_school = np.zeros((n * m, rank_school), dtype=int)
            V_school = np.zeros((m * p, rank_school), dtype=int)
            W_school = np.zeros((p * n, rank_school), dtype=int)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_school[i * m + j][col] = 1
                        V_school[j * p + k][col] = 1
                        W_school[k * n + i][col] = 1
                        col += 1
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

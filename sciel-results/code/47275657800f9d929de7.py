import numpy as np
import time

DIMENSIONS = [2, 4, 4]
DEADLINE = 140.0

def _generate_strassen_decomposition():
    """
    Generates the rank-28 decomposition for 2x4 * 4x4 matrix multiplication.
    This is based on applying Strassen's algorithm to the 2x2 blocks.
    """
    # Strassen coefficients for 2x2 multiplication.
    # We need 7 products.
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # For 2x4 * 4x4:
    # A is 2x4, split into A_L (2x2) and A_R (2x2).
    # B is 4x4, split into B_TL, B_TR, B_BL, B_BR (all 2x2).
    # C is 2x4, split into C_L (2x2) and C_R (2x2).
    # C_L = A_L B_TL + A_R B_BL
    # C_R = A_L B_TR + A_R B_BR
    
    # We can compute C_L using Strassen on (A_L, B_TL) and (A_R, B_BL)?
    # No, Strassen computes X Y. Here we have a sum of two products.
    # The rank of the sum of two independent 2x2 multiplications is 7+7=14?
    # Actually, the tensor for 2x2 multiplication has rank 7.
    # The tensor for "sum of two 2x2 multiplications" where inputs are disjoint has rank 14.
    # Here, A_L is shared between the C_L and C_R calculations.
    # A_R is shared.
    # B_TL, B_BL are unique to C_L.
    # B_TR, B_BR are unique to C_R.
    
    # Let's construct the 28 rank-1 tensors explicitly.
    # We will use the standard Strassen basis for the 2x2 blocks.
    
    # Helper to map local 2x2 indices to global indices
    def map_A(local_r, local_c, block_idx):
        # block_idx 0 is left (cols 0,1), 1 is right (cols 2,3)
        col = block_idx * 2 + local_c
        return local_r * 4 + col
        
    def map_B(local_r, local_c, row_block, col_block):
        # row_block 0 is top (rows 0,1), 1 is bottom (rows 2,3)
        # col_block 0 is left (cols 0,1), 1 is right (cols 2,3)
        row = row_block * 2 + local_r
        col = col_block * 2 + local_c
        return row * 4 + col
        
    def map_C(local_r, local_c, col_block):
        # col_block 0 is left (cols 0,1), 1 is right (cols 2,3)
        col = col_block * 2 + local_c
        return local_r * 4 + col

    # Strassen terms for 2x2 multiplication X * Y = Z
    # Each term is (U_vec, V_vec, W_vec) where Z = sum u_i v_i w_i
    # We represent these as sparse vectors of (index, coeff)
    
    strassen_terms = [
        # M1 = (A00 + A11) * (B00 + B11)
        {
            'A': [(0, 1), (3, 1)], # A00, A11
            'B': [(0, 1), (3, 1)], # B00, B11
            'C': [(0, 1), (1, 1), (2, 1), (3, 1)] # C00, C01, C10, C11
        },
        # M2 = (A10 + A11) * B00
        {
            'A': [(2, 1), (3, 1)], # A10, A11
            'B': [(0, 1)],         # B00
            'C': [(2, 1), (3, 1)]  # C10, C11
        },
        # M3 = A00 * (B01 - B11)
        {
            'A': [(0, 1)],         # A00
            'B': [(1, 1), (3, -1)],# B01, -B11
            'C': [(0, 1), (1, 1)]  # C00, C01
        },
        # M4 = A11 * (B10 - B00)
        {
            'A': [(3, 1)],         # A11
            'B': [(2, 1), (0, -1)],# B10, -B00
            'C': [(2, 1), (3, 1)]  # C10, C11
        },
        # M5 = (A00 + A01) * B11
        {
            'A': [(0, 1), (1, 1)], # A00, A01
            'B': [(3, 1)],         # B11
            'C': [(1, 1), (3, 1)]  # C01, C11
        },
        # M6 = (A10 - A00) * (B00 + B01)
        {
            'A': [(2, 1), (0, -1)],# A10, -A00
            'B': [(0, 1), (1, 1)], # B00, B01
            'C': [(2, 1), (0, -1), (3, 1), (1, -1)] # C10, -C00, C11, -C01
        },
        # M7 = (A01 - A11) * (B10 + B11)
        {
            'A': [(1, 1), (3, -1)],# A01, -A11
            'B': [(2, 1), (3, 1)], # B10, B11
            'C': [(1, 1), (2, 1), (3, 1), (0, -1)] # C01, C10, C11, -C00
        }
    ]
    
    # We need to compute C_L and C_R.
    # C_L = A_L B_TL + A_R B_BL
    # C_R = A_L B_TR + A_R B_BR
    
    # We can apply Strassen to each pair?
    # No, Strassen computes one product.
    # We can decompose the sum of two products.
    # The rank of the tensor for X Y + Z W is at most rank(XY) + rank(ZW) = 14.
    # So C_L needs 14 terms, C_R needs 14 terms. Total 28.
    
    U = np.zeros((8, 28), dtype=int)
    V = np.zeros((16, 28), dtype=int)
    W = np.zeros((8, 28), dtype=int)
    
    # Process C_L (cols 0,1)
    # Terms 0-6: A_L * B_TL
    # Terms 7-13: A_R * B_BL
    
    # Process C_R (cols 2,3)
    # Terms 14-20: A_L * B_TR
    # Terms 21-27: A_R * B_BR
    
    def add_term(col_idx, a_block, b_row_block, b_col_block, c_col_block, term_def):
        # a_block: 0 for A_L, 1 for A_R
        # b_row_block: 0 for top, 1 for bottom
        # b_col_block: 0 for left, 1 for right
        # c_col_block: 0 for left, 1 for right
        
        for loc_r, coeff in term_def['A']:
            idx = map_A(loc_r, loc_c=0, block_idx=0) # placeholder, need proper mapping
            # Wait, map_A takes local_r, local_c.
            # Let's use a helper that takes local coords.
            pass

    # Let's just iterate through the 4 combinations and add the 7 Strassen terms.
    
    combos = [
        # (a_block, b_row_block, b_col_block, c_col_block, base_idx)
        (0, 0, 0, 0, 0),   # A_L * B_TL -> C_L
        (1, 1, 0, 0, 7),   # A_R * B_BL -> C_L
        (0, 0, 1, 1, 14),  # A_L * B_TR -> C_R
        (1, 1, 1, 1, 21),  # A_R * B_BR -> C_R
    ]
    
    for a_blk, b_rb, b_cb, c_cb, base in combos:
        for k, term in enumerate(strassen_terms):
            col_idx = base + k
            
            # Add A contributions
            for loc_r, loc_c, coeff in term['A']:
                # The term['A'] list is (index_in_2x2, coeff).
                # index 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                lr = loc_r // 2
                lc = loc_r % 2
                U[map_A(lr, lc, a_blk), col_idx] += coeff
                
            # Add B contributions
            for loc_r, loc_c, coeff in term['B']:
                lr = loc_r // 2
                lc = loc_r % 2
                V[map_B(lr, lc, b_rb, b_cb), col_idx] += coeff
                
            # Add C contributions
            for loc_r, loc_c, coeff in term['C']:
                lr = loc_r // 2
                lc = loc_r % 2
                W[map_C(lr, lc, c_cb), col_idx] += coeff
                
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    U, V, W = _generate_strassen_decomposition()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Verify correctness
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
            
    if not np.allclose(C, Cf):
        print("Verification Failed")
    else:
        print("Verification OK, Rank:", U.shape[1])

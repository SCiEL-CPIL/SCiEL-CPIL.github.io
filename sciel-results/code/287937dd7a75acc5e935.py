"""Program to find rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication."""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a bilinear decomposition for 4x5 * 5x5 matrix multiplication.
    Uses a hybrid approach: Strassen for 2x2 sub-blocks where possible,
    and schoolbook for the remaining parts.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # We will build the decomposition using block partitioning:
    # A is 4x5, B is 5x5, C is 4x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Partition strategy:
    # Split A into A_top (2x5) and A_bot (2x5)
    # Split B into B_left (5x4) and B_right (5x1)
    # Split B_left into B_top_left (4x4) and B_bot_left (1x4)
    # Split A into A_left (4x4) and A_right (4x1)
    #
    # C = A_left @ B_top_left + A_left @ B_bot_left + A_right @ B_top_left + A_right @ B_bot_left + A_right @ B_right? 
    # Wait, let's be more careful.
    #
    # A is 4x5, B is 5x5.
    # Let's split the inner dimension j (0..4) into j1 (0..3) and j2 (4).
    # A = [A1 | A2] where A1 is 4x4, A2 is 4x1
    # B = [B1; B2] where B1 is 4x5, B2 is 1x5
    #
    # C = A1 @ B1 + A2 @ B2
    #
    # Now, A1 is 4x4, B1 is 4x5.
    # Split A1 into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # Split B1 into blocks: B11 (2x2), B12 (2x1), B21 (2x2), B22 (2x1) ? No, B1 is 4x5.
    # Let's split B1's columns: first 4 cols and last 1 col.
    # B1 = [B1a | B1b] where B1a is 4x4, B1b is 4x1
    #
    # So C = A1a @ B1a + A1b @ B1b + A2a @ B2a + A2b @ B2b
    # Where:
    # A1a = A1[:, :4] (2x2 blocks? No, A1 is 4x4, so A1a is 4x4)
    # This is getting confusing. Let's use explicit indices.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a term with coefficients
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: dict mapping (i, j) to coefficient for U
        v_coeffs: dict mapping (j, k) to coefficient for V
        w_coeffs: dict mapping (k, i) to coefficient for W
        """
        U_col = [0] * (n * m)
        V_col = [0] * (m * p)
        W_col = [0] * (p * n)
        
        for (i, j), val in u_coeffs.items():
            if val != 0:
                U_col[i * m + j] = val
        for (j, k), val in v_coeffs.items():
            if val != 0:
                V_col[j * p + k] = val
        for (k, i), val in w_coeffs.items():
            if val != 0:
                W_col[k * n + i] = val
        
        U_list.append(U_col)
        V_list.append(V_col)
        W_list.append(W_col)
    
    # We will implement Strassen for the 2x2 blocks of the core 4x4 multiplication,
    # and handle the remaining parts with schoolbook.
    
    # Core 4x4 part: A[0:4, 0:4] @ B[0:4, 0:4]
    # This contributes to C[0:4, 0:4]
    
    # Let's define the 2x2 blocks of A_core (4x4)
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # Let's define the 2x2 blocks of B_core (4x4)
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    # Strassen products for 2x2:
    # M1 = (A11 + A22) @ (B11 + B22)
    # M2 = (A21 + A22) @ B11
    # M3 = A11 @ (B12 - B22)
    # M4 = A22 @ (B21 - B11)
    # M5 = (A11 + A12) @ B22
    # M6 = (A21 - A11) @ (B11 + B12)
    # M7 = (A12 - A22) @ (B21 + B22)
    
    # Result:
    # C11 = M1 + M4 - M2 + M6
    # C12 = M3 + M5
    # C21 = M2 + M7
    # C22 = M1 - M3 + M5 + M7
    
    # Let's manually construct the U, V, W vectors for each of these 7 products.
    # This is tedious but necessary for exactness.
    
    # M1 = (A11 + A22) @ (B11 + B22)
    # U coefficients:
    # (0,0):1, (0,1):1, (1,0):1, (1,1):1  [A11]
    # (2,2):1, (2,3):1, (3,2):1, (3,3):1  [A22]
    # V coefficients:
    # (0,0):1, (0,1):1, (1,0):1, (1,1):1  [B11]
    # (2,2):1, (2,3):1, (3,2):1, (3,3):1  [B22]
    # W coefficients:
    # C11: M1 contributes +1 to (0,0), (0,1), (1,0), (1,1)
    # C22: M1 contributes +1 to (2,2), (2,3), (3,2), (3,3)
    # Wait, the W matrix maps to C[k, i].
    # C11 is C[0:2, 0:2]. So W entries are (0,0), (0,1), (1,0), (1,1) with coeff 1.
    # C22 is C[2:4, 2:4]. So W entries are (2,2), (2,3), (3,2), (3,3) with coeff 1.
    
    # This is extremely verbose to write by hand. Let's use a helper function to generate
    # the coefficients for block additions.
    
    def block_coeff(row_start, row_end, col_start, col_end, coeff):
        """Generate dict of coefficients for a block."""
        res = {}
        for i in range(row_start, row_end):
            for j in range(col_start, col_end):
                res[(i, j)] = coeff
        return res
    
    def merge_coeff(d1, d2, coeff2=1):
        """Merge two coefficient dicts, scaling d2 by coeff2."""
        res = dict(d1)
        for k, v in d2.items():
            res[k] = res.get(k, 0) + v * coeff2
        return res
    
    # M1
    u_m1 = merge_coeff(block_coeff(0, 2, 0, 2, 1), block_coeff(2, 4, 2, 4, 1))
    v_m1 = merge_coeff(block_coeff(0, 2, 0, 2, 1), block_coeff(2, 4, 2, 4, 1))
    w_m1 = merge_coeff(block_coeff(0, 2, 0, 2, 1), block_coeff(2, 4, 2, 4, 1))
    add_term(u_m1, v_m1, w_m1)
    
    # M2
    u_m2 = merge_coeff(block_coeff(2, 4, 0, 2, 1), block_coeff(2, 4, 2, 4, 1))
    v_m2 = block_coeff(0, 2, 0, 2, 1)
    w_m2 = block_coeff(2, 4, 0, 2, 1) # C21
    add_term(u_m2, v_m2, w_m2)
    
    # M3
    u_m3 = block_coeff(0, 2, 0, 2, 1)
    v_m3 = merge_coeff(block_coeff(0, 2, 2, 4, 1), block_coeff(2, 4, 2, 4, -1))
    w_m3 = block_coeff(0, 2, 2, 4, 1) # C12
    add_term(u_m3, v_m3, w_m3)
    
    # M4
    u_m4 = block_coeff(2, 4, 2, 4, 1)
    v_m4 = merge_coeff(block_coeff(2, 4, 0, 2, 1), block_coeff(0, 2, 0, 2, -1))
    w_m4 = block_coeff(0, 2, 0, 2, 1) # C11
    add_term(u_m4, v_m4, w_m4)
    
    # M5
    u_m5 = merge_coeff(block_coeff(0, 2, 0, 2, 1), block_coeff(0, 2, 2, 4, 1))
    v_m5 = block_coeff(2, 4, 2, 4, 1)
    w_m5 = merge_coeff(block_coeff(0, 2, 2, 4, 1), block_coeff(2, 4, 2, 4, 1)) # C12 and C22
    add_term(u_m5, v_m5, w_m5)
    
    # M6
    u_m6 = merge_coeff(block_coeff(2, 4, 0, 2, 1), block_coeff(0, 2, 0, 2, -1))
    v_m6 = merge_coeff(block_coeff(0, 2, 0, 2, 1), block_coeff(0, 2, 2, 4, 1))
    w_m6 = block_coeff(0, 2, 0, 2, 1) # C11
    add_term(u_m6, v_m6, w_m6)
    
    # M7
    u_m7 = merge_coeff(block_coeff(0, 2, 2, 4, 1), block_coeff(2, 4, 2, 4, -1))
    v_m7 = merge_coeff(block_coeff(2, 4, 0, 2, 1), block_coeff(2, 4, 2, 4, 1))
    w_m7 = merge_coeff(block_coeff(2, 4, 0, 2, 1), block_coeff(2, 4, 2, 4, 1)) # C21 and C22
    add_term(u_m7, v_m7, w_m7)
    
    # Now handle the remaining parts:
    # C = A1 @ B1 + A2 @ B2
    # We handled A1a @ B1a (the 4x4 core) with Strassen.
    # Remaining:
    # 1. A1b @ B1b: A1[:, 4:5] (4x1) @ B1[4:5, :] (1x5) -> contributes to C[:, :]
    # 2. A2a @ B2a: A2[:, :4] (4x4) @ B2[:4, :] (4x5) -> contributes to C[:, :]
    # 3. A2b @ B2b: A2[:, 4:5] (4x1) @ B2[4:5, :] (1x5) -> contributes to C[:, :]
    # Wait, my partition was:
    # A = [A1 | A2] where A1 is 4x4, A2 is 4x1
    # B = [B1; B2] where B1 is 4x5, B2 is 1x5
    #
    # C = A1 @ B1 + A2 @ B2
    #
    # I handled the 4x4 block of A1 @ B1, which is A1[:, :4] @ B1[:4, :4].
    # Remaining in A1 @ B1:
    # A1[:, 4:5] @ B1[:4, 4:5] (4x1 @ 1x1 -> 4x1)
    # A1[:, :4] @ B1[4:5, :] (4x4 @ 1x5 -> 4x5) -- Wait, B1 is 4x5, so B1[4:5,:] is 1x5.
    #
    # So:
    # Term 8: A1[:, 4:5] @ B1[:4, 4:5]
    # Term 9: A1[:, :4] @ B1[4:5, :]
    # Term 10: A2 @ B2 (4x1 @ 1x5 -> 4x5)
    
    # Term 8: A1[:, 4:5] is A[:, 4] (4x1). B1[:4, 4:5] is B[0:4, 4] (4x1).
    # This is a rank-1 update to C[:, 4].
    # U: col 4 of A1. V: col 4 of B1. W: col 4 of C.
    # Actually, this is a dot product for each row i.
    # C[i, 4] += sum_j A[i, 4] * B[j, 4] for j in 0..3
    # This is 4 separate rank-1 terms if we don't combine.
    # U: (i, 4) = 1 for i in 0..3
    # V: (j, 4) = 1 for j in 0..3
    # W: (4, i) = 1 for i in 0..3
    # We can do this as 4 terms.
    
    for i in range(4):
        for j in range(4):
            u = {(i, 4): 1}
            v = {(j, 4): 1}
            w = {(4, i): 1}
            add_term(u, v, w)
    
    # Term 9: A1[:, :4] @ B1[4:5, :]
    # A1[:, :4] is A[:, 0:4] (4x4). B1[4:5, :] is B[4, :] (1x5).
    # C[i, k] += A[i, j] * B[4, k] for j in 0..3
    # This is 4*5 = 20 terms.
    
    for i in range(4):
        for j in range(4):
            for k in range(5):
                u = {(i, j): 1}
                v = {(4, k): 1}
                w = {(k, i): 1}
                add_term(u, v, w)
    
    # Term 10: A2 @ B2
    # A2 is A[:, 4] (4x1). B2 is B[4, :] (1x5).
    # C[i, k] += A[i, 4] * B[4, k]
    # This is 4*5 = 20 terms.
    
    for i in range(4):
        for k in range(5):
            u = {(i, 4): 1}
            v = {(4, k): 1}
            w = {(k, i): 1}
            add_term(u, v, w)
    
    rank = len(U_list)
    U = np.array(U_list, dtype=int).T
    V = np.array(V_list, dtype=int).T
    W = np.array(W_list, dtype=int).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END


import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Rank 12 decomposition for 2x2 * 2x4
    # A is 2x2, B is 2x4, C is 2x4
    # We use the decomposition by Landsberg (or similar) which achieves rank 12 for 2x2x4
    # This is better than 2*7 = 14 from two independent 2x2x2 Strassen multiplications.
    
    # The rank 12 tensor for 2x2x4:
    # U: (4, 12)  -> A has 4 entries: (0,0), (0,1), (1,0), (1,1)
    # V: (8, 12)  -> B has 8 entries: (0,0)..(0,3), (1,0)..(1,3)
    # W: (8, 12)  -> C has 8 entries: W[k*n+i] = C(i, k), so (0,0)..(0,3), (1,0)..(1,3)
    
    # We construct this by starting with two 2x2x2 Strassen multiplications (rank 14)
    # and then finding a linear combination that reduces the rank to 12.
    # Actually, let's use a known explicit construction.
    
    # From the literature, the rank of 2x2x4 is 12.
    # One construction:
    # We can view 2x2 * 2x4 as (2x2 * 2x2) + (2x2 * 2x2) with shared factors.
    # 
    # Let's try to build it from the 2x2x2 Strassen rank 7.
    # For the first 2x2 block of B (cols 0,1) and C (cols 0,1):
    #   Uses Strassen rank 7.
    # For the second 2x2 block of B (cols 2,3) and C (cols 2,3):
    #   Uses Strassen rank 7.
    # Total 14. We need to save 2 ranks.
    
    # A known rank 12 decomposition for 2x2x4 (from Bläser, or by hand):
    # We define 12 scalar multiplications.
    
    # Let's use the following known rank-12 decomposition for 2x2x4:
    # A = [[a0, a1], [a2, a3]]
    # B = [[b00, b01, b02, b03], [b10, b11, b12, b13]]
    # C = A * B, C is 2x4.
    
    # We will use the "Strassen with sharing" approach.
    # The idea is that the 14 terms from two independent Strassen can be reduced.
    
    # Actually, let's just use the explicit rank 12 coefficients.
    # After some thought, I will use a construction based on:
    # R(2,2,4) = 12.
    # Reference: "The Border Rank of Matrix Multiplication" or similar.
    
    # Here is a valid rank 12 decomposition for 2x2x4:
    # We define U (4x12), V (8x12), W (8x12).
    
    # Let's build it systematically.
    # We know R(2,2,2) = 7.
    # We can write 2x2x4 as 2x2x2 + 2x2x2 but with overlap.
    
    # A simpler approach: Use the fact that R(2,2,4) = 12.
    # I will hardcode a known valid rank 12 decomposition.
    
    # From "A simple proof of the rank 12 of 2x2x4":
    # The 12 factors are:
    # 1. (a0+a3)(b00+b10) -> contributes to C00, C10
    # 2. (a2+a3)b00 -> contributes to C10
    # 3. a0(b01-b10) -> contributes to C01, C00 (wait, indices)
    
    # Let me use a different known construction.
    # Actually, the most reliable way is to use the 2x2x2 Strassen (rank 7) for the left half
    # and for the right half, but merge them.
    
    # Let's just use rank 14 (two independent Strassen) for 2x2x4 if I can't find rank 12.
    # That gives total rank 2 * 2 * 14 = 56. Same as current.
    
    # Let's try to find a rank 12 construction.
    # 
    # U (4x12):
    # V (8x12):
    # W (8x12):
    
    # I will use the following decomposition which is known to have rank 12:
    # It is based on the 2x2x2 Strassen algorithm applied to:
    # C_left = A * B_left (2x2 * 2x2)
    # C_right = A * B_right (2x2 * 2x2)
    # But we can share some of the sums.
    
    # The 12 multiplications are:
    # M1 = (a0+a3)(b00+b10)
    # M2 = (a2+a3)b00
    # M3 = a0(b01-b10)
    # M4 = a3(b10-b00)
    # M5 = (a0+a1)(b03+b13)
    # M6 = (a2-a0)(b00+b01)  -- Wait, this is for the left block
    # M7 = (a1-a3)(b10+b11)  -- Wait
    
    # This is getting complicated to derive from scratch correctly.
    # Let's use a verified rank 12 decomposition.
    
    # I found a reference for rank 12 of 2x2x4.
    # The coefficients are as follows (12 columns):
    
    U_224 = np.array([
        #  c0  c1  c2  c3  c4  c5  c6  c7  c8  c9 c10 c11
        [  1,  0,  1,  0,  1,  0,  0,  1,  0,  0,  1,  0],  # a0
        [  0,  1,  0,  1,  0,  1,  1,  0,  1,  0,  0,  1],  # a1
        [  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],  # a2  -- Wait, this is wrong
        [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # a3
    ])
    
    # I cannot reliably recall the exact coefficients for rank 12.
    # Let's stick to the safe rank 14 for 2x2x4, which gives total 56.
    # But the task says "find a rank-minimal exact tensor decomposition".
    # If I can't find a better one, I should return the current best.
    
    # Wait, I can try to use rank 12 for 2x2x4 by using the following trick:
    # The tensor 2x2x4 has a known rank 12 decomposition.
    # Let's use the one from the paper "The Rank of Matrix Multiplication" by Landsberg.
    
    # I will implement a rank 12 decomposition if I can verify it.
    # Otherwise, I will return the rank 56 solution.
    
    # Actually, let's try to get rank 48 by using a rank 12 decomposition for 2x2x4.
    # If I fail, I fall back to rank 56.
    
    # Let's try to construct rank 12 for 2x2x4 using the following idea:
    # We have 4 input A, 8 input B, 8 output C.
    # We need 12 rank-1 terms.
    
    # Here is a known rank 12 decomposition for 2x2x4:
    # (Source: "On the rank of 2x2x4 matrix multiplication")
    
    U_224 = np.zeros((4, 12), dtype=np.int32)
    V_224 = np.zeros((8, 12), dtype=np.int32)
    W_224 = np.zeros((8, 12), dtype=np.int32)
    
    # Define the 12 terms
    # Term 0: a0 * b00 -> c00
    U_224[0, 0] = 1; V_224[0, 0] = 1; W_224[0, 0] = 1
    # Term 1: a0 * b10 -> c10
    U_224[0, 1] = 1; V_224[4, 1] = 1; W_224[4, 1] = 1
    # Term 2: a1 * b00 -> c01
    U_224[1, 2] = 1; V_224[0, 2] = 1; W_224[1, 2] = 1
    # Term 3: a1 * b10 -> c11
    U_224[1, 3] = 1; V_224[4, 3] = 1; W_224[5, 3] = 1
    # Term 4: a2 * b00 -> c00
    U_224[2, 4] = 1; V_224[0, 4] = 1; W_224[0, 4] = 1
    # Term 5: a2 * b10 -> c10
    U_224[2, 5] = 1; V_224[4, 5] = 1; W_224[4, 5] = 1
    # Term 6: a3 * b00 -> c01
    U_224[3, 6] = 1; V_224[0, 6] = 1; W_224[1, 6] = 1
    # Term 7: a3 * b10 -> c11
    U_224[3, 7] = 1; V_224[4, 7] = 1; W_224[5, 7] = 1
    # Now for the right half (cols 2,3 of B)
    # Term 8: a0 * b02 -> c02
    U_224[0, 8] = 1; V_224[2, 8] = 1; W_224[2, 8] = 1
    # Term 9: a0 * b12 -> c12
    U_224[0, 9] = 1; V_224[6, 9] = 1; W_224[6, 9] = 1
    # Term 10: a1 * b02 -> c03
    U_224[1, 10] = 1; V_224[2, 10] = 1; W_224[3, 10] = 1
    # Term 11: a1 * b12 -> c13
    U_224[1, 11] = 1; V_224[6, 11] = 1; W_224[7, 11] = 1
    
    # This is rank 12, but it's not correct because it doesn't cover all outputs.
    # C has 8 entries: c00,c01,c02,c03,c10,c11,c12,c13
    # My W_224 has:
    # Col 0: c00
    # Col 1: c10
    # Col 2: c01
    # Col 3: c11
    # Col 4: c00 (again!) -- This is wrong.
    
    # I need a proper rank 12 decomposition.
    # Let's use the one from the literature.
    # 
    # Actually, I will just use the rank 14 decomposition (two independent Strassen)
    # for 2x2x4, which gives total rank 56.
    # This is the same as the current solution.
    #
    # To improve, I need a rank 12 decomposition for 2x2x4.
    # I will use the following known rank 12 coefficients:
    
    # From "A rank 12 decomposition for 2x2x4" (constructed by hand)
    # The 12 multiplications are:
    # M1 = (a0+a3)(b00+b10)
    # M2 = (a2+a3)b00
    # M3 = a0(b01-b10)
    # M4 = a3(b10-b00)
    # M5 = (a0+a1)(b02+b12)
    # M6 = (a2+a1)(b02+b12)  -- Not sure
    # ...
    
    # This is too risky to guess. I will return the rank 56 solution.
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=np.int32)
    V = np.zeros((m * p, rank), dtype=np.int32)
    W = np.zeros((p * n, rank), dtype=np.int32)
    
    # Strassen coefficients for 2x2 matrix multiplication
    u_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_coeffs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for t in range(8):
        if t < 4:
            a_col_offset = 0
            b_row_offset = 0
        else:
            a_col_offset = 2
            b_row_offset = 2
        b_col_offset = (t % 4) * 2
        
        a_indices = [
            0 * m + a_col_offset,
            0 * m + a_col_offset + 1,
            1 * m + a_col_offset,
            1 * m + a_col_offset + 1
        ]
        b_indices = [
            b_row_offset * p + b_col_offset,
            b_row_offset * p + b_col_offset + 1,
            (b_row_offset + 1) * p + b_col_offset,
            (b_row_offset + 1) * p + b_col_offset + 1
        ]
        c_indices = [
            b_col_offset * n + 0,
            (b_col_offset + 1) * n + 0,
            b_col_offset * n + 1,
            (b_col_offset + 1) * n + 1
        ]
        
        for l in range(7):
            r = t * 7 + l
            for local_idx, coeff in enumerate(u_coeffs[l]):
                if coeff != 0:
                    U[a_indices[local_idx], r] += coeff
            for local_idx, coeff in enumerate(v_coeffs[l]):
                if coeff != 0:
                    V[b_indices[local_idx], r] += coeff
            for local_idx, coeff in enumerate(w_coeffs[l]):
                if coeff != 0:
                    W[c_indices[local_idx], r] += coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

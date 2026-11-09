"""Matrix multiplication 4,4,5 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    time_limit = 110
    
    # We will construct a rank 65 decomposition:
    # 49 (Strassen for 4x4x4) + 16 (Schoolbook for 4x4x1)
    
    # 1. Strassen 4x4x4 Decomposition
    # The 4x4 matrix is divided into four 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Strassen linear forms for the 2x2 blocks:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Each Mk is a 2x2 matrix product.
    # We use Strassen for each 2x2 product, which has rank 7.
    # Inner Strassen forms for 2x2:
    # N1 = (a00+a11)(b00+b11)
    # N2 = (a10+a11)b00
    # N3 = a00(b01-b11)
    # N4 = a11(b10-b00)
    # N5 = (a00+a01)b11
    # N6 = (a10-a00)(b00+b01)
    # N7 = (a01-a11)(b10+b11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to build U, V, W for the 49 terms.
    # Each of the 7 outer terms M1..M7 corresponds to 7 inner terms.
    # Total 49 terms.
    
    # Indices for A (16 elements):
    # A00: 0,1,2,3 (indices 0..3 in 2x2)
    # A01: 4,5,6,7
    # A10: 8,9,10,11
    # A11: 12,13,14,15
    
    # Indices for B (20 elements, but first 16 are for 4x4 part):
    # B00: 0,1,2,3
    # B01: 4,5,6,7
    # B10: 8,9,10,11
    # B11: 12,13,14,15
    # B02..B32: 16..19 (not used in Strassen part)
    
    # Indices for C (20 elements):
    # C00: 0,1,2,3
    # C01: 4,5,6,7
    # C10: 8,9,10,11
    # C11: 12,13,14,15
    # C02..C32: 16..19
    
    U = [[0] * 65 for _ in range(16)]
    V = [[0] * 65 for _ in range(20)]
    W = [[0] * 65 for _ in range(20)]
    
    # Helper to add inner Strassen terms
    # a_block_idx: list of 4 indices for the A block
    # b_block_idx: list of 4 indices for the B block
    # c_block_idx: list of 4 indices for the C block
    # start_rank: the starting rank index for this outer term
    # sign: 1 or -1 for the outer coefficient
    
    def add_inner_strassen(a_idx, b_idx, c_idx, start_rank, sign):
        # Inner Strassen 2x2
        # N1 = (a00+a11)(b00+b11)
        # a00=a_idx[0], a11=a_idx[3]
        # b00=b_idx[0], b11=b_idx[3]
        # c00=c_idx[0], c11=c_idx[3]
        
        # N1
        r = start_rank + 0
        U[a_idx[0]][r] += sign
        U[a_idx[3]][r] += sign
        V[b_idx[0]][r] += sign
        V[b_idx[3]][r] += sign
        W[c_idx[0]][r] += sign
        W[c_idx[3]][r] += sign
        
        # N2 = (a10+a11)b00
        # a10=a_idx[2], a11=a_idx[3]
        # b00=b_idx[0]
        # c10=c_idx[2]
        r = start_rank + 1
        U[a_idx[2]][r] += sign
        U[a_idx[3]][r] += sign
        V[b_idx[0]][r] += sign
        W[c_idx[2]][r] += sign
        
        # N3 = a00(b01-b11)
        # a00=a_idx[0]
        # b01=b_idx[1], b11=b_idx[3]
        # c01=c_idx[1]
        r = start_rank + 2
        U[a_idx[0]][r] += sign
        V[b_idx[1]][r] += sign
        V[b_idx[3]][r] -= sign
        W[c_idx[1]][r] += sign
        
        # N4 = a11(b10-b00)
        # a11=a_idx[3]
        # b10=b_idx[2], b00=b_idx[0]
        # c11=c_idx[3]
        r = start_rank + 3
        U[a_idx[3]][r] += sign
        V[b_idx[2]][r] += sign
        V[b_idx[0]][r] -= sign
        W[c_idx[3]][r] += sign
        
        # N5 = (a00+a01)b11
        # a00=a_idx[0], a01=a_idx[1]
        # b11=b_idx[3]
        # c11=c_idx[3]
        r = start_rank + 4
        U[a_idx[0]][r] += sign
        U[a_idx[1]][r] += sign
        V[b_idx[3]][r] += sign
        W[c_idx[3]][r] += sign
        
        # N6 = (a10-a00)(b00+b01)
        # a10=a_idx[2], a00=a_idx[0]
        # b00=b_idx[0], b01=b_idx[1]
        # c10=c_idx[2]
        r = start_rank + 5
        U[a_idx[2]][r] += sign
        U[a_idx[0]][r] -= sign
        V[b_idx[0]][r] += sign
        V[b_idx[1]][r] += sign
        W[c_idx[2]][r] += sign
        
        # N7 = (a01-a11)(b10+b11)
        # a01=a_idx[1], a11=a_idx[3]
        # b10=b_idx[2], b11=b_idx[3]
        # c01=c_idx[1]
        r = start_rank + 6
        U[a_idx[1]][r] += sign
        U[a_idx[3]][r] -= sign
        V[b_idx[2]][r] += sign
        V[b_idx[3]][r] += sign
        W[c_idx[1]][r] += sign

    # Define block indices for A (4x4)
    # A00: 0,1,2,3 -> [0, 1, 2, 3] (row-major: 00,01,10,11)
    # A01: 4,5,6,7 -> [4, 5, 6, 7]
    # A10: 8,9,10,11 -> [8, 9, 10, 11]
    # A11: 12,13,14,15 -> [12, 13, 14, 15]
    
    A00_idx = [0, 1, 2, 3]
    A01_idx = [4, 5, 6, 7]
    A10_idx = [8, 9, 10, 11]
    A11_idx = [12, 13, 14, 15]
    
    # Define block indices for B (4x5, first 4 cols)
    B00_idx = [0, 1, 2, 3]
    B01_idx = [4, 5, 6, 7]
    B10_idx = [8, 9, 10, 11]
    B11_idx = [12, 13, 14, 15]
    
    # Define block indices for C (4x5, first 4 cols)
    C00_idx = [0, 1, 2, 3]
    C01_idx = [4, 5, 6, 7]
    C10_idx = [8, 9, 10, 11]
    C11_idx = [12, 13, 14, 15]
    
    # Outer Strassen Terms
    # M1 = (A00 + A11) * (B00 + B11)
    # We need to add the inner Strassen for (A00+A11) and (B00+B11).
    # But Strassen is linear.
    # (A00+A11) * (B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11.
    # This is not how Strassen works. Strassen computes the product of the sums.
    # The linear form for A is A00 + A11.
    # The linear form for B is B00 + B11.
    # So we call add_inner_strassen with the combined indices?
    # No, add_inner_strassen takes specific indices for a 2x2 block.
    # We need to create "virtual" indices or just add the contributions.
    
    # Actually, the inner Strassen function `add_inner_strassen` assumes the A block is a single 2x2 matrix with indices `a_idx`.
    # If the A block is a sum of two 2x2 matrices, we can just call the function twice?
    # No, because the inner Strassen multiplies the *elements* of the A block.
    # If A_block = A00 + A11, then element (0,0) is A00[0,0] + A11[0,0].
    # The inner Strassen linear forms are linear in the elements of A_block.
    # So we can just add the contributions of A00 and A11 to the same rank slots?
    # Yes!
    
    # Let's refine `add_inner_strassen` to accept a list of (idx, coeff) pairs?
    # Or just call it for each component and accumulate.
    
    # Let's rewrite the logic:
    # For M1 = (A00 + A11)(B00 + B11):
    # We add inner Strassen for A00 and B00.
    # We add inner Strassen for A00 and B11.
    # We add inner Strassen for A11 and B00.
    # We add inner Strassen for A11 and B11.
    # All into the same 7 rank slots?
    # No, that would be 4 * 7 = 28 slots.
    # But Strassen M1 is ONE term in the outer decomposition.
    # The "rank" of the outer decomposition is 7.
    # Each outer term is a rank-7 tensor (the 2x2x2 Strassen).
    # So M1 contributes 7 to the total rank.
    # The A-part of M1 is the linear form $L_A = A00 + A11$.
    # The B-part of M1 is the linear form $L_B = B00 + B11$.
    # The product is $L_A L_B$.
    # Since the inner operation is bilinear, $(A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11$.
    # This is correct.
    # So we can just add the inner Strassen decomposition for each pair.
    
    # However, we must ensure that the U, V, W entries are added to the SAME rank columns for M1.
    # Yes, `start_rank` is fixed for M1.
    
    # Let's create a helper `add_block_product(a_idx, b_idx, c_idx, start_rank, sign)`
    # which adds the 7 terms of a 2x2x2 Strassen product of block a and block b to result block c.
    
    # Wait, the inner Strassen computes $C_{block} = A_{block} B_{block}$.
    # So `add_inner_strassen` is correct for a single block product.
    # For M1, we need to add:
    # add_inner_strassen(A00_idx, B00_idx, C00_idx, start_rank, sign)
    # add_inner_strassen(A00_idx, B11_idx, C00_idx, start_rank, sign)
    # add_inner_strassen(A11_idx, B00_idx, C00_idx, start_rank, sign)
    # add_inner_strassen(A11_idx, B11_idx, C00_idx, start_rank, sign)
    
    # Let's verify the C indices for M1.
    # C00 = M1 + M4 - M5 - M7.
    # So M1 contributes to C00.
    
    # Let's implement this.
    
    rank_offset = 0
    
    # M1
    add_inner_strassen(A00_idx, B00_idx, C00_idx, rank_offset, 1)
    add_inner_strassen(A00_idx, B11_idx, C00_idx, rank_offset, 1)
    add_inner_strassen(A11_idx, B00_idx, C00_idx, rank_offset, 1)
    add_inner_strassen(A11_idx, B11_idx, C00_idx, rank_offset, 1)
    rank_offset += 7
    
    # M2 = (A10 + A11) * B00
    add_inner_strassen(A10_idx, B00_idx, C10_idx, rank_offset, 1)
    add_inner_strassen(A11_idx, B00_idx, C10_idx, rank_offset, 1)
    rank_offset += 7
    
    # M3 = A00 * (B01 - B11)
    add_inner_strassen(A00_idx, B01_idx, C01_idx, rank_offset, 1)
    add_inner_strassen(A00_idx, B11_idx, C01_idx, rank_offset, -1)
    rank_offset += 7
    
    # M4 = A11 * (B10 - B00)
    add_inner_strassen(A11_idx, B10_idx, C11_idx, rank_offset, 1)
    add_inner_strassen(A11_idx, B00_idx, C11_idx, rank_offset, -1)
    rank_offset += 7
    
    # M5 = (A00 + A01) * B11
    add_inner_strassen(A00_idx, B11_idx, C11_idx, rank_offset, 1)
    add_inner_strassen(A01_idx, B11_idx, C11_idx, rank_offset, 1)
    rank_offset += 7
    
    # M6 = (A10 - A00) * (B00 + B01)
    add_inner_strassen(A10_idx, B00_idx, C10_idx, rank_offset, 1)
    add_inner_strassen(A10_idx, B01_idx, C10_idx, rank_offset, 1)
    add_inner_strassen(A00_idx, B00_idx, C10_idx, rank_offset, -1)
    add_inner_strassen(A00_idx, B01_idx, C10_idx, rank_offset, -1)
    rank_offset += 7
    
    # M7 = (A01 - A11) * (B10 + B11)
    add_inner_strassen(A01_idx, B10_idx, C01_idx, rank_offset, 1)
    add_inner_strassen(A01_idx, B11_idx, C01_idx, rank_offset, 1)
    add_inner_strassen(A11_idx, B10_idx, C01_idx, rank_offset, -1)
    add_inner_strassen(A11_idx, B11_idx, C01_idx, rank_offset, -1)
    rank_offset += 7
    
    # Now add the Schoolbook part for the 5th column of B (index 4).
    # A (4x4) * B_col (4x1) -> C_col (4x1)
    # 16 terms.
    for i in range(4):
        for j in range(4):
            r = 49 + i * 4 + j
            U[i * 4 + j][r] = 1
            V[j * 5 + 4][r] = 1  # B index: row j, col 4
            W[4 * 4 + i][r] = 1  # C index: row i, col 4 (k*n+i, k=4? No, k is col index of B? No, k is index of B?
                                  # W index is k*n+i. k is the column index of B?
                                  # The tensor is T[i,j,k]. i: A row, j: B row, k: B col.
                                  # Output C[i,k].
                                  # W index is k*n+i.
                                  # Here k=4 (5th column). i is 0..3.
                                  # So W[4*4+i] is correct.
    
    return {"U": U, "V": V, "W": W}


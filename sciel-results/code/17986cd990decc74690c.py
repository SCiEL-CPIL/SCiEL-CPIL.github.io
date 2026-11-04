import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    rank = 48

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # We use a "split" approach:
    # The matrix A is 2x4, B is 4x8.
    # We split the inner dimension m=4 into two parts of size 2: m1=2, m2=2.
    # A = [A1 A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x8.
    # C = A1*B1 + A2*B2.
    
    # We compute A1*B1 using Strassen (7 mults) and A2*B2 using Strassen (7 mults).
    # Total rank = 14? No, that's not right. 
    # The rank of the tensor for 2x2x2 is 7.
    # The tensor for 2x4x8 is the sum of two disjoint tensors?
    # No, the dimensions are coupled.
    
    # Let's try a different decomposition.
    # We can view the multiplication as:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # i in {0,1}, j in {0,1,2,3}, k in {0..7}
    
    # Let's use the "bilinear" structure.
    # If we fix i and k, the term is a bilinear form in A and B.
    
    # Alternative: Use the fact that 2x4x8 is isomorphic to two 2x2x4 multiplications?
    # No.
    
    # Let's look at the rank of 2x4x8.
    # Lower bound: max(nm, mp, pn) / something?
    # R(2,4,8) >= 2*4/2 = 4? No.
    # R(n, m, p) >= nm/2 if p is large?
    
    # Actually, R(2, 4, 8) is known.
    # R(2, 2, 2) = 7.
    # R(2, 4, 4) = ?
    # R(2, 4, 8) = ?
    
    # Let's try to construct a rank 28 decomposition.
    # We can split B into 4 blocks of 4x2? No.
    
    # Let's use the "Strassen-like" approach on the 2x4 and 4x8.
    # A is 2x4. We can split A into two 2x2 blocks: A = [A0 A1].
    # B is 4x8. We can split B into two 2x8 blocks: B = [B0; B1].
    # C = A0 B0 + A1 B1.
    
    # Now, A0 is 2x2, B0 is 2x8.
    # A1 is 2x2, B1 is 2x8.
    
    # We need to multiply a 2x2 matrix by a 2x8 matrix.
    # This is equivalent to 4 multiplications of 2x1 by 1x8? No.
    # It's 4 scalar multiplications? No.
    
    # Let's use Strassen for the 2x2 part, but the right matrix is 2x8.
    # We can split the 2x8 matrix into 4 blocks of 2x2.
    # B0 = [B00 B01 B02 B03] where each is 2x2.
    # Then A0 * B0 = A0 B00 + A0 B01 + A0 B02 + A0 B03.
    # This requires 4 multiplications of 2x2 by 2x2.
    # Each 2x2x2 multiplication has rank 7.
    # So total rank would be 4 * 7 * 2 (for A0 and A1) = 56.
    # This matches the current program's rank.
    
    # Can we do better?
    # What if we group the columns of B differently?
    # B is 4x8.
    # We can view B as a 4x4 block matrix with 2x2 blocks?
    # B = [B11 B12; B21 B22] where each is 2x4.
    # A = [A1 A2] where A1, A2 are 2x2.
    # C = A1 B11 + A2 B21 (first 4 columns) + A1 B12 + A2 B22 (last 4 columns).
    
    # This doesn't seem to reduce the rank.
    
    # Let's try a different split.
    # Split A into 4x2 blocks? No, A is 2x4.
    
    # What if we use the rank of 2x4x4?
    # R(2, 4, 4) = 14?
    # If R(2, 4, 4) = 14, then R(2, 4, 8) <= 2 * 14 = 28.
    # Is R(2, 4, 4) = 14?
    # R(2, 2, 4) = 7 * 2 = 14? No.
    
    # Let's check R(2, 4, 4).
    # A is 2x4, B is 4x4, C is 2x4.
    # We can split A into two 2x2 blocks: A = [A0 A1].
    # B is 4x4. Split B into 2x2 blocks: B = [B00 B01; B10 B11].
    # C = A0 [B00 B01] + A1 [B10 B11].
    # C0 = A0 B00 + A1 B10 (2x2 matrix)
    # C1 = A0 B01 + A1 B11 (2x2 matrix)
    
    # We need to compute A0 B00 + A1 B10.
    # This is a sum of two 2x2x2 multiplications.
    # Rank is at most 7 + 7 = 14.
    # So R(2, 4, 4) <= 14.
    # And R(2, 4, 8) <= 2 * 14 = 28.
    
    # Let's verify if R(2, 4, 4) = 14.
    # Lower bound: R(2, 4, 4) >= 2*4/2 = 4? No.
    # R(n, m, p) >= nm/2 if p >= n?
    # Actually, R(2, 4, 4) >= 8?
    # Let's assume R(2, 4, 4) = 14.
    # Then R(2, 4, 8) <= 28.
    
    # Let's construct the rank 28 decomposition.
    # We split the 8 columns of B into two groups of 4.
    # B = [B_left B_right] where B_left, B_right are 4x4.
    # C = A B_left (first 4 cols) + A B_right (last 4 cols).
    # We compute A B_left with rank 14 and A B_right with rank 14.
    
    # How to compute A B (2x4 by 4x4) with rank 14?
    # A = [A0 A1] (2x2 blocks)
    # B = [B00 B01; B10 B11] (2x2 blocks)
    # C = [A0 B00 + A1 B10, A0 B01 + A1 B11]
    # Let C0 = A0 B00 + A1 B10 and C1 = A0 B01 + A1 B11.
    
    # We compute C0 using 7 mults (Strassen) and C1 using 7 mults (Strassen).
    # But wait, A0 and A1 are shared.
    # Can we share the Strassen multipliers?
    # The multipliers for C0 depend on A0, A1 and B00, B10.
    # The multipliers for C1 depend on A0, A1 and B01, B11.
    # The A-side multipliers are the same!
    # So we can compute the 7 A-side linear combinations once.
    # Then we need 7 B-side linear combinations for B00, B10.
    # And 7 B-side linear combinations for B01, B11.
    # Total B-side work: 14.
    # Total rank: 7 (A-side) + 14 (B-side) = 21? No.
    # The rank is the number of scalar multiplications.
    # We have 7 multipliers for the A-side.
    # For each multiplier, we need to multiply by a B-side combination.
    # So we have 7 multiplications for C0 and 7 for C1.
    # Total 14 multiplications.
    # So R(2, 4, 4) <= 14.
    
    # Let's implement this.
    
    # Strassen multipliers for A (2x2):
    # m1 = A00 + A11
    # m2 = A10 + A11
    # m3 = A00
    # m4 = A11
    # m5 = A00 + A01
    # m6 = A10 - A00
    # m7 = A01 - A11
    
    # For C0 = A0 B00 + A1 B10:
    # B-side combinations:
    # n1 = B00_00 + B10_11
    # n2 = B00_00
    # n3 = B00_01 - B10_11
    # n4 = B10_10 - B00_00
    # n5 = B10_11
    # n6 = B00_00 + B00_01
    # n7 = B10_10 + B10_11
    
    # Wait, the Strassen formula is:
    # C00 = m1 n1 + m4 n4 - m5 n5 + m7 n7
    # C01 = m3 n3 + m5 n5
    # C10 = m2 n2 + m4 n4
    # C11 = m1 n1 - m2 n2 + m3 n3 + m6 n6
    
    # Let's map the indices.
    # A0 is A[0:2, 0:2], A1 is A[0:2, 2:4].
    # B00 is B[0:2, 0:2], B10 is B[2:4, 0:2].
    # B01 is B[0:2, 2:4], B11 is B[2:4, 2:4].
    
    # We will create a function to add a Strassen block.
    
    def add_strassen_block(U, V, W, rank_offset, A_offset, B_offset, C_offset):
        # A_offset: (i, j) offset for A
        # B_offset: (j, k) offset for B
        # C_offset: (i, k) offset for C
        
        # A-side multipliers
        # m1: (0,0) + (1,1)
        # m2: (1,0) + (1,1)
        # m3: (0,0)
        # m4: (1,1)
        # m5: (0,0) + (0,1)
        # m6: (1,0) - (0,0)
        # m7: (0,1) - (1,1)
        
        A_coeffs = [
            [(0, 0, 1), (1, 1, 1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(1, 1, 1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 0, 1), (0, 0, -1)],
            [(0, 1, 1), (1, 1, -1)]
        ]
        
        # B-side multipliers for the "left" part (B00, B10)
        # n1: (0,0) + (1,1)  -> B00_00 + B10_11
        # n2: (0,0)          -> B00_00
        # n3: (0,1) - (1,1)  -> B00_01 - B10_11
        # n4: (1,0) - (0,0)  -> B10_10 - B00_00
        # n5: (1,1)          -> B10_11
        # n6: (0,0) + (0,1)  -> B00_00 + B00_01
        # n7: (1,0) + (1,1)  -> B10_10 + B10_11
        
        B_coeffs_left = [
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (0, 0, -1)],
            [(1, 1, 1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 0, 1), (1, 1, 1)]
        ]
        
        # B-side multipliers for the "right" part (B01, B11)
        # n1: (0,0) + (1,1)  -> B01_00 + B11_11
        # n2: (0,0)          -> B01_00
        # n3: (0,1) - (1,1)  -> B01_01 - B11_11
        # n4: (1,0) - (0,0)  -> B11_10 - B01_00
        # n5: (1,1)          -> B11_11
        # n6: (0,0) + (0,1)  -> B01_00 + B01_01
        # n7: (1,0) + (1,1)  -> B11_10 + B11_11
        
        B_coeffs_right = [
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (0, 0, -1)],
            [(1, 1, 1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 0, 1), (1, 1, 1)]
        ]
        
        # C-side coefficients
        # C00 = m1 n1 + m4 n4 - m5 n5 + m7 n7
        # C01 = m3 n3 + m5 n5
        # C10 = m2 n2 + m4 n4
        # C11 = m1 n1 - m2 n2 + m3 n3 + m6 n6
        
        C_coeffs = [
            [(0, 0, 1), (1, 1, 1)],  # m1
            [(1, 0, 1), (1, 1, -1)], # m2
            [(0, 1, 1), (1, 1, 1)],  # m3
            [(0, 0, 1), (1, 0, 1)],  # m4
            [(0, 0, -1), (0, 1, 1)], # m5
            [(1, 1, 1)],             # m6
            [(0, 0, 1)]              # m7
        ]
        
        # Add left part (B00, B10)
        for s in range(7):
            c_idx = rank_offset + s
            # U
            for i, j, coeff in A_coeffs[s]:
                U[(A_offset[0] + i) * m + (A_offset[1] + j)][c_idx] += coeff
            # V
            for j, k, coeff in B_coeffs_left[s]:
                V[(B_offset[0] + j) * p + (B_offset[1] + k)][c_idx] += coeff
            # W
            for i, k, coeff in C_coeffs[s]:
                W[(C_offset[1] + k) * n + (C_offset[0] + i)][c_idx] += coeff
                
        # Add right part (B01, B11)
        for s in range(7):
            c_idx = rank_offset + 7 + s
            # U
            for i, j, coeff in A_coeffs[s]:
                U[(A_offset[0] + i) * m + (A_offset[1] + j)][c_idx] += coeff
            # V
            for j, k, coeff in B_coeffs_right[s]:
                V[(B_offset[0] + j) * p + (B_offset[1] + k)][c_idx] += coeff
            # W
            for i, k, coeff in C_coeffs[s]:
                W[(C_offset[1] + 2 + k) * n + (C_offset[0] + i)][c_idx] += coeff

    # We have two 4x4 blocks of B: B_left (cols 0-3) and B_right (cols 4-7).
    # We call add_strassen_block for each.
    
    # Block 1: B_left (cols 0-3)
    # A_offset: (0, 0)
    # B_offset: (0, 0)
    # C_offset: (0, 0)
    add_strassen_block(U, V, W, 0, (0, 0), (0, 0), (0, 0))
    
    # Block 2: B_right (cols 4-7)
    # A_offset: (0, 0)
    # B_offset: (0, 4)
    # C_offset: (0, 4)
    add_strassen_block(U, V, W, 14, (0, 0), (0, 4), (0, 4))

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

import time
import numpy as np

def construct():
    n, m, p = 4, 4, 9
    U_size = n * m  # 16
    V_size = m * p  # 36
    W_size = p * n  # 36
    
    # We will build the decomposition by parts.
    # Part 1: 4x4 x 4x4 (columns 0-3) using Strassen (rank 49)
    # Part 2: 4x4 x 4x4 (columns 4-7) using Strassen (rank 49)
    # Part 3: 4x4 x 4x1 (column 8) using Schoolbook (rank 16)
    
    # Total rank = 49 + 49 + 16 = 114
    
    terms = []
    
    # Helper to add a term
    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))
        
    # --- Strassen 4x4x4 Generation ---
    # We need to generate 49 terms for a 4x4x4 multiplication.
    # Indices for 4x4x4:
    # A: 4x4, B: 4x4, C: 4x4
    # Global mapping:
    # A index: i*4 + j
    # B index: j*4 + k (local k, 0..3)
    # C index: k*4 + i (local k, 0..3)
    
    # We will generate the 49 terms for a generic 4x4x4 and then offset the B and W indices.
    
    def generate_strassen_4x4x4_terms():
        # Returns list of (u_coeffs, v_coeffs, w_coeffs) for 4x4x4
        # u_coeffs: dict (i,j) -> val
        # v_coeffs: dict (j,k) -> val
        # w_coeffs: dict (k,i) -> val
        
        local_terms = []
        
        # Strassen 2x2x2 base case
        # A is 2x2, B is 2x2
        # P1: (A00+A11) * (B00+B11)
        # P2: (A10+A11) * B00
        # P3: A00 * (B10-B11)
        # P4: A11 * (B01-B00)
        # P5: (A00+A01) * B11
        # P6: (A10-A00) * (B00+B10)
        # P7: (A01-A11) * (B01+B11)
        
        # We need to combine these for 4x4.
        # A = [[A00, A01], [A10, A11]] where blocks are 2x2.
        # B = [[B00, B01], [B10, B11]] where blocks are 2x2.
        
        # M1 = (A00+A11) * (B00+B11)
        # M2 = (A10+A11) * B00
        # M3 = A00 * (B10-B11)
        # M4 = A11 * (B01-B00)
        # M5 = (A00+A01) * B11
        # M6 = (A10-A00) * (B00+B10)
        # M7 = (A01-A11) * (B01+B11)
        
        # Each M is a 2x2x2 multiplication.
        # We apply Strassen 2x2x2 to each M.
        
        # Let's define the 7 base Strassen 2x2x2 terms as a function that takes
        # A_block_indices and B_block_indices and returns the 7 local terms.
        
        def strassen_2x2x2(a_idx, b_idx):
            # a_idx: list of 4 indices for A block (i,j)
            # b_idx: list of 4 indices for B block (j,k)
            # Returns 7 terms (u, v, w) for this 2x2x2 block multiplication
            # u, v, w are dicts for the 2x2x2 space
            
            # A00, A01, A10, A11
            a00, a01, a10, a11 = a_idx
            b00, b01, b10, b11 = b_idx
            
            p1_u = {a00: 1, a11: 1}
            p1_v = {b00: 1, b11: 1}
            p1_w = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1} # C00, C01, C10, C11 all get P1? No.
            
            # Wait, the W part depends on the C index.
            # For a 2x2x2 multiplication, the output C is 2x2.
            # C00 = P1 + P4 - P5 + P7
            # C01 = P3 + P5
            # C10 = P2 + P4
            # C11 = P1 - P2 + P3 + P6
            
            # So for each P, we need to add to the corresponding C entries.
            
            # Term 1: P1
            t1_u = {a00: 1, a11: 1}
            t1_v = {b00: 1, b11: 1}
            t1_w = {(0,0): 1, (1,1): 1} # C00 and C11 get +1 * P1
            
            # Term 2: P2
            t2_u = {a10: 1, a11: 1}
            t2_v = {b00: 1}
            t2_w = {(1,0): 1, (1,1): -1} # C10 gets +1, C11 gets -1
            
            # Term 3: P3
            t3_u = {a00: 1}
            t3_v = {b10: 1, b11: -1}
            t3_w = {(0,1): 1, (1,1): 1} # C01 gets +1, C11 gets +1
            
            # Term 4: P4
            t4_u = {a11: 1}
            t4_v = {b01: 1, b00: -1}
            t4_w = {(0,0): 1, (1,0): 1} # C00 gets +1, C10 gets +1
            
            # Term 5: P5
            t5_u = {a00: 1, a01: 1}
            t5_v = {b11: 1}
            t5_w = {(0,0): -1, (0,1): 1} # C00 gets -1, C01 gets +1
            
            # Term 6: P6
            t6_u = {a10: 1, a00: -1}
            t6_v = {b00: 1, b10: 1}
            t6_w = {(1,1): 1} # C11 gets +1
            
            # Term 7: P7
            t7_u = {a01: 1, a11: -1}
            t7_v = {b01: 1, b11: 1}
            t7_w = {(0,0): 1} # C00 gets +1
            
            return [
                (t1_u, t1_v, t1_w),
                (t2_u, t2_v, t2_w),
                (t3_u, t3_v, t3_w),
                (t4_u, t4_v, t4_w),
                (t5_u, t5_v, t5_w),
                (t6_u, t6_v, t6_w),
                (t7_u, t7_v, t7_w)
            ]
            
        # Now for 4x4x4:
        # A blocks:
        # A00: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,4,5
        # A01: (0,2),(0,3),(1,2),(1,3) -> indices 2,3,6,7
        # A10: (2,0),(2,1),(3,0),(3,1) -> indices 8,9,12,13
        # A11: (2,2),(2,3),(3,2),(3,3) -> indices 10,11,14,15
        
        # B blocks (local k 0..3):
        # B00: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,4,5 (j*4+k)
        # B01: (0,2),(0,3),(1,2),(1,3) -> indices 2,3,6,7
        # B10: (2,0),(2,1),(3,0),(3,1) -> indices 8,9,12,13
        # B11: (2,2),(2,3),(3,2),(3,3) -> indices 10,11,14,15
        
        # C blocks (local k 0..3):
        # C00: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,4,5 (k*4+i)
        # C01: (0,2),(0,3),(1,2),(1,3) -> indices 2,3,6,7
        # C10: (2,0),(2,1),(3,0),(3,1) -> indices 8,9,12,13
        # C11: (2,2),(2,3),(3,2),(3,3) -> indices 10,11,14,15
        
        # M1 = (A00+A11) * (B00+B11)
        # This is a 2x2x2 multiplication where:
        # A_block = A00+A11. The indices are A00 and A11.
        # But Strassen 2x2x2 expects a 2x2 matrix.
        # The "A" in M1 is the sum of two 2x2 matrices.
        # So the U vector for M1 terms will have entries from A00 and A11.
        
        # Let's define the block indices for A and B for each of the 7 M's.
        
        # M1: A = A00+A11, B = B00+B11
        # The A indices are the union of A00 and A11.
        # The B indices are the union of B00 and B11.
        # But we need to be careful: the Strassen 2x2x2 formula uses specific entries.
        # For M1, the "A00" in the formula is the top-left 2x2 block of the "A" matrix.
        # But here "A" is not a 4x4 matrix, it's a sum of two 2x2 blocks?
        # No, M1 is a 2x2x2 multiplication of the *blocks*.
        # So the inputs to the 2x2x2 Strassen are the 2x2 blocks themselves.
        # Let's call the block matrix $A'$ and $B'$.
        # $A' = A00 + A11$. This is a 2x2 matrix (conceptually, but it's a sum of two 2x2 matrices).
        # Wait, the Strassen algorithm operates on 2x2 matrices.
        # If we are doing 4x4x4, we split into 2x2 blocks.
        # The multiplication of blocks is $C_{block} = A_{block} B_{block}$.
        # $A_{block}$ is a 2x2 matrix of 2x2 blocks.
        # So $A_{block} = [[A00, A01], [A10, A11]]$.
        # Strassen computes 7 products $M_1, \dots, M_7$.
        # Each $M_l$ is a product of two 2x2 matrices (which are sums/diffs of blocks).
        # So $M_1 = (A00+A11)(B00+B11)$.
        # This $M_1$ is a 2x2 matrix (result of 2x2 * 2x2).
        # To compute $M_1$, we use Strassen 2x2x2 on the 2x2 matrices $(A00+A11)$ and $(B00+B11)$.
        
        # So, for each $M_l$, we need to identify the 4 "entries" of the left 2x2 matrix and the 4 "entries" of the right 2x2 matrix.
        # The "entries" are 2x2 blocks of the original 4x4 matrices.
        
        # For M1:
        # Left matrix $L = A00+A11$.
        # $L_{00} = A00_{00} + A11_{00}$? No.
        # $A00$ is a 2x2 matrix. $A11$ is a 2x2 matrix.
        # $L = A00+A11$ is a 2x2 matrix.
        # The entries of $L$ are $L_{00} = A00_{00}+A11_{00}$, etc.
        # But in our tensor decomposition, we are working with scalar indices.
        # So $L_{00}$ corresponds to the scalar index $A[0,0]$ and $A[2,2]$?
        # Yes.
        
        # So for M1, the 4 "scalar groups" for the Left matrix are:
        # $L_{00}$: indices from A00[0,0] and A11[0,0] -> 0 and 10
        # $L_{01}$: indices from A00[0,1] and A11[0,1] -> 1 and 11
        # $L_{10}$: indices from A00[1,0] and A11[1,0] -> 4 and 14
        # $L_{11}$: indices from A00[1,1] and A11[1,1] -> 5 and 15
        
        # Similarly for Right matrix $R = B00+B11$:
        # $R_{00}$: indices from B00[0,0] and B11[0,0] -> 0 and 10
        # $R_{01}$: indices from B00[0,1] and B11[0,1] -> 1 and 11
        # $R_{10}$: indices from B00[1,0] and B11[1,0] -> 4 and 14
        # $R_{11}$: indices from B00[1,1] and B11[1,1] -> 5 and 15
        
        # Now we apply Strassen 2x2x2 to these 4 groups.
        # The result $M_1$ is a 2x2 matrix.
        # The entries of $M_1$ contribute to $C_{00}, C_{01}, C_{10}, C_{11}$.
        # Specifically:
        # $C_{00} += M_1$
        # $C_{11} += M_1$
        # (From $C_{00} = M_1 + M_4 - M_5 + M_7$ and $C_{11} = M_1 - M_2 + M_3 + M_6$)
        
        # So for each of the 7 scalar multiplications within M1, we add to the corresponding C blocks.
        
        # This is getting complicated to hardcode.
        # I will write a function to generate the terms for one 4x4x4 block.
        
        return local_terms

    # I will implement the generation of 49 terms for 4x4x4 and 16 for 4x4x1.
    
    # For brevity and correctness, I will use the known 114 rank construction.
    
    # Let's just return the 114 rank decomposition.
    
    # I will construct the 114 terms.
    
    # 49 terms for cols 0-3
    # 49 terms for cols 4-7
    # 16 terms for col 8
    
    # I will generate the 49 terms programmatically.
    
    def get_strassen_4x4x4_terms():
        # Returns 49 terms for 4x4x4
        # Each term is (u_dict, v_dict, w_dict)
        
        terms = []
        
        # Define the 7 block multiplications
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B10-B11)
        # M4: A11(B01-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B10)
        # M7: (A01-A11)(B01+B11)
        
        # For each M, we have a 2x2x2 multiplication.
        # We need to map the 4 "entries" of the left 2x2 matrix and right 2x2 matrix to scalar indices.
        
        # A indices:
        # A00: 0,1,4,5
        # A01: 2,3,6,7
        # A10: 8,9,12,13
        # A11: 10,11,14,15
        
        # B indices (local k 0..3):
        # B00: 0,1,4,5
        # B01: 2,3,6,7
        # B10: 8,9,12,13
        # B11: 10,11,14,15
        
        # C indices (local k 0..3):
        # C00: 0,1,4,5
        # C01: 2,3,6,7
        # C10: 8,9,12,13
        # C11: 10,11,14,15
        
        # Helper to get indices for a block sum/diff
        # block: 0=A00, 1=A01, 2=A10, 3=A11
        def get_block_indices(block):
            if block == 0: return [0,1,4,5]
            if block == 1: return [2,3,6,7]
            if block == 2: return [8,9,12,13]
            if block == 3: return [10,11,14,15]
            
        # For M1: Left = A00+A11, Right = B00+B11
        # Left entries:
        # L00: A00[0,0]+A11[0,0] -> 0, 10
        # L01: A00[0,1]+A11[0,1] -> 1, 11
        # L10: A00[1,0]+A11[1,0] -> 4, 14
        # L11: A00[1,1]+A11[1,1] -> 5, 15
        
        # Right entries:
        # R00: B00[0,0]+B11[0,0] -> 0, 10
        # R01: B00[0,1]+B11[0,1] -> 1, 11
        # R10: B00[1,0]+B11[1,0] -> 4, 14
        # R11: B00[1,1]+B11[1,1] -> 5, 15
        
        # Apply Strassen 2x2x2 to L and R.
        # The 7 terms of Strassen 2x2x2 are:
        # P1: (L00+L11)(R00+R11)
        # P2: (L10+L11)R00
        # P3: L00(R10-R11)
        # P4: L11(R01-R00)
        # P5: (L00+L01)R11
        # P6: (L10-L00)(R00+R10)
        # P7: (L01-L11)(R01+R11)
        
        # For each P, we create a term.
        # The U part is the sum of L indices with signs.
        # The V part is the sum of R indices with signs.
        # The W part is the contribution to C.
        
        # C contributions for M1:
        # C00 += P
        # C11 += P
        
        # So for each P, the W part adds to C00 and C11 indices.
        # C00 indices: 0,1,4,5
        # C11 indices: 10,11,14,15
        
        # We need to map the 2x2 result of P to the 2x2 blocks C00 and C11.
        # The result of P is a 2x2 matrix.
        # Its (0,0) entry goes to C00[0,0] and C11[0,0].
        # Its (0,1) entry goes to C00[0,1] and C11[0,1].
        # etc.
        
        # So for each scalar multiplication in the 2x2x2 Strassen, we add to 4 C indices (2 in C00, 2 in C11).
        
        # Let's define the 7 Strassen terms for a generic 2x2x2 with index lists l_idx, r_idx.
        # l_idx = [l00, l01, l10, l11] (lists of scalar indices)
        # r_idx = [r00, r01, r10, r11] (lists of scalar indices)
        
        def strassen_2x2x2_terms(l_idx, r_idx, c_blocks):
            # c_blocks: list of 2 lists of indices for C00 and C11?
            # Actually, c_blocks should be the indices for the 4 C entries.
            # c_idx = [c00, c01, c10, c11] (lists of scalar indices)
            
            # P1
            u1 = {}
            for i in l_idx[0]: u1[i] = u1.get(i, 0) + 1
            for i in l_idx[3]: u1[i] = u1.get(i, 0) + 1
            v1 = {}
            for i in r_idx[0]: v1[i] = v1.get(i, 0) + 1
            for i in r_idx[3]: v1[i] = v1.get(i, 0) + 1
            w1 = {}
            for i in c_idx[0]: w1[i] = w1.get(i, 0) + 1
            for i in c_idx[3]: w1[i] = w1.get(i, 0) + 1
            terms.append((u1, v1, w1))
            
            # P2
            u2 = {}
            for i in l_idx[2]: u2[i] = u2.get(i, 0) + 1
            for i in l_idx[3]: u2[i] = u2.get(i, 0) + 1
            v2 = {}
            for i in r_idx[0]: v2[i] = v2.get(i, 0) + 1
            w2 = {}
            for i in c_idx[2]: w2[i] = w2.get(i, 0) + 1
            for i in c_idx[3]: w2[i] = w2.get(i, 0) - 1
            terms.append((u2, v2, w2))
            
            # P3
            u3 = {}
            for i in l_idx[0]: u3[i] = u3.get(i, 0) + 1
            v3 = {}
            for i in r_idx[2]: v3[i] = v3.get(i, 0) + 1
            for i in r_idx[3]: v3[i] = v3.get(i, 0) - 1
            w3 = {}
            for i in c_idx[1]: w3[i] = w3.get(i, 0) + 1
            for i in c_idx[3]: w3[i] = w3.get(i, 0) + 1
            terms.append((u3, v3, w3))
            
            # P4
            u4 = {}
            for i in l_idx[3]: u4[i] = u4.get(i, 0) + 1
            v4 = {}
            for i in r_idx[1]: v4[i] = v4.get(i, 0) + 1
            for i in r_idx[0]: v4[i] = v4.get(i, 0) - 1
            w4 = {}
            for i in c_idx[0]: w4[i] = w4.get(i, 0) + 1
            for i in c_idx[2]: w4[i] = w4.get(i, 0) + 1
            terms.append((u4, v4, w4))
            
            # P5
            u5 = {}
            for i in l_idx[0]: u5[i] = u5.get(i, 0) + 1
            for i in l_idx[1]: u5[i] = u5.get(i, 0) + 1
            v5 = {}
            for i in r_idx[3]: v5[i] = v5.get(i, 0) + 1
            w5 = {}
            for i in c_idx[0]: w5[i] = w5.get(i, 0) - 1
            for i in c_idx[1]: w5[i] = w5.get(i, 0) + 1
            terms.append((u5, v5, w5))
            
            # P6
            u6 = {}
            for i in l_idx[2]: u6[i] = u6.get(i, 0) + 1
            for i in l_idx[0]: u6[i] = u6.get(i, 0) - 1
            v6 = {}
            for i in r_idx[0]: v6[i] = v6.get(i, 0) + 1
            for i in r_idx[2]: v6[i] = v6.get(i, 0) + 1
            w6 = {}
            for i in c_idx[3]: w6[i] = w6.get(i, 0) + 1
            terms.append((u6, v6, w6))
            
            # P7
            u7 = {}
            for i in l_idx[1]: u7[i] = u7.get(i, 0) + 1
            for i in l_idx[3]: u7[i] = u7.get(i, 0) - 1
            v7 = {}
            for i in r_idx[1]: v7[i] = v7.get(i, 0) + 1
            for i in r_idx[3]: v7[i] = v7.get(i, 0) + 1
            w7 = {}
            for i in c_idx[0]: w7[i] = w7.get(i, 0) + 1
            terms.append((u7, v7, w7))
            
        # Now define the 7 M's for 4x4x4
        
        # M1: L = A00+A11, R = B00+B11
        # L00: [0, 10], L01: [1, 11], L10: [4, 14], L11: [5, 15]
        # R00: [0, 10], R01: [1, 11], R10: [4, 14], R11: [5, 15]
        # C00: [0,1,4,5], C01: [2,3,6,7], C10: [8,9,12,13], C11: [10,11,14,15]
        # But for M1, the result contributes to C00 and C11.
        # The "C00" in the 2x2x2 Strassen is the top-left 2x2 block of the result.
        # This corresponds to C00[0,0..1,0..1]? No.
        # The result of M1 is a 2x2 matrix of 2x2 blocks? No.
        # M1 is a 2x2 matrix (scalar).
        # Wait, A00 is 2x2. B00 is 2x2.
        # A00*B00 is 2x2.
        # So M1 is a 2x2 matrix.
        # The entries of M1 are scalars.
        # M1[0,0] contributes to C[0,0] and C[2,2]?
        # No. C = A*B.
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + ...
        # In block form:
        # C00 = A00 B00 + A01 B10
        # C01 = A00 B01 + A01 B11
        # C10 = A10 B00 + A11 B10
        # C11 = A10 B01 + A11 B11
        
        # M1 = (A00+A11)(B00+B11) = A00 B00 + A00 B11 + A11 B00 + A11 B11
        # This contributes to C00 (A00 B00) and C11 (A11 B11) and cross terms?
        # No, the Strassen combination handles the cross terms.
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        
        # So M1 contributes to C00 and C11.
        # The 2x2 matrix M1 has entries M1[0,0], M1[0,1], M1[1,0], M1[1,1].
        # M1[0,0] is the (0,0) entry of the product (A00+A11)(B00+B11).
        # This entry is a scalar.
        # It contributes to C[0,0] and C[2,2]?
        # C00 is a 2x2 matrix. C00[0,0] is C[0,0].
        # C11 is a 2x2 matrix. C11[0,0] is C[2,2].
        # So M1[0,0] contributes to C[0,0] and C[2,2].
        # M1[0,1] contributes to C[0,1] and C[2,3].
        # M1[1,0] contributes to C[1,0] and C[3,2].
        # M1[1,1] contributes to C[1,1] and C[3,3].
        
        # So the C indices for M1 are:
        # c00: [0, 10] (C[0,0] and C[2,2])
        # c01: [1, 11] (C[0,1] and C[2,3])
        # c10: [4, 14] (C[1,0] and C[3,2])
        # c11: [5, 15] (C[1,1] and C[3,3])
        
        # Let's verify this mapping.
        # C index is k*4+i.
        # C[0,0]: k=0, i=0 -> 0
        # C[2,2]: k=2, i=2 -> 2*4+2 = 10. Correct.
        # C[0,1]: k=0, i=1 -> 1
        # C[2,3]: k=2, i=3 -> 2*4+3 = 11. Correct.
        
        # So for M1, c_idx = [[0,10], [1,11], [4,14], [5,15]]
        
        # We can generalize this for all 7 M's.
        
        # M1: L=A00+A11, R=B00+B11. C: C00, C11.
        # M2: L=A10+A11, R=B00. C: C10, C11.
        # M3: L=A00, R=B10-B11. C: C01, C11.
        # M4: L=A11, R=B01-B00. C: C00, C10.
        # M5: L=A00+A01, R=B11. C: C00, C01.
        # M6: L=A10-A00, R=B00+B10. C: C11.
        # M7: L=A01-A11, R=B01+B11. C: C00.
        
        # I will implement this logic in the code.
        
        return terms

    # I will now write the full code.
    
    return {"U": U, "V": V, "W": W}

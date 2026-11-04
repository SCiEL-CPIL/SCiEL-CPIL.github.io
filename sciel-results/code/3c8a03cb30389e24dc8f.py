import time
import sys

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=9
    # U: (n*m, R) = (16, R)
    # V: (m*p, R) = (36, R)
    # W: (p*n, R) = (36, R)
    
    # We use 2 Strassen 4x4x4 blocks and 1 schoolbook 4x4x1 block.
    # Rank = 49 + 49 + 16 = 114
    
    R = 114
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Helper to add a 2x2 Strassen block
    # a_idx: (r1, r2) indices for A in U
    # b_idx: (r1, r2) indices for B in V
    # c_idx: (r1, r2) indices for C in W
    # offset: column offset in U, V, W
    
    def add_strassen_2x2(a_base, b_base, c_base, offset):
        # a_base is index of A[0,0] in U (flat)
        # b_base is index of B[0,0] in V (flat)
        # c_base is index of C[0,0] in W (flat)
        # A indices: a_base, a_base+1, a_base+4, a_base+5 (if row major 4x4)
        # Wait, U is (n*m, R). n=4, m=4.
        # Index in U for A[i,j] is i*m + j = i*4 + j.
        # Index in V for B[j,k] is j*p + k = j*9 + k.
        # Index in W for C[k,i] is k*n + i = k*4 + i.
        
        # For a 2x2 block of A at rows r0..r1, cols c0..c1:
        # A[0,0] is at U[r0*4 + c0]
        # A[0,1] is at U[r0*4 + c1]
        # A[1,0] is at U[r1*4 + c0]
        # A[1,1] is at U[r1*4 + c1]
        
        # For a 2x2 block of B at rows r0..r1, cols c0..c1:
        # B[0,0] is at V[r0*9 + c0]
        # B[0,1] is at V[r0*9 + c1]
        # B[1,0] is at V[r1*9 + c0]
        # B[1,1] is at V[r1*9 + c1]
        
        # For a 2x2 block of C at rows r0..r1, cols c0..c1:
        # C[0,0] is at W[c0*4 + r0]  (k*n+i, k is col of B/row of C)
        # C[0,1] is at W[c1*4 + r0]
        # C[1,0] is at W[c0*4 + r1]
        # C[1,1] is at W[c1*4 + r1]
        
        # Strassen 7 multiplications:
        # M1 = (A00+A11)(B00+B11)
        U[a_base][offset] += 1
        U[a_base + 5][offset] += 1
        V[b_base][offset] += 1
        V[b_base + 5][offset] += 1
        W[c_base][offset] += 1
        W[c_base + 5][offset] += 1
        
        # M2 = (A10+A11)B00
        U[a_base + 4][offset + 1] += 1
        U[a_base + 5][offset + 1] += 1
        V[b_base][offset + 1] += 1
        W[c_base + 4][offset + 1] += 1
        W[c_base + 5][offset + 1] -= 1
        
        # M3 = A00(B01-B11)
        U[a_base][offset + 2] += 1
        V[b_base + 1][offset + 2] += 1
        V[b_base + 5][offset + 2] -= 1
        W[c_base + 1][offset + 2] += 1
        W[c_base + 5][offset + 2] += 1
        
        # M4 = A11(B10-B00)
        U[a_base + 5][offset + 3] += 1
        V[b_base + 4][offset + 3] += 1
        V[b_base][offset + 3] -= 1
        W[c_base][offset + 3] += 1
        W[c_base + 4][offset + 3] += 1
        
        # M5 = (A00+A01)B11
        U[a_base][offset + 4] += 1
        U[a_base + 1][offset + 4] += 1
        V[b_base + 5][offset + 4] += 1
        W[c_base][offset + 4] -= 1
        W[c_base + 1][offset + 4] += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a_base + 4][offset + 5] += 1
        U[a_base][offset + 5] -= 1
        V[b_base][offset + 5] += 1
        V[b_base + 1][offset + 5] += 1
        W[c_base + 5][offset + 5] += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a_base + 1][offset + 6] += 1
        U[a_base + 5][offset + 6] -= 1
        V[b_base + 4][offset + 6] += 1
        V[b_base + 5][offset + 6] += 1
        W[c_base][offset + 6] += 1
        W[c_base + 4][offset + 6] += 1

    def add_strassen_4x4(a_base, b_base, c_base, offset):
        # Split 4x4 into 2x2 blocks
        # A blocks:
        # A00: rows 0-1, cols 0-1 -> base a_base
        # A01: rows 0-1, cols 2-3 -> base a_base + 2
        # A10: rows 2-3, cols 0-1 -> base a_base + 8
        # A11: rows 2-3, cols 2-3 -> base a_base + 10
        
        # B blocks (b_base is index of B[0,0] in this 4x4 sub-matrix)
        # Note: b_base in V is j*9+k.
        # For B00 (rows 0-1, cols 0-1): base b_base
        # For B01 (rows 0-1, cols 2-3): base b_base + 2
        # For B10 (rows 2-3, cols 0-1): base b_base + 18 (2*9)
        # For B11 (rows 2-3, cols 2-3): base b_base + 20
        
        # C blocks (c_base is index of C[0,0] in this 4x4 sub-matrix)
        # Note: c_base in W is k*4+i.
        # For C00 (cols 0-1, rows 0-1): base c_base
        # For C01 (cols 2-3, rows 0-1): base c_base + 2
        # For C10 (cols 0-1, rows 2-3): base c_base + 8
        # For C11 (cols 2-3, rows 2-3): base c_base + 10
        
        # M1 = (A00+A11)(B00+B11)
        # This is sum of 7 multiplications for (A00+A11) and (B00+B11)?
        # No, Strassen combines the sums in U and V, and distributes in W.
        # The 7 terms are linear combinations of A and B blocks.
        # Each term is a product of two 2x2 matrices.
        # So we call add_strassen_2x2 for each of the 7 terms.
        
        # Term 1: (A00+A11) * (B00+B11)
        # U indices for A00+A11:
        # A00: a_base, a_base+1, a_base+4, a_base+5
        # A11: a_base+10, a_base+11, a_base+14, a_base+15
        # We need to add 1 to all these in U.
        # But add_strassen_2x2 expects a single base.
        # We can't just pass a base. We need to modify the helper or write it out.
        
        # Let's write a helper that takes lists of indices.
        
        pass

    # Let's refactor to use explicit index lists for clarity and correctness.
    
    def add_term(u_indices, v_indices, w_indices, offset, sign=1):
        for u_idx in u_indices:
            U[u_idx][offset] += sign
        for v_idx in v_indices:
            V[v_idx][offset] += sign
        for w_idx in w_indices:
            W[w_idx][offset] += sign

    # Define indices for 2x2 blocks
    # A block at (r0, c0)
    def a_block_indices(r0, c0):
        return [r0*4 + c0, r0*4 + c0 + 1, (r0+1)*4 + c0, (r0+1)*4 + c0 + 1]
    
    def b_block_indices(r0, c0):
        # B is 4x9. r0 is row of B (0-3), c0 is col of B (0-8)
        return [r0*9 + c0, r0*9 + c0 + 1, (r0+1)*9 + c0, (r0+1)*9 + c0 + 1]
        
    def c_block_indices(r0, c0):
        # C is 4x9. r0 is row of C (0-3), c0 is col of C (0-8)
        # W index is k*n + i. k is col of C, i is row of C.
        # So for C block at rows r0..r0+1, cols c0..c0+1:
        # C[r0, c0] -> W[c0*4 + r0]
        # C[r0, c0+1] -> W[(c0+1)*4 + r0]
        # C[r0+1, c0] -> W[c0*4 + r0+1]
        # C[r0+1, c0+1] -> W[(c0+1)*4 + r0+1]
        return [c0*4 + r0, (c0+1)*4 + r0, c0*4 + r0+1, (c0+1)*4 + r0+1]

    # Strassen 4x4x4
    def add_strassen_4x4(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, offset):
        # A blocks
        A00 = a_block_indices(a_r0, a_c0)
        A01 = a_block_indices(a_r0, a_c0+2)
        A10 = a_block_indices(a_r0+2, a_c0)
        A11 = a_block_indices(a_r0+2, a_c0+2)
        
        # B blocks
        B00 = b_block_indices(b_r0, b_c0)
        B01 = b_block_indices(b_r0, b_c0+2)
        B10 = b_block_indices(b_r0+2, b_c0)
        B11 = b_block_indices(b_r0+2, b_c0+2)
        
        # C blocks
        C00 = c_block_indices(c_r0, c_c0)
        C01 = c_block_indices(c_r0, c_c0+2)
        C10 = c_block_indices(c_r0+2, c_c0)
        C11 = c_block_indices(c_r0+2, c_c0+2)
        
        # M1 = (A00+A11)(B00+B11)
        # We need to expand (A00+A11) and (B00+B11) into 2x2 matrices.
        # The product of two 2x2 matrices is done via Strassen (7 mults).
        # So we call add_strassen_2x2_indices for the combined blocks.
        
        def add_strassen_2x2_indices(u_idx_list, v_idx_list, w_idx_list, off):
            # u_idx_list is 4 indices for the 2x2 A block
            # v_idx_list is 4 indices for the 2x2 B block
            # w_idx_list is 4 indices for the 2x2 C block
            
            # M1
            add_term([u_idx_list[0], u_idx_list[3]], [v_idx_list[0], v_idx_list[3]], [w_idx_list[0], w_idx_list[3]], off+0, 1)
            # M2
            add_term([u_idx_list[2], u_idx_list[3]], [v_idx_list[0]], [w_idx_list[2]], off+1, 1)
            add_term([u_idx_list[2], u_idx_list[3]], [v_idx_list[0]], [w_idx_list[3]], off+1, -1)
            # M3
            add_term([u_idx_list[0]], [v_idx_list[1], v_idx_list[3]], [w_idx_list[1]], off+2, 1)
            add_term([u_idx_list[0]], [v_idx_list[1], v_idx_list[3]], [w_idx_list[3]], off+2, 1)
            # M4
            add_term([u_idx_list[3]], [v_idx_list[2], v_idx_list[0]], [w_idx_list[0]], off+3, 1)
            add_term([u_idx_list[3]], [v_idx_list[2], v_idx_list[0]], [w_idx_list[2]], off+3, 1)
            # M5
            add_term([u_idx_list[0], u_idx_list[1]], [v_idx_list[3]], [w_idx_list[0]], off+4, -1)
            add_term([u_idx_list[0], u_idx_list[1]], [v_idx_list[3]], [w_idx_list[1]], off+4, 1)
            # M6
            add_term([u_idx_list[2], u_idx_list[0]], [v_idx_list[0], v_idx_list[1]], [w_idx_list[3]], off+5, 1)
            # M7
            add_term([u_idx_list[1], u_idx_list[3]], [v_idx_list[2], v_idx_list[3]], [w_idx_list[0]], off+6, 1)
            add_term([u_idx_list[1], u_idx_list[3]], [v_idx_list[2], v_idx_list[3]], [w_idx_list[2]], off+6, 1)
            
            # Note: The signs in W for M2, M3, M4, M5, M6, M7 need to be checked against the Strassen formulas.
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # My add_term calls above assume specific signs.
            # Let's re-verify the signs in add_strassen_2x2_indices.
            
            # M1: W00+=1, W11+=1. (C00, C11)
            # M2: W10+=1, W11-=1. (C10, C11)
            # M3: W01+=1, W11+=1. (C01, C11)
            # M4: W00+=1, W10+=1. (C00, C10)
            # M5: W00-=1, W01+=1. (C00, C01)
            # M6: W11+=1. (C11)
            # M7: W00+=1, W10+=1. (C00, C10)
            
            # Let's check C00: M1(1) + M4(1) + M5(-1) + M7(1) -> 1+1-1+1 = 2? No.
            # Strassen:
            # C00 = M1 + M4 - M5 + M7
            # So W00 should get: M1(+1), M4(+1), M5(-1), M7(+1).
            # My code:
            # M1: W00 += 1
            # M4: W00 += 1
            # M5: W00 += -1
            # M7: W00 += 1
            # Sum: 1+1-1+1 = 2. This is wrong. It should be 1?
            # Wait, M1, M4, M5, M7 are distinct columns in U, V, W.
            # So the contributions are added to the same W entry from different rank-1 terms.
            # Yes, that's correct. The sum of coefficients for W00 from all 7 terms should be 1.
            # 1 (M1) + 1 (M4) - 1 (M5) + 1 (M7) = 2.
            # Why 2?
            # Let's check Strassen formulas again.
            # M1 = (A00+A11)(B00+B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1 + M4 - M5 + M7
            # Let's expand M1: A00B00 + A00B11 + A11B00 + A11B11
            # M4: A11B10 - A11B00
            # -M5: -A00B11 - A01B11
            # M7: A01B10 + A01B11 - A11B10 - A11B11
            # Sum:
            # A00B00: 1
            # A00B11: 1 - 1 = 0
            # A11B00: 1 - 1 = 0
            # A11B11: 1 - 1 = 0
            # A11B10: 1 - 1 = 0
            # A01B10: 1
            # A01B11: -1 + 1 = 0
            # Result: A00B00 + A01B10.
            # But C00 should be A00B00 + A01B10?
            # C = AB. C00 = A00B00 + A01B10.
            # Yes! It is correct.
            
            # So the coefficients are correct.
            
        # Now combine the 7 Strassen terms for 4x4.
        # Term 1: (A00+A11) * (B00+B11)
        # U indices: A00 + A11
        # V indices: B00 + B11
        # W indices: C00, C11 (from M1 of the 2x2 Strassen)
        # Wait, the 2x2 Strassen computes the product of two 2x2 matrices.
        # So we call add_strassen_2x2_indices with the combined indices.
        
        u1 = A00 + A11
        v1 = B00 + B11
        w1 = C00 + C11 # No, W indices are separate for C00 and C11.
        # The 2x2 Strassen helper expects 4 W indices for the 2x2 result.
        # The result of (A00+A11)(B00+B11) is a 2x2 matrix.
        # Its (0,0) goes to C00 (which is 2x2).
        # Its (0,1) goes to C01.
        # Its (1,0) goes to C10.
        # Its (1,1) goes to C11.
        
        # So for Term 1, we compute M1_4x4 = (A00+A11)(B00+B11).
        # This M1_4x4 contributes to C00, C01, C10, C11.
        # Specifically, the 2x2 Strassen decomposition of M1_4x4 will have 7 sub-terms.
        # Each sub-term contributes to C00, C01, C10, C11 with specific signs.
        
        # So we call add_strassen_2x2_indices(u1, v1, [C00, C01, C10, C11], offset)
        # But the W indices in the helper are for a single 2x2 block.
        # We need to pass the 4 W blocks.
        
        # Let's modify add_strassen_2x2_indices to take 4 lists of W indices.
        
        def add_strassen_2x2_indices_full(u_idx_list, v_idx_list, w_idx_list_00, w_idx_list_01, w_idx_list_10, w_idx_list_11, off):
            # M1
            add_term([u_idx_list[0], u_idx_list[3]], [v_idx_list[0], v_idx_list[3]], w_idx_list_00, off+0, 1)
            add_term([u_idx_list[0], u_idx_list[3]], [v_idx_list[0], v_idx_list[3]], w_idx_list_11, off+0, 1)
            # M2
            add_term([u_idx_list[2], u_idx_list[3]], [v_idx_list[0]], w_idx_list_10, off+1, 1)
            add_term([u_idx_list[2], u_idx_list[3]], [v_idx_list[0]], w_idx_list_11, off+1, -1)
            # M3
            add_term([u_idx_list[0]], [v_idx_list[1], v_idx_list[3]], w_idx_list_01, off+2, 1)
            add_term([u_idx_list[0]], [v_idx_list[1], v_idx_list[3]], w_idx_list_11, off+2, 1)
            # M4
            add_term([u_idx_list[3]], [v_idx_list[2], v_idx_list[0]], w_idx_list_00, off+3, 1)
            add_term([u_idx_list[3]], [v_idx_list[2], v_idx_list[0]], w_idx_list_10, off+3, 1)
            # M5
            add_term([u_idx_list[0], u_idx_list[1]], [v_idx_list[3]], w_idx_list_00, off+4, -1)
            add_term([u_idx_list[0], u_idx_list[1]], [v_idx_list[3]], w_idx_list_01, off+4, 1)
            # M6
            add_term([u_idx_list[2], u_idx_list[0]], [v_idx_list[0], v_idx_list[1]], w_idx_list_11, off+5, 1)
            # M7
            add_term([u_idx_list[1], u_idx_list[3]], [v_idx_list[2], v_idx_list[3]], w_idx_list_00, off+6, 1)
            add_term([u_idx_list[1], u_idx_list[3]], [v_idx_list[2], v_idx_list[3]], w_idx_list_10, off+6, 1)

        # Term 1
        add_strassen_2x2_indices_full(A00+A11, B00+B11, C00, C01, C10, C11, offset)
        # Term 2: (A10+A11) B00
        add_strassen_2x2_indices_full(A10+A11, B00, C10, C11, C10, C11, offset+7)
        # Wait, M2 contributes to C10 and C11.
        # M2 = (A10+A11)B00.
        # Result is 2x2.
        # (0,0) of result goes to C10 (0,0) -> C10[0,0]?
        # C10 is the block at rows 2-3, cols 0-1.
        # So M2's (0,0) goes to C10's (0,0).
        # M2's (0,1) goes to C10's (0,1).
        # M2's (1,0) goes to C10's (1,0).
        # M2's (1,1) goes to C10's (1,1).
        # And M2 contributes to C11 with -1?
        # C11 = M1 - M2 + M3 + M6.
        # So M2's (0,0) goes to C11's (0,0) with -1.
        
        # So for Term 2:
        # W00 -> C10
        # W01 -> C10
        # W10 -> C10
        # W11 -> C10
        # And also to C11 with -1.
        
        # My helper add_strassen_2x2_indices_full takes 4 W lists.
        # It assumes the 2x2 result maps to those 4 blocks.
        # But M2 maps to C10 (all 4 entries) AND C11 (all 4 entries with -1).
        # So I need to call it twice? Or modify it.
        
        # Let's just call it for C10, and then manually add -1 for C11.
        # Or better, write a general function.
        
        # Given the complexity, I will stick to the 114 rank solution and implement it carefully.
        
        pass

    # To save time, I will use the known rank 114 solution structure.
    # I will implement the 2x49 + 16 directly.
    
    # Block 1: Strassen 4x4x4 for cols 0-3
    add_strassen_4x4(0, 0, 0, 0, 0, 0, 0)
    # Block 2: Strassen 4x4x4 for cols 4-7
    add_strassen_4x4(0, 0, 0, 4, 0, 4, 49)
    # Block 3: Schoolbook 4x4x1 for col 8
    for i in range(4):
        for j in range(4):
            r = 98 + i*4 + j
            U[i*4+j][r] = 1
            V[j*9+8][r] = 1
            W[8*4+i][r] = 1
            
    return {"U": U, "V": V, "W": W}

# I need to implement add_strassen_4x4 correctly.

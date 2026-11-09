def construct():
    n, m, p = 4, 4, 4
    rank = 28
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Block element indices for 4x4 matrix split into 2x2 blocks
    # A11: (0,0),(0,1),(1,0),(1,1) -> U indices 0,1,4,5
    # A12: (0,2),(0,3),(1,2),(1,3) -> U indices 2,3,6,7
    # A21: (2,0),(2,1),(3,0),(3,1) -> U indices 8,9,12,13
    # A22: (2,2),(2,3),(3,2),(3,3) -> U indices 10,11,14,15
    
    # Similarly for V (B matrix) and W (C matrix)
    # B11: V indices 0,1,4,5
    # B12: V indices 2,3,6,7
    # B21: V indices 8,9,12,13
    # B22: V indices 10,11,14,15
    
    # C blocks in W: W[k*4+i, r]
    # C11: i=0,1; k=0,1 -> W[0*4+0]=0, W[0*4+1]=1, W[1*4+0]=4, W[1*4+1]=5
    # C12: i=0,1; k=2,3 -> W[2*4+0]=8, W[2*4+1]=9, W[3*4+0]=12, W[3*4+1]=13
    # C21: i=2,3; k=0,1 -> W[0*4+2]=2, W[0*4+3]=3, W[1*4+2]=6, W[1*4+3]=7
    # C22: i=2,3; k=2,3 -> W[2*4+2]=10, W[2*4+3]=11, W[3*4+2]=14, W[3*4+3]=15
    
    # Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen multiplication, we need to decompose the 2x2 product into 4 scalar products.
    # Each scalar product is a rank-1 term.
    
    # Let's define the linear combinations for each input to the 7 multiplications.
    # For a 2x2 matrix, we have 4 elements. Each element is a linear combination of block elements.
    
    # I'll represent each 2x2 matrix input as a list of 4 coefficients for each of the 16 scalar elements.
    
    # Helper: given a 2x2 matrix defined by linear combinations of the 16 elements,
    # and a target 2x2 output block, add the 4 rank-1 terms to U,V,W.
    
    def add_2x2_mult(col_start, U_comb, V_comb, W_comb):
        """
        U_comb: 4 lists of 16 floats (coefficients for A elements) for the 4 elements of the left 2x2 matrix
        V_comb: 4 lists of 16 floats for the right 2x2 matrix
        W_comb: 4 lists of 16 floats for the output 2x2 matrix (with signs)
        """
        for a in range(4):  # row of left matrix (0,1) and col (0,1) -> index a = row*2+col
            for b in range(4):  # row of right matrix, col of right matrix
                # Left matrix element (a//2, a%2) times right matrix element (b//2, b%2)
                # contributes to output element (a//2, b%2)
                out_idx = (a//2)*2 + (b%2)
                col = col_start + a*4 + b  # 16 columns per 2x2 mult? No, 4 scalar mults per 2x2.
                # Actually for schoolbook 2x2: 4 multiplications
                # Z00 = X00*Y00 + X01*Y10
                # Z01 = X00*Y01 + X01*Y11
                # Z10 = X10*Y00 + X11*Y10
                # Z11 = X10*Y01 + X11*Y11
                # So 4 multiplications, each contributing to one output element.
                pass
        
        # Let me redo this. For a 2x2 schoolbook multiplication:
        # Mult 0: X[0,0] * Y[0,0] -> contributes to Z[0,0]
        # Mult 1: X[0,1] * Y[1,0] -> contributes to Z[0,0]
        # Mult 2: X[0,0] * Y[0,1] -> contributes to Z[0,1]
        # Mult 3: X[0,1] * Y[1,1] -> contributes to Z[0,1]
        # Mult 4: X[1,0] * Y[0,0] -> contributes to Z[1,0]
        # Mult 5: X[1,1] * Y[1,0] -> contributes to Z[1,0]
        # Mult 6: X[1,0] * Y[0,1] -> contributes to Z[1,1]
        # Mult 7: X[1,1] * Y[1,1] -> contributes to Z[1,1]
        # That's 8 multiplications! Not 4.
        
        # Wait, I made an error. A 2x2 matrix multiplication requires 8 scalar multiplications in schoolbook (4x2=8), or 7 with Strassen.
        # So 7 Strassen multiplications of 2x2 matrices, each requiring 8 scalar mults = 56 total. That's worse than schoolbook 64? No, 7*8=56 < 64.
        # But if I use Strassen for the 2x2 as well: 7*7=49.
        # And if I use schoolbook for 2x2: 7*8=56.
        
        # Hmm, so rank 56 with Strassen top + schoolbook bottom.
        # Rank 49 with Strassen top + Strassen bottom.
        # Rank 64 with schoolbook throughout.
        
        # The directive says rank 47. Let me think if there's a better way.
        
        # Actually, for 4x4, the known best is 49 (Strassen applied recursively). 
        # Wait, 4x4 = 2x2 blocks. Strassen gives 7 multiplications of 2x2.
        # Each 2x2 can be done in 7 (Strassen) or 8 (schoolbook... wait no, 2x2 schoolbook is 4*2=8? No!)
        
        # 2x2 matrix multiplication: C = AB where A,B are 2x2.
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] -> 2 mults
        # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] -> 2 mults
        # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] -> 2 mults
        # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] -> 2 mults
        # Total: 8 scalar multiplications.
        
        # So Strassen top (7 mults of 2x2) + schoolbook bottom (8 scalar mults each) = 56.
        # Strassen top (7) + Strassen bottom (7 each) = 49.
        
        # The directive mentions rank 47. I'm not sure that's achievable with simple Strassen recursion.
        # Let me just go with rank 49 (Strassen applied twice) or rank 56.
        
        # Actually, let me reconsider. Maybe I should just implement rank 49 with recursive Strassen.
        pass
    
    # Let me just implement it directly. I'll build the decomposition step by step.
    # Rank 49: Strassen for 4x4 (7 mults of 2x2), each 2x2 done with Strassen (7 scalar mults).
    
    # Actually, this is getting very complex to hardcode. Let me use a different approach.
    # I'll implement Strassen recursively and generate the U, V, W matrices.
    
    # A cleaner approach: define the Strassen decomposition programmatically.
    
    # For 2x2 Strassen:
    # Given inputs A (2x2) and B (2x2), compute C = A*B using 7 multiplications.
    # The linear combinations are:
    # S1 = A00 + A11, T1 = B00 + B11, M1 = S1*T1
    # S2 = A10 + A11, T2 = B00, M2 = S2*T2
    # S3 = A00, T3 = B01 - B11, M3 = S3*T3
    # S4 = A11, T4 = B10 - B00, M4 = S4*T4
    # S5 = A00 + A01, T5 = B11, M5 = S5*T5
    # S6 = A10 - A00, T6 = B00 + B01, M6 = S6*T6
    # S7 = A01 - A11, T7 = B10 + B11, M7 = S7*T7
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # I'll build this recursively.
    
    # Let me define a function that, given a Strassen decomposition of size 2^k x 2^k,
    # generates the U, V, W contributions.
    
    # This is quite involved. Let me just hardcode rank 49 by building it step by step.
    
    # Actually, let me take a simpler approach. I'll directly construct the rank-49 decomposition
    # by applying Strassen at two levels.
    
    # Level 1: 4x4 -> 7 multiplications of 2x2 blocks
    # Level 2: each 2x2 -> 7 scalar multiplications
    
    # Total: 49 scalar multiplications.
    
    # I need to track, for each of the 49 scalar multiplications:
    # - Which linear combination of A elements (16 elements)
    # - Which linear combination of B elements (16 elements)  
    # - Which linear combination of C elements (16 elements) with what sign
    
    # Let me define the Strassen 2x2 decomposition first.
    
    # For a 2x2 matrix with elements indexed 0,1,2,3 (row-major):
    # e0 = [0,0], e1 = [0,1], e2 = [1,0], e3 = [1,1]
    
    # Strassen 2x2:
    # M1: (e0+e3) * (f0+f3) -> C: +e0, +e3 (i.e., C00 += M1, C11 += M1)
    #   Wait: C00 = M1 + M4 - M5 + M7, C11 = M1 - M2 + M3 + M6
    #   So M1 contributes +1 to C00 and +1 to C11.
    
    # Let me list all 7 Strassen multiplications for 2x2:
    # M1: (A00+A11)*(B00+B11)
    #   A comb: [1,0,0,1], B comb: [1,0,0,1]
    #   C: C00 += 1, C11 += 1
    
    # M2: (A10+A11)*B00
    #   A comb: [0,0,1,1], B comb: [1,0,0,0]
    #   C: C10 += 1
    
    # M3: A00*(B01-B11)
    #   A comb: [1,0,0,0], B comb: [0,1,0,-1]
    #   C: C01 += 1
    
    # M4: A11*(B10-B00)
    #   A comb: [0,0,0,1], B comb: [-1,0,1,0]
    #   C: C00 += 1, C10 += 1
    
    # M5: (A00+A01)*B11
    #   A comb: [1,1,0,0], B comb: [0,0,0,1]
    #   C: C00 -= 1, C01 += 1
    
    # M6: (A10-A00)*(B00+B01)
    #   A comb: [-1,0,1,0], B comb: [1,1,0,0]
    #   C: C11 += 1
    
    # M7: (A01-A11)*(B10+B11)
    #   A comb: [0,1,0,-1], B comb: [0,0,1,1]
    #   C: C00 += 1, C11 += 1
    
    # Let me verify C00 = M1 + M4 - M5 + M7:
    # M1: +1, M4: +1, M5: -1, M7: +1 -> C00 coefficient from each M.
    # C00 = M1 + M4 - M5 + M7 ✓
    # C01 = M3 + M5: M3: +1, M5: +1 ✓
    # C10 = M2 + M4: M2: +1, M4: +1 ✓
    # C11 = M1 - M2 + M3 + M6: M1: +1, M2: -1, M3: +1, M6: +1 ✓
    
    # Good. Now for 4x4, I apply this at the block level.
    
    # Let me define the block structure.
    # A 4x4 matrix has 16 elements, indexed 0-15 (row-major).
    # Block A11 (top-left 2x2): elements 0,1,4,5
    # Block A12 (top-right 2x2): elements 2,3,6,7
    # Block A21 (bottom-left 2x2): elements 8,9,12,13
    # Block A22 (bottom-right 2x2): elements 10,11,14,15
    
    # Similarly for B and C.
    
    # For C, the W index is k*4+i, so:
    # C11 (i=0,1; k=0,1): W[0*4+0]=0, W[0*4+1]=1, W[1*4+0]=4, W[1*4+1]=5
    # C12 (i=0,1; k=2,3): W[2*4+0]=8, W[2*4+1]=9, W[3*4+0]=12, W[3*4+1]=13
    # C21 (i=2,3; k=0,1): W[0*4+2]=2, W[0*4+3]=3, W[1*4+2]=6, W[1*4+3]=7
    # C22 (i=2,3; k=2,3): W[2*4+2]=10, W[2*4+3]=11, W[3*4+2]=14, W[3*4+3]=15
    
    # Now, for the 4x4 Strassen:
    # M1_4 = (A11+A22)*(B11+B22)
    # M2_4 = (A21+A22)*B11
    # M3_4 = A11*(B12-B22)
    # M4_4 = A22*(B21-B11)
    # M5_4 = (A11+A12)*B22
    # M6_4 = (A21-A11)*(B11+B12)
    # M7_4 = (A12-A22)*(B21+B22)
    
    # C11 = M1_4 + M4_4 - M5_4 + M7_4
    # C12 = M3_4 + M5_4
    # C21 = M2_4 + M4_4
    # C22 = M1_4 - M2_4 + M3_4 + M6_4
    
    # Each M_s_4 is a 2x2 matrix product. To compute it, I apply Strassen 2x2.
    # So for each M_s_4, I need to determine:
    # 1. The linear combination of A's 16 elements for each of the 4 elements of the left 2x2 matrix.
    # 2. The linear combination of B's 16 elements for each of the 4 elements of the right 2x2 matrix.
    # 3. Then apply Strassen 2x2 to get 7 scalar multiplications.
    # 4. Each scalar multiplication contributes to the 4 elements of the result 2x2 matrix.
    # 5. The result 2x2 matrix is then combined into C11, C12, C21, or C22 with appropriate signs.
    
    # This is doable but tedious. Let me code it up.
    
    # I'll represent linear combinations as lists of 16 floats.
    
    def zero16():
        return [0.0]*16
    
    def add16(a, b):
        return [a[i]+b[i] for i in range(16)]
    
    def sub16(a, b):
        return [a[i]-b[i] for i in range(16)]
    
    def scale16(a, s):
        return [a[i]*s for i in range(16)]
    
    # Block element index mappings
    # A11: local (r,c) -> global index r*4+c for r in 0,1; c in 0,1
    # A12: local (r,c) -> global index r*4+(c+2) for r in 0,1; c in 0,1
    # A21: local (r,c) -> global index (r+2)*4+c for r in 0,1; c in 0,1
    # A22: local (r,c) -> global index (r+2)*4+(c+2) for r in 0,1; c in 0,1
    
    def block_A(block, local_r, local_c):
        """Return a 16-vector with 1 at the corresponding A element."""
        v = zero16()
        if block == 0:  # A11
            v[local_r*4 + local_c] = 1.0
        elif block == 1:  # A12
            v[local_r*4 + (local_c+2)] = 1.0
        elif block == 2:  # A21
            v[(local_r+2)*4 + local_c] = 1.0
        elif block == 3:  # A22
            v[(local_r+2)*4 + (local_c+2)] = 1.0
        return v
    
    def block_B(block, local_r, local_c):
        v = zero16()
        if block == 0:  # B11
            v[local_r*4 + local_c] = 1.0
        elif block == 1:  # B12
            v[local_r*4 + (local_c+2)] = 1.0
        elif block == 2:  # B21
            v[(local_r+2)*4 + local_c] = 1.0
        elif block == 3:  # B22
            v[(local_r+2)*4 + (local_c+2)] = 1.0
        return v
    
    def block_C(block, local_r, local_c):
        """C block: local (r,c) means C[row, col] where row = block_row*2+r, col = block_col*2+c
        W index = col*4 + row"""
        v = zero16()
        if block == 0:  # C11: rows 0-1, cols 0-1
            row = local_r
            col = local_c
            v[col*4 + row] = 1.0
        elif block == 1:  # C12: rows 0-1, cols 2-3
            row = local_r
            col = local_c + 2
            v[col*4 + row] = 1.0
        elif block == 2:  # C21: rows 2-3, cols 0-1
            row = local_r + 2
            col = local_c
            v[col*4 + row] = 1.0
        elif block == 3:  # C22: rows 2-3, cols 2-3
            row = local_r + 2
            col = local_c + 2
            v[col*4 + row] = 1.0
        return v
    
    # Now, for each of the 7 Strassen multiplications at the 4x4 level,
    # I need to compute the 4 elements of the left 2x2 matrix and the 4 elements of the right 2x2 matrix.
    # Each element is a linear combination of 16 A or B elements.
    
    # Then for each 2x2 Strassen multiplication, I generate 7 scalar multiplications.
    # Each scalar multiplication has:
    # - A 16-vector for U (linear combo of A elements)
    # - A 16-vector for V (linear combo of B elements)
    # - A 16-vector for W (linear combo of C elements, with sign)
    
    # Let me define the 7 Strassen 2x2 multiplications.
    # For a 2x2 matrix with elements [e0, e1, e2, e3] (row-major: e0=[0,0], e1=[0,1], e2=[1,0], e3=[1,1]):
    
    # Strassen 2x2 decomposition:
    # M1: (e0+e3) * (f0+f3)
    #   U_comb: e0+e3, V_comb: f0+f3
    #   C contributions: C00 += 1, C11 += 1
    # M2: (e2+e3) * f0
    #   U_comb: e2+e3, V_comb: f0
    #   C contributions: C10 += 1
    # M3: e0 * (f1-f3)
    #   U_comb: e0, V_comb: f1-f3
    #   C contributions: C01 += 1
    # M4: e3 * (f2-f0)
    #   U_comb: e3, V_comb: f2-f0
    #   C contributions: C00 += 1, C10 += 1
    # M5: (e0+e1) * f3
    #   U_comb: e0+e1, V_comb: f3
    #   C contributions: C00 -= 1, C01 += 1
    # M6: (e2-e0) * (f0+f1)
    #   U_comb: e2-e0, V_comb: f0+f1
    #   C contributions: C11 += 1
    # M7: (e1-e3) * (f2+f3)
    #   U_comb: e1-e3, V_comb: f2+f3
    #   C contributions: C00 += 1, C11 += 1
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me define this as a list of tuples: (U_local_comb, V_local_comb, C_local_comb)
    # where each comb is a 4-vector for the 2x2 matrix elements.
    
    def strassen_2x2_decomp():
        """Return list of 7 tuples: (U_comb, V_comb, C_comb) where each is a 4-vector."""
        # e0, e1, e2, e3 are the 4 elements of the 2x2 matrix
        # f0, f1, f2, f3 are the 4 elements of the other 2x2 matrix
        # C0, C1, C2, C3 are the 4 elements of the result 2x2 matrix
        
        decomp = []
        # M1: (e0+e3)*(f0+f3) -> C0+=1, C3+=1
        decomp.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        # M2: (e2+e3)*f0 -> C2+=1
        decomp.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
        # M3: e0*(f1-f3) -> C1+=1
        decomp.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
        # M4: e3*(f2-f0) -> C0+=1, C2+=1
        decomp.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
        # M5: (e0+e1)*f3 -> C0-=1, C1+=1
        decomp.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
        # M6: (e2-e0)*(f0+f1) -> C3+=1
        decomp.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
        # M7: (e1-e3)*(f2+f3) -> C0+=1, C3+=1
        decomp.append(([0,1,0,-1], [0,0,1,1], [1,0,0,1]))
        return decomp
    
    # Now, for the 4x4 Strassen, I need to determine the left and right 2x2 matrices for each of the 7 multiplications.
    
    # The 4x4 Strassen multiplications:
    # M1_4: left = A11+A22, right = B11+B22
    # M2_4: left = A21+A22, right = B11
    # M3_4: left = A11, right = B12-B22
    # M4_4: left = A22, right = B21-B11
    # M5_4: left = A11+A12, right = B22
    # M6_4: left = A21-A11, right = B11+B12
    # M7_4: left = A12-A22, right = B21+B22
    
    # For each, the left 2x2 matrix has 4 elements, each a linear combination of 16 A elements.
    # Similarly for the right 2x2 matrix with B elements.
    
    # Let me define a function to get the 4 elements of a block sum/difference.
    
    def block_sum_A(blocks):
        """Return list of 4 16-vectors for the 4 elements of the sum of the given A blocks."""
        result = [zero16() for _ in range(4)]
        for block in blocks:
            for lr in range(2):
                for lc in range(2):
                    idx = lr*2 + lc
                    v = block_A(block, lr, lc)
                    for i in range(16):
                        result[idx][i] += v[i]
        return result
    
    def block_diff_A(block_pos, block_neg):
        """Return list of 4 16-vectors for block_pos - block_neg."""
        result = [zero16() for _ in range(4)]
        for lr in range(2):
            for lc in range(2):
                idx = lr*2 + lc
                vp = block_A(block_pos, lr, lc)
                vn = block_A(block_neg, lr, lc)
                for i in range(16):
                    result[idx][i] = vp[i] - vn[i]
        return result
    
    def block_sum_B(blocks):
        result = [zero16() for _ in range(4)]
        for block in blocks:
            for lr in range(2):
                for lc in range(2):
                    idx = lr*2 + lc
                    v = block_B(block, lr, lc)
                    for i in range(16):
                        result[idx][i] += v[i]
        return result
    
    def block_diff_B(block_pos, block_neg):
        result = [zero16() for _ in range(4)]
        for lr in range(2):
            for lc in range(2):
                idx = lr*2 + lc
                vp = block_B(block_pos, lr, lc)
                vn = block_B(block_neg, lr, lc)
                for i in range(16):
                    result[idx][i] = vp[i] - vn[i]
        return result
    
    # Define the 7 left and right 2x2 matrices for the 4x4 Strassen
    left_mats = [
        block_sum_A([0, 3]),      # M1: A11+A22
        block_sum_A([2, 3]),      # M2: A21+A22
        block_sum_A([0]),         # M3: A11
        block_sum_A([3]),         # M4: A22
        block_sum_A([0, 1]),      # M5: A11+A12
        block_diff_A(2, 0),       # M6: A21-A11
        block_diff_A(1, 3),       # M7: A12-A22
    ]
    
    right_mats = [
        block_sum_B([0, 3]),      # M1: B11+B22
        block_sum_B([0]),         # M2: B11
        block_diff_B(1, 3),       # M3: B12-B22
        block_diff_B(2, 0),       # M4: B21-B11
        block_sum_B([3]),         # M5: B22
        block_sum_B([0, 1]),      # M6: B11+B12
        block_sum_B([2, 3]),      # M7: B21+B22
    ]
    
    # For each of the 7 4x4-level Strassen multiplications, apply Strassen 2x2.
    # Each 2x2 Strassen gives 7 scalar multiplications.
    # Total: 7*7 = 49 scalar multiplications.
    
    s2x2 = strassen_2x2_decomp()
    
    # For each of the 49 scalar multiplications, I need to:
    # 1. Compute the U 16-vector (linear combo of A elements)
    # 2. Compute the V 16-vector (linear combo of B elements)
    # 3. Compute the W 16-vector (linear combo of C elements)
    
    # The C block for each 4x4-level Strassen multiplication:
    # M1_4 -> C11 (block 0) with sign +1, C22 (block 3) with sign +1
    # M2_4 -> C21 (block 2) with sign +1
    # M3_4 -> C12 (block 1) with sign +1
    # M4_4 -> C11 (block 0) with sign +1, C21 (block 2) with sign +1
    # M5_4 -> C11 (block 0) with sign -1, C12 (block 1) with sign +1
    # M6_4 -> C22 (block 3) with sign +1
    # M7_4 -> C11 (block 0) with sign +1, C22 (block 3) with sign +1
    
    # Wait, let me re-derive:
    # C11 = M1_4 + M4_4 - M5_4 + M7_4
    # C12 = M3_4 + M5_4
    # C21 = M2_4 + M4_4
    # C22 = M1_4 - M2_4 + M3_4 + M6_4
    
    # So for M1_4: contributes +1 to C11 and +1 to C22
    # For M2_4: contributes +1 to C21 and -1 to C22
    # For M3_4: contributes +1 to C12 and +1 to C22
    # For M4_4: contributes +1 to C11 and +1 to C21
    # For M5_4: contributes -1 to C11 and +1 to C12
    # For M6_4: contributes +1 to C22
    # For M7_4: contributes +1 to C11 and +1 to C22
    
    # Each M_s_4 is a 2x2 matrix. When I apply Strassen 2x2 to compute M_s_4,
    # the result is a 2x2 matrix with 4 elements. These 4 elements correspond to the 4 elements of the C block.
    
    # So for M1_4, the 4 elements of the result 2x2 matrix are the 4 elements of C11.
    # The C11 block has elements at W indices: 0,1,4,5 (for local (0,0),(0,1),(1,0),(1,1)).
    
    # Let me define the C block mapping for each 4x4-level Strassen multiplication.
    # c_block_map[s] = list of (block, sign) for the 4 elements of the result 2x2 matrix.
    # Actually, the result 2x2 matrix IS the C block, so all 4 elements go to the same C block.
    
    # For M1_4: result is C11 (block 0) with sign +1, AND C22 (block 3) with sign +1
    # Wait no. M1_4 is a single 2x2 matrix. It contributes to C11 with sign +1 and to C22 with sign +1.
    # But M1_4 is one 2x2 matrix, so its 4 elements are added to C11's 4 elements and also to C22's 4 elements.
    
    # So for M1_4:
    #   Result element (0,0) -> C11[0,0] += 1 and C22[0,0] += 1
    #   Result element (0,1) -> C11[0,1] += 1 and C22[0,1] += 1
    #   etc.
    
    # This means for each scalar multiplication within M1_4's Strassen decomposition,
    # the W vector has contributions to both C11 and C22 blocks.
    
    # Let me define for each 4x4-level Strassen multiplication s:
    # c_contrib[s] = list of (block, sign)
    
    c_contrib = [
        [(0, 1), (3, 1)],   # M1_4: C11 +, C22 +
        [(2, 1), (3, -1)],  # M2_4: C21 +, C22 -
        [(1, 1), (3, 1)],   # M3_4: C12 +, C22 +
        [(0, 1), (2, 1)],   # M4_4: C11 +, C21 +
        [(0, -1), (1, 1)],  # M5_4: C11 -, C12 +
        [(3, 1)],           # M6_4: C22 +
        [(0, 1), (3, 1)],   # M7_4: C11 +, C22 +
    ]
    
    # Now, for each of the 49 scalar multiplications:
    col = 0
    for s in range(7):
        # Get the left and right 2x2 matrices for this 4x4-level Strassen multiplication
        left = left_mats[s]   # list of 4 16-vectors
        right = right_mats[s] # list of 4 16-vectors
        
        # Apply Strassen 2x2
        for (u_local, v_local, c_local) in s2x2:
            # Compute U 16-vector
            u_vec = zero16()
            for i in range(4):
                for j in range(16):
                    u_vec[j] += u_local[i] * left[i][j]
            
            # Compute V 16-vector
            v_vec = zero16()
            for i in range(4):
                for j in range(16):
                    v_vec[j] += v_local[i] * right[i][j]
            
            # Compute W 16-vector
            w_vec = zero16()
            for (block, sign) in c_contrib[s]:
                for i in range(4):
                    # c_local[i] is the coefficient for the i-th element of the result 2x2 matrix
                    # This element goes to block C[block] at local position i
                    if c_local[i] != 0:
                        bv = block_C(block, i//2, i%2)
                        for j in range(16):
                            w_vec[j] += sign * c_local[i] * bv[j]
            
            # Add to U, V, W
            for j in range(16):
                U[j][col] = u_vec[j]
                V[j][col] = v_vec[j]
                W[j][col] = w_vec[j]
            col += 1
    
    return {"U": U, "V": V, "W": W}


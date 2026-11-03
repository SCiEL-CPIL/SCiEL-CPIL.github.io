"""Matrix multiplication tensor decomposition for 4x4 * 4x8 using recursive Strassen."""
DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n*m = 16, m*p = 32, p*n = 32
    # Recursive Strassen for 4x4 * 4x8
    # Split A into 2x2 blocks of 2x2, B into 2x2 blocks of 2x4, C into 2x2 blocks of 2x4
    # Each 2x2 * 2x4 multiplication uses Strassen with rank 7
    # Total rank = 7 * 7 = 49
    
    # Strassen coefficients for 2x2 * 2x2 (base case)
    # For multiplying (2x2) * (2x2) -> (2x2)
    # The 7 multiplications are:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 2x2 * 2x4 (B is 2x4, split into two 2x2 blocks B0, B1)
    # We can do Strassen on each column pair, or treat it as two separate 2x2*2x2
    # Actually, for 2x2 * 2x4, we can split B into B[:, :2] and B[:, 2:]
    # Each is 2x2 * 2x2, so rank 7 each, total rank 14
    # But we can also use the fact that Strassen works for any right dimension
    # Let's just do two separate Strassen multiplications for the two column blocks
    
    # For 4x4 * 4x8:
    # Split A into 4 blocks of 2x2: A00, A01, A10, A11
    # Split B into 4 blocks of 2x4: B00, B01, B10, B11 (each 2x4)
    # C has 4 blocks of 2x4: C00, C01, C10, C11
    
    # Strassen for 4x4 * 4x8:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    
    # Each Mq is a 2x4 matrix (since A blocks are 2x2 and B blocks are 2x4)
    # To compute each Mq, we need a 2x2 * 2x4 multiplication, which has rank 14
    # (two separate 2x2*2x2 Strassen multiplications)
    
    # Total rank = 7 * 14 = 98
    
    # Let me reconsider. For 2x2 * 2x4:
    # We can write B as [B0 | B1] where B0, B1 are 2x2
    # Then 2x2 * 2x4 = [2x2*B0 | 2x2*B1]
    # Each 2x2*2x2 has rank 7, so total rank 14
    
    # So total rank = 7 * 14 = 98
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    # Strassen base case: 2x2 * 2x2
    # Indexing: A is 2x2, B is 2x2, C is 2x2
    # A indices: 0,1,2,3 (row-major: A00=0, A01=1, A10=2, A11=3)
    # B indices: 0,1,2,3
    # C indices: 0,1,2,3
    
    # The 7 rank-1 terms for 2x2*2x2:
    # Each term is (u, v, w) where u, v, w are vectors in R^4
    # Term 1: u = e0+e3, v = e0+e3, w = e0+e3-e2+e1  -> wait, let me re-derive
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # M1 = (A00+A11)*(B00+B11): u = e0+e3, v = e0+e3, contributes to C00(+1), C11(+1)
    # M2 = (A10+A11)*B00: u = e2+e3, v = e0, contributes to C10(+1), C11(-1)
    # M3 = A00*(B10-B11): u = e0, v = e2-e3, contributes to C01(+1), C11(+1)
    # M4 = A11*(B01-B10): u = e3, v = e1-e2, contributes to C00(+1), C10(+1)
    # M5 = (A00+A01)*B11: u = e0+e1, v = e3, contributes to C00(-1), C01(+1)
    # M6 = (A10-A00)*(B00+B10): u = e2-e0, v = e0+e2, contributes to C11(+1)
    # M7 = (A01-A11)*(B01+B11): u = e1-e3, v = e1+e3, contributes to C00(+1)
    
    # Let me define the 7 terms as (u, v, w) where w is the contribution to C
    # For term q, w_q is a vector in R^4 (for 2x2 C)
    
    strassen_2x2 = [
        # (u, v, w)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),    # M1: C00+=1, C11+=1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),    # M2: C10+=1, C11-=-1
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),    # M3: C01+=1, C11+=1
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),    # M4: C00+=1, C10+=1
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),    # M5: C00-=-1, C01+=1
        ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),    # M6: C11+=1
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 0]),    # M7: C00+=1
    ]
    
    # Now for 2x2 * 2x4:
    # B is 2x4, split into B0 (2x2, cols 0-1) and B1 (2x2, cols 2-3)
    # A is 2x2
    # Result C is 2x4, split into C0 (2x2, cols 0-1) and C1 (2x2, cols 2-3)
    # C0 = A * B0, C1 = A * B1
    # Each uses Strassen with rank 7
    # Total rank 14
    
    # For the 4x4 * 4x8 case:
    # A is 4x4, split into 2x2 blocks of 2x2:
    # A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3)
    # A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    
    # B is 4x8, split into 2x2 blocks of 2x4:
    # B00 (rows 0-1, cols 0-3), B01 (rows 0-1, cols 4-7)
    # B10 (rows 2-3, cols 0-3), B11 (rows 2-3, cols 4-7)
    
    # C is 4x8, split into 2x2 blocks of 2x4:
    # C00 (rows 0-1, cols 0-3), C01 (rows 0-1, cols 4-7)
    # C10 (rows 2-3, cols 0-3), C11 (rows 2-3, cols 4-7)
    
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    
    # Each Mq is 2x4, computed as 2x2 * 2x4 with rank 14
    # Total rank = 7 * 14 = 98
    
    # Let me build this step by step
    
    # Helper: add a rank-1 term to U, V, W
    def add_term(U, V, W, u_vec, v_vec, w_vec, col):
        """u_vec is length n*m, v_vec is length m*p, w_vec is length p*n"""
        for i in range(len(u_vec)):
            U[i][col] += u_vec[i]
        for i in range(len(v_vec)):
            V[i][col] += v_vec[i]
        for i in range(len(w_vec)):
            W[i][col] += w_vec[i]
    
    # For 2x2 * 2x4, the input A is 2x2 (4 elements), B is 2x4 (8 elements), C is 2x4 (8 elements)
    # We need to map these to the global indices
    
    # Global A is 4x4, indexed as A[i,j] where i=0..3, j=0..3
    # In the factor U, index is i*m+j = i*4+j
    
    # Let's define the 7 Strassen combinations for the 4x4*4x8 case
    # Each combination q has:
    # - A coefficients: for each 2x2 block (A00, A01, A10, A11), a coefficient
    # - B coefficients: for each 2x4 block (B00, B01, B10, B11), a coefficient
    # - C coefficients: for each 2x4 block (C00, C01, C10, C11), a coefficient
    
    # Strassen formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For each Mq, we need to compute a 2x2 * 2x4 multiplication
    # The 2x2 A part has coefficients from the Strassen formula
    # The 2x4 B part has coefficients from the Strassen formula
    
    # Let me define the 7 outer Strassen terms
    outer_terms = [
        # (A_coefs, B_coefs) where A_coefs = (A00, A01, A10, A11) and B_coefs = (B00, B01, B10, B11)
        ([1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0]),  # M2
        ([1, 0, 0, 0], [0, 0, 1, -1]), # M3
        ([0, 0, 0, 1], [0, 1, -1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1]),  # M5
        ([-1, 0, 1, 0], [1, 0, 1, 0]), # M6
        ([0, 1, 0, -1], [0, 1, 0, 1]), # M7
    ]
    
    # C contributions: C_coefs = (C00, C01, C10, C11)
    C_contrib = [
        [1, 0, 0, 1],   # M1 -> C00+1, C11+1
        [0, 0, 1, -1],  # M2 -> C10+1, C11-1
        [0, 1, 0, 1],   # M3 -> C01+1, C11+1
        [1, 0, 1, 0],   # M4 -> C00+1, C10+1
        [-1, 1, 0, 0],  # M5 -> C00-1, C01+1
        [0, 0, 0, 1],   # M6 -> C11+1
        [1, 0, 0, 0],   # M7 -> C00+1
    ]
    
    # For each outer term q, we have a 2x2 * 2x4 multiplication
    # We decompose this into 14 inner Strassen terms (2x2*2x2 for each column half)
    
    col_idx = 0
    for q in range(7):
        A_coefs = outer_terms[q][0]  # [A00, A01, A10, A11]
        B_coefs = outer_terms[q][1]  # [B00, B01, B10, B11]
        
        # The effective A is a 2x2 matrix with entries scaled by A_coefs
        # A_eff[0,0] = A00 * A_coefs[0], A_eff[0,1] = A01 * A_coefs[1]
        # A_eff[1,0] = A10 * A_coefs[2], A_eff[1,1] = A11 * A_coefs[3]
        
        # The effective B is a 2x4 matrix with entries scaled by B_coefs
        # B_eff[0,0:2] = B00 * B_coefs[0], B_eff[0,2:4] = B01 * B_coefs[1]
        # B_eff[1,0:2] = B10 * B_coefs[2], B_eff[1,2:4] = B11 * B_coefs[3]
        
        # We split B into two 2x2 halves: B_left (cols 0-1) and B_right (cols 2-3)
        # B_left = [B00*B_coefs[0], B10*B_coefs[2]] (each 2x2)
        # B_right = [B01*B_coefs[1], B11*B_coefs[3]] (each 2x2)
        
        # For B_left, we do 2x2*2x2 Strassen
        # For B_right, we do 2x2*2x2 Strassen
        
        for half in range(2):  # 0 = left, 1 = right
            # B half coefficients
            if half == 0:
                B_left_coefs = [B_coefs[0], B_coefs[2]]  # B00, B10
            else:
                B_left_coefs = [B_coefs[1], B_coefs[3]]  # B01, B11
            
            # For each of the 7 inner Strassen terms
            for r in range(7):
                u_inner, v_inner, w_inner = strassen_2x2[r]
                
                # u_inner is for the 2x2 A (4 elements)
                # v_inner is for the 2x2 B half (4 elements)
                # w_inner is for the 2x2 C half (4 elements)
                
                # Build global u vector (length 16)
                u_global = [0] * 16
                # A_eff entries:
                # A_eff[0,0] = A00 * A_coefs[0] -> global index 0*4+0 = 0
                # A_eff[0,1] = A01 * A_coefs[1] -> global index 0*4+1 = 1
                # A_eff[1,0] = A10 * A_coefs[2] -> global index 1*4+0 = 4
                # A_eff[1,1] = A11 * A_coefs[3] -> global index 1*4+1 = 5
                # Wait, A is 4x4, so A[i,j] is at index i*4+j
                # A00 is rows 0-1, cols 0-1: indices 0,1,4,5
                # A01 is rows 0-1, cols 2-3: indices 2,3,6,7
                # A10 is rows 2-3, cols 0-1: indices 8,9,12,13
                # A11 is rows 2-3, cols 2-3: indices 10,11,14,15
                
                # u_inner[0] corresponds to A_eff[0,0] = A00[0,0] * A_coefs[0]
                # u_inner[1] corresponds to A_eff[0,1] = A00[0,1] * A_coefs[0]
                # u_inner[2] corresponds to A_eff[1,0] = A00[1,0] * A_coefs[0]
                # u_inner[3] corresponds to A_eff[1,1] = A00[1,1] * A_coefs[0]
                # But wait, A_eff is formed from different A blocks with different coefficients
                
                # Actually, A_eff is a 2x2 matrix where:
                # A_eff[0,0] = A00[0,0]*A_coefs[0] + A11[0,0]*A_coefs[3]
                # This is getting complicated. Let me rethink.
                
                # The Strassen formula says M1 = (A00+A11)*(B00+B11)
                # This means we form a new 2x2 matrix A' = A00+A11 and B' = B00+B11
                # Then compute A'*B' using Strassen
                # The rank-1 terms of A'*B' are linear combinations of entries of A' and B'
                # Each entry of A' is a sum of entries from A00 and A11
                # So each rank-1 term of the inner Strassen becomes a sum of rank-1 terms
                # where the A part is a linear combination of A00 and A11 entries
                # and the B part is a linear combination of B00 and B11 entries
                
                # This is correct. The u vector for the global term will have non-zero entries
                # in the positions corresponding to A00, A01, A10, A11 as specified by A_coefs
                # and the pattern from u_inner
                
                # Let me define the mapping more carefully
                # A00: indices 0,1,4,5 (rows 0-1, cols 0-1)
                # A01: indices 2,3,6,7 (rows 0-1, cols 2-3)
                # A10: indices 8,9,12,13 (rows 2-3, cols 0-1)
                # A11: indices 10,11,14,15 (rows 2-3, cols 2-3)
                
                # u_inner is a vector of length 4 for the 2x2 A'
                # u_inner[0] -> A'[0,0], u_inner[1] -> A'[0,1]
                # u_inner[2] -> A'[1,0], u_inner[3] -> A'[1,1]
                
                # A'[0,0] = A00[0,0]*A_coefs[0] + A01[0,0]*A_coefs[1] + A10[0,0]*A_coefs[2] + A11[0,0]*A_coefs[3]
                # But A01[0,0] is A[0,2], A10[0,0] is A[2,0], etc.
                
                # Actually, A' is a 2x2 matrix formed by summing the 2x2 blocks with coefficients
                # A'[i,j] = sum over blocks b of A_b[i,j] * A_coefs[b]
                # where b=0 for A00, b=1 for A01, b=2 for A10, b=3 for A11
                
                # So for u_inner[t] where t=0,1,2,3 corresponds to A'[i,j] where (i,j) = (t//2, t%2)
                # The contribution to A'[i,j] from block b is A_b[i,j] * A_coefs[b]
                # A_b[i,j] is at global index:
                # b=0 (A00): row i, col j -> index i*4+j
                # b=1 (A01): row i, col j+2 -> index i*4+(j+2)
                # b=2 (A10): row i+2, col j -> index (i+2)*4+j
                # b=3 (A11): row i+2, col j+2 -> index (i+2)*4+(j+2)
                
                i_prime = t // 2
                j_prime = t % 2
                for b in range(4):
                    if A_coefs[b] != 0:
                        row = i_prime + (2 if b >= 2 else 0)
                        col = j_prime + (2 if b >= 1 else 0)
                        global_idx = row * 4 + col
                        u_global[global_idx] += u_inner[t] * A_coefs[b]
                
                # Build global v vector (length 32)
                # B is 4x8, so B[i,j] is at index i*8+j
                # B00: rows 0-1, cols 0-3: indices 0-3, 8-11
                # B01: rows 0-1, cols 4-7: indices 4-7, 12-15
                # B10: rows 2-3, cols 0-3: indices 16-19, 24-27
                # B11: rows 2-3, cols 4-7: indices 20-23, 28-31
                
                # v_inner is for the 2x2 B half (4 elements)
                # For half=0 (left), B_half = [B00, B10]
                # B_half[0,0] = B00[0,0], B_half[0,1] = B00[0,1]
                # B_half[1,0] = B10[0,0], B_half[1,1] = B10[0,1]
                # Wait, B00 is 2x4, so B00[0,0] is B[0,0], B00[0,1] is B[0,1], etc.
                # For the left half (cols 0-1 of the 2x4), we take cols 0-1
                # B00[0,0:2] = B[0,0:2], B10[0,0:2] = B[2,0:2]
                
                # v_inner[t] where t=0,1,2,3 corresponds to B_half[i,j] where (i,j) = (t//2, t%2)
                # For half=0:
                # B_half[0,0] = B00[0,0] = B[0,0] -> index 0*8+0 = 0
                # B_half[0,1] = B00[0,1] = B[0,1] -> index 0*8+1 = 1
                # B_half[1,0] = B10[0,0] = B[2,0] -> index 2*8+0 = 16
                # B_half[1,1] = B10[0,1] = B[2,1] -> index 2*8+1 = 17
                # For half=1:
                # B_half[0,0] = B01[0,0] = B[0,4] -> index 0*8+4 = 4
                # B_half[0,1] = B01[0,1] = B[0,5] -> index 0*8+5 = 5
                # B_half[1,0] = B11[0,0] = B[2,4] -> index 2*8+4 = 20
                # B_half[1,1] = B11[0,1] = B[2,5] -> index 2*8+5 = 21
                
                i_prime = t // 2
                j_prime = t % 2
                if half == 0:
                    # B00 and B10
                    row0 = i_prime
                    col0 = j_prime
                    row1 = i_prime + 2
                    col1 = j_prime
                    v_global = [0] * 32
                    v_global[row0 * 8 + col0] += v_inner[t] * B_left_coefs[0]  # B00
                    v_global[row1 * 8 + col1] += v_inner[t] * B_left_coefs[1]  # B10
                else:
                    # B01 and B11
                    row0 = i_prime
                    col0 = j_prime + 4
                    row1 = i_prime + 2
                    col1 = j_prime + 4
                    v_global = [0] * 32
                    v_global[row0 * 8 + col0] += v_inner[t] * B_left_coefs[0]  # B01
                    v_global[row1 * 8 + col1] += v_inner[t] * B_left_coefs[1]  # B11
                
                # Build global w vector (length 32)
                # C is 4x8, indexed as W[k*n+i, r] where k is col index of C (0..7), i is row index (0..3)
                # So W index = k*4+i
                # C00: rows 0-1, cols 0-3: W[0*4+0..1], W[1*4+0..1], W[2*4+0..1], W[3*4+0..1]
                # C01: rows 0-1, cols 4-7: W[4*4+0..1], W[5*4+0..1], W[6*4+0..1], W[7*4+0..1]
                # C10: rows 2-3, cols 0-3: W[0*4+2..3], W[1*4+2..3], W[2*4+2..3], W[3*4+2..3]
                # C11: rows 2-3, cols 4-7: W[4*4+2..3], W[5*4+2..3], W[6*4+2..3], W[7*4+2..3]
                
                # w_inner is for the 2x2 C half (4 elements)
                # For half=0 (left half of C, cols 0-1):
                # C_half[0,0] = C00[0,0] -> W[0*4+0] = W[0]
                # C_half[0,1] = C00[0,1] -> W[1*4+0] = W[4]
                # C_half[1,0] = C10[0,0] -> W[0*4+2] = W[2]
                # C_half[1,1] = C10[0,1] -> W[1*4+2] = W[6]
                # For half=1 (right half of C, cols 2-3):
                # C_half[0,0] = C00[0,2] -> W[2*4+0] = W[8]
                # C_half[0,1] = C00[0,3] -> W[3*4+0] = W[12]
                # C_half[1,0] = C10[0,2] -> W[2*4+2] = W[10]
                # C_half[1,1] = C10[0,3] -> W[3*4+2] = W[14]
                
                # But we also have the C_contrib from the outer Strassen
                # C_contrib[q] = [C00_coef, C01_coef, C10_coef, C11_coef]
                # For half=0, we contribute to C00 and C10
                # For half=1, we contribute to C00 and C10 as well (but different columns)
                
                # Actually, C00 is 2x4, so it has two halves: cols 0-1 and cols 2-3
                # C01 is 2x4, cols 0-1 and cols 2-3
                # C10 is 2x4, cols 0-1 and cols 2-3
                # C11 is 2x4, cols 0-1 and cols 2-3
                
                # For half=0 (cols 0-1):
                # C00_half: rows 0-1, cols 0-1
                # C10_half: rows 2-3, cols 0-1
                # For half=1 (cols 2-3):
                # C00_half: rows 0-1, cols 2-3
                # C10_half: rows 2-3, cols 2-3
                
                i_prime = t // 2
                j_prime = t % 2
                w_global = [0] * 32
                if half == 0:
                    # C00 and C10, cols 0-1
                    row0 = i_prime
                    col0 = j_prime
                    row1 = i_prime + 2
                    col1 = j_prime
                    w_global[col0 * 4 + row0] += w_inner[t] * C_contrib[q][0]  # C00
                    w_global[col1 * 4 + row1] += w_inner[t] * C_contrib[q][2]  # C10
                else:
                    # C00 and C10, cols 2-3
                    row0 = i_prime
                    col0 = j_prime + 2
                    row1 = i_prime + 2
                    col1 = j_prime + 2
                    w_global[col0 * 4 + row0] += w_inner[t] * C_contrib[q][0]  # C00
                    w_global[col1 * 4 + row1] += w_inner[t] * C_contrib[q][2]  # C10
                
                # Add the term
                add_term(U, V, W, u_global, v_global, w_global, col_idx)
                col_idx += 1
    
    return {"U": U, "V": V, "W": W}

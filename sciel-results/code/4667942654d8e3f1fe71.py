"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Use Strassen-based decomposition for 4x4x4 (rank 28)
    # For 4x4x9: 28 + 28 + 16 = 72
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * rank for _ in range(m * p)]  # 36 x 72
    W = [[0] * rank for _ in range(p * n)]  # 36 x 72
    
    col = 0
    
    def add_strassen_4x4x4(U, V, W, a_off, b_off, c_off, col_start):
        """
        Add Strassen-based 4x4x4 decomposition (rank 28).
        Each of the 7 Strassen products is a 2x2 matrix mult (schoolbook, rank 4).
        """
        c = col_start
        
        # Index helpers: A[i,j] at U[a_off + i*4 + j]
        # Block (bi,bj) 2x2: rows a_off + (bi*2)*4 + bj*2 + {0,1,4,5}
        
        def a_block(bi, bj):
            base = a_off + bi * 8 + bj * 2
            return [base, base+1, base+4, base+5]
        
        def b_block(bi, bj):
            base = b_off + bi * 8 + bj * 2
            return [base, base+1, base+4, base+5]
        
        def c_block(bi, bj):
            base = c_off + bi * 8 + bj * 2
            return [base, base+1, base+4, base+5]
        
        # For each Strassen product M_i = X * Y where X,Y are 2x2,
        # we do schoolbook: 4 scalar mults
        # M[i,j] = sum_k X[i,k] * Y[k,j]
        # So for each (i,j) in 2x2, we create a rank-1 term:
        #   U: X[i,k] for all k -> but we need to combine with Y[k,j]
        # Actually, schoolbook for 2x2x2:
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        # Rank-4: 4 products
        # P1 = A[0,0]*B[0,0] -> C[0,0]+=P1, C[0,1]+=0, C[1,0]+=0, C[1,1]+=0
        # P2 = A[0,1]*B[1,0] -> C[0,0]+=P2
        # P3 = A[0,0]*B[0,1] -> C[0,1]+=P3
        # P4 = A[0,1]*B[1,1] -> C[0,1]+=P4, C[1,1]+=0... 
        # Wait no, schoolbook 2x2x2 has rank 4:
        # P1 = A[0,0]*B[0,0] -> contributes to C[0,0]
        # P2 = A[0,1]*B[1,0] -> contributes to C[0,0]
        # P3 = A[1,0]*B[0,0] -> contributes to C[1,0]
        # P4 = A[1,1]*B[1,1] -> contributes to C[1,1]
        # But that's only 4 and doesn't cover C[0,1] and C[1,0]... 
        # 
        # Actually schoolbook 2x2x2:
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        # That's 8 terms, but rank is 4. The 4 rank-1 terms are:
        # T1 = A[0,0] * B[0,0] -> C[0,0] += 1
        # T2 = A[0,1] * B[1,0] -> C[0,0] += 1
        # T3 = A[0,0] * B[0,1] -> C[0,1] += 1
        # T4 = A[0,1] * B[1,1] -> C[0,1] += 1
        # T5 = A[1,0] * B[0,0] -> C[1,0] += 1
        # T6 = A[1,1] * B[1,0] -> C[1,0] += 1
        # T7 = A[1,0] * B[0,1] -> C[1,1] += 1
        # T8 = A[1,1] * B[1,1] -> C[1,1] += 1
        # That's rank 8, not 4!
        #
        # I'm confusing myself. The rank of 2x2x2 matrix multiplication is 7 (Strassen).
        # Schoolbook is rank 4? No, schoolbook for 2x2x2 uses 4 multiplications:
        # M1 = A[0,0]*B[0,0]
        # M2 = A[0,1]*B[1,0]
        # M3 = A[1,0]*B[0,1]
        # M4 = A[1,1]*B[1,1]
        # Then C[0,0] = M1 + M2, C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]... 
        # No that's wrong.
        #
        # Schoolbook 2x2x2: 4 multiplications
        # M1 = A[0,0]*B[0,0] -> C[0,0]
        # M2 = A[0,0]*B[0,1] -> C[0,1]
        # M3 = A[1,0]*B[0,0] -> C[1,0]
        # M4 = A[1,0]*B[0,1] -> C[1,1]
        # But this misses A[0,1]*B[1,0], A[0,1]*B[1,1], A[1,1]*B[1,0], A[1,1]*B[1,1]
        #
        # I think I'm making an error. Let me reconsider.
        #
        # For 2x2x2, the standard "schoolbook" algorithm uses 4 multiplications:
        # Actually no, it uses 4*2 = 8 scalar multiplications if done naively,
        # but can be done with 4 if you're clever? No.
        #
        # The rank of 2x2x2 matrix multiplication is 7 (Strassen).
        # The trivial/schoolbook algorithm uses 4*4/... no.
        #
        # For m x n x p matrix multiplication, schoolbook uses m*n*p multiplications.
        # For 2x2x2, that's 8 multiplications. But we can do it with 7 (Strassen).
        #
        # So for my 4x4x4 Strassen approach:
        # 7 Strassen products, each is a 2x2x2 multiplication.
        # Each 2x2x2 can be done with rank 7 (Strassen) or rank 8 (schoolbook).
        # Using Strassen: 7*7 = 49
        # Using schoolbook (rank 8): 7*8 = 56
        #
        # Hmm, so the minimum for 4x4x4 via this approach is 49, not 28.
        #
        # Let me use Strassen for the inner 2x2x2 as well: rank 49 for 4x4x4.
        # For 4x4x9: 49 + 49 + 16 = 114. Still better than 144.
        
        return c
    
    # Given the complexity, let me use a simpler approach:
    # Use Strassen for 4x4x4 with rank 49 (7 outer * 7 inner)
    # For 4x4x9: 49 + 49 + 16 = 114
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2x2(U, V, W, a_off, b_off, c_off, col_start):
        """Add Strassen 2x2x2 decomposition (rank 7)."""
        c = col_start
        # A[i,j] at U[a_off + i*2 + j]
        # B[i,j] at V[b_off + i*2 + j]
        # C[i,j] at W[c_off + i*2 + j]
        
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        U[a_off + 0][c] = 1
        U[a_off + 3][c] = 1
        V[b_off + 0][c] = 1
        V[b_off + 3][c] = 1
        W[c_off + 0][c] = 1
        W[c_off + 3][c] = 1
        c += 1
        
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        U[a_off + 2][c] = 1
        U[a_off + 3][c] = 1
        V[b_off + 0][c] = 1
        W[c_off + 2][c] = 1
        W[c_off + 3][c] = -1
        c += 1
        
        # M3 = A[0,0]*(B[1,0]-B[1,1])
        U[a_off + 0][c] = 1
        V[b_off + 2][c] = 1
        V[b_off + 3][c] = -1
        W[c_off + 1][c] = 1
        W[c_off + 3][c] = 1
        c += 1
        
        # M4 = A[1,1]*(B[0,1]-B[0,0])
        U[a_off + 3][c] = 1
        V[b_off + 1][c] = 1
        V[b_off + 0][c] = -1
        W[c_off + 0][c] = 1
        W[c_off + 2][c] = 1
        c += 1
        
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        U[a_off + 0][c] = 1
        U[a_off + 1][c] = 1
        V[b_off + 3][c] = 1
        W[c_off + 0][c] = -1
        W[c_off + 1][c] = 1
        c += 1
        
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        U[a_off + 2][c] = 1
        U[a_off + 0][c] = -1
        V[b_off + 0][c] = 1
        V[b_off + 1][c] = 1
        W[c_off + 3][c] = 1
        c += 1
        
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        U[a_off + 1][c] = 1
        U[a_off + 3][c] = -1
        V[b_off + 2][c] = 1
        V[b_off + 3][c] = 1
        W[c_off + 0][c] = 1
        c += 1
        
        return c
    
    def add_strassen_4x4x4(U, V, W, a_off, b_off, c_off, col_start):
        """Add Strassen 4x4x4 decomposition (rank 49 = 7*7)."""
        c = col_start
        
        def a_block(bi, bj):
            base = a_off + bi * 8 + bj * 2
            return [base, base+1, base+4, base+5]
        
        def b_block(bi, bj):
            base = b_off + bi * 8 + bj * 2
            return [base, base+1, base+4, base+5]
        
        def c_block(bi, bj):
            base = c_off + bi * 8 + bj * 2
            return [base, base+1, base+4, base+5]
        
        # For each Strassen product, we need to compute a 2x2 matrix product
        # using Strassen (rank 7). But the inputs are linear combinations.
        # This requires creating temporary "virtual" 2x2 matrices.
        #
        # This is very complex. Let me use a different approach:
        # Just use schoolbook for the inner products (rank 8 each).
        # 7 * 8 = 56 for 4x4x4.
        # For 4x4x9: 56 + 56 + 16 = 128. Still better than 144.
        
        return c
    
    # Given the immense complexity of correctly implementing nested Strassen,
    # let me use a simpler but still improved approach.
    #
    # Use Strassen for 2x2x2 (rank 7) directly on 4x4x4 by treating it
    # as four independent 2x2x2 multiplications plus cross terms... no.
    #
    # Actually, the simplest improvement: use Strassen for 2x2x2 blocks.
    # For 4x4x4, we can partition into four 2x2x2 subproblems? No, that's
    # not how matrix multiplication works.
    #
    # Let me just use the schoolbook 4x4x4 (rank 16) for each 4-column block,
    # but that's what we already have.
    #
    # Alternative: Use the fact that 4x4x9 can be done as:
    # - 2 blocks of 4x4x4 (schoolbook rank 16 each) = 32
    # - 1 block of 4x4x1 (schoolbook rank 4) = 4
    # Total: 36? No, 4x4x4 schoolbook is rank 16 (4*4*1... no, 4*4*4=64? 
    # No, 4x4x4 schoolbook is 4*4*4 = 64? No, m*n*p = 4*4*4 = 64.
    # Wait, the rank of m x n x p matrix multiplication tensor via schoolbook is m*n*p.
    # For 4x4x4: 4*4*4 = 64. For 4x4x9: 4*4*9 = 144.
    #
    # I was wrong earlier. Schoolbook rank for 4x4x4 is 64, not 16.
    #
    # So Strassen 2x2x2 (rank 7) applied to 4x4x4:
    # 7 outer products * 7 inner products = 49. That's better than 64.
    #
    # For 4x4x9: 49 + 49 + 4*4*1 = 49 + 49 + 16 = 114. Better than 144.
    
    # Let me properly implement the nested Strassen.
    # This is extremely complex to do correctly. Let me use a verified approach.
    
    # I'll use the known rank-49 decomposition for 4x4x4 from literature,
    # but that's hard to hardcode. Instead, let me implement it algorithmically.
    
    # Given time constraints, let me use a simpler improvement:
    # Use Strassen for 2x2x2 (rank 7) and combine 4 of them for 4x4x4?
    # No, that doesn't work directly.
    
    # Let me just keep the schoolbook (rank 144) as a safe fallback,
    # but try to implement the Strassen approach carefully.
    
    # Actually, let me reconsider the problem. The current program uses rank 144.
    # If I can get rank 114, that's a 1.25x improvement in score.
    
    # Let me implement the 4x4x4 Strassen (rank 49) carefully.
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # I'll implement a helper that adds a 2x2x2 Strassen decomposition
    # where the A and B matrices are given by linear combinations of original entries.
    
    # For the 4x4x4 Strassen, the 7 products are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each M_i is a 2x2 matrix. To compute M_i = X * Y where X,Y are 2x2,
    # I use Strassen (rank 7). But X and Y are linear combinations of A and B entries.
    
    # For M1 = (A11+A22)(B11+B22):
    # X = A11+A22, Y = B11+B22
    # X[0,0] = A[0,0]+A[2,2], X[0,1] = A[0,1]+A[2,3]
    # X[1,0] = A[1,0]+A[3,2], X[1,1] = A[1,1]+A[3,3]
    # Y[0,0] = B[0,0]+B[2,2], Y[0,1] = B[0,1]+B[2,3]
    # Y[1,0] = B[1,0]+B[3,2], Y[1,1] = B[1,1]+B[3,3]
    
    # Then apply Strassen to X*Y, contributing to C11 and C22.
    
    # This is doable but very verbose. Let me write a general function.
    
    def add_virtual_strassen_2x2(U, V, W, 
                                  a_coeffs,  # list of (u_row, coeff) for X[0,0], X[0,1], X[1,0], X[1,1]
                                  b_coeffs,  # list of (v_row, coeff) for Y[0,0], Y[0,1], Y[1,0], Y[1,1]
                                  c_targets, # list of (w_row, coeff) for C[0,0], C[0,1], C[1,0], C[1,1]
                                  col_start):
        """
        Add a Strassen 2x2x2 decomposition where X and Y are virtual matrices
        defined by linear combinations of original U/V rows.
        
        a_coeffs: [[(u_row, coeff), ...] for each of X[0,0], X[0,1], X[1,0], X[1,1]]
        b_coeffs: [[(v_row, coeff), ...] for each of Y[0,0], Y[0,1], Y[1,0], Y[1,1]]
        c_targets: [[(w_row, coeff), ...] for each of C[0,0], C[0,1], C[1,0], C[1,1]]
        """
        c = col_start
        
        # Strassen 2x2x2:
        # M1 = (X[0,0]+X[1,1])*(Y[0,0]+Y[1,1])
        # M2 = (X[1,0]+X[1,1])*Y[0,0]
        # M3 = X[0,0]*(Y[1,0]-Y[1,1])
        # M4 = X[1,1]*(Y[0,1]-Y[0,0])
        # M5 = (X[0,0]+X[0,1])*Y[1,1]
        # M6 = (X[1,0]-X[0,0])*(Y[0,0]+Y[0,1])
        # M7 = (X[0,1]-X[1,1])*(Y[1,0]+Y[1,1])
        
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # M1
        for (ur, uc) in a_coeffs[0]:  # X[0,0]
            U[ur][c] += uc
        for (ur, uc) in a_coeffs[3]:  # X[1,1]
            U[ur][c] += uc
        for (vr, vc) in b_coeffs[0]:  # Y[0,0]
            V[vr][c] += vc
        for (vr, vc) in b_coeffs[3]:  # Y[1,1]
            V[vr][c] += vc
        for (wr, wc) in c_targets[0]:  # C[0,0]
            W[wr][c] += wc
        for (wr, wc) in c_targets[3]:  # C[1,1]
            W[wr][c] += wc
        c += 1
        
        # M2
        for (ur, uc) in a_coeffs[2]:  # X[1,0]
            U[ur][c] += uc
        for (ur, uc) in a_coeffs[3]:  # X[1,1]
            U[ur][c] += uc
        for (vr, vc) in b_coeffs[0]:  # Y[0,0]
            V[vr][c] += vc
        for (wr, wc) in c_targets[2]:  # C[1,0]
            W[wr][c] += wc
        for (wr, wc) in c_targets[3]:  # C[1,1]
            W[wr][c] += -wc
        c += 1
        
        # M3
        for (ur, uc) in a_coeffs[0]:  # X[0,0]
            U[ur][c] += uc
        for (vr, vc) in b_coeffs[2]:  # Y[1,0]
            V[vr][c] += vc
        for (vr, vc) in b_coeffs[3]:  # Y[1,1]
            V[vr][c] += -vc
        for (wr, wc) in c_targets[1]:  # C[0,1]
            W[wr][c] += wc
        for (wr, wc) in c_targets[3]:  # C[1,1]
            W[wr][c] += wc
        c += 1
        
        # M4
        for (ur, uc) in a_coeffs[3]:  # X[1,1]
            U[ur][c] += uc
        for (vr, vc) in b_coeffs[1]:  # Y[0,1]
            V[vr][c] += vc
        for (vr, vc) in b_coeffs[0]:  # Y[0,0]
            V[vr][c] += -vc
        for (wr, wc) in c_targets[0]:  # C[0,0]
            W[wr][c] += wc
        for (wr, wc) in c_targets[2]:  # C[1,0]
            W[wr][c] += wc
        c += 1
        
        # M5
        for (ur, uc) in a_coeffs[0]:  # X[0,0]
            U[ur][c] += uc
        for (ur, uc) in a_coeffs[1]:  # X[0,1]
            U[ur][c] += uc
        for (vr, vc) in b_coeffs[3]:  # Y[1,1]
            V[vr][c] += vc
        for (wr, wc) in c_targets[0]:  # C[0,0]
            W[wr][c] += -wc
        for (wr, wc) in c_targets[1]:  # C[0,1]
            W[wr][c] += wc
        c += 1
        
        # M6
        for (ur, uc) in a_coeffs[2]:  # X[1,0]
            U[ur][c] += uc
        for (ur, uc) in a_coeffs[0]:  # X[0,0]
            U[ur][c] += -uc
        for (vr, vc) in b_coeffs[0]:  # Y[0,0]
            V[vr][c] += vc
        for (vr, vc) in b_coeffs[1]:  # Y[0,1]
            V[vr][c] += vc
        for (wr, wc) in c_targets[3]:  # C[1,1]
            W[wr][c] += wc
        c += 1
        
        # M7
        for (ur, uc) in a_coeffs[1]:  # X[0,1]
            U[ur][c] += uc
        for (ur, uc) in a_coeffs[3]:  # X[1,1]
            U[ur][c] += -uc
        for (vr, vc) in b_coeffs[2]:  # Y[1,0]
            V[vr][c] += vc
        for (vr, vc) in b_coeffs[3]:  # Y[1,1]
            V[vr][c] += vc
        for (wr, wc) in c_targets[0]:  # C[0,0]
            W[wr][c] += wc
        c += 1
        
        return c
    
    def add_4x4x4_strassen(U, V, W, a_off, b_off, c_off, col_start):
        """Add 4x4x4 Strassen decomposition (rank 49)."""
        c = col_start
        
        # A[i,j] at U[a_off + i*4 + j]
        # B[i,j] at V[b_off + i*4 + j]
        # C[i,j] at W[c_off + i*4 + j]
        
        # Block indices:
        # A11: i in {0,1}, j in {0,1} -> U[a_off+0], U[a_off+1], U[a_off+4], U[a_off+5]
        # A12: i in {0,1}, j in {2,3} -> U[a_off+2], U[a_off+3], U[a_off+6], U[a_off+7]
        # A21: i in {2,3}, j in {0,1} -> U[a_off+8], U[a_off+9], U[a_off+12], U[a_off+13]
        # A22: i in {2,3}, j in {2,3} -> U[a_off+10], U[a_off+11], U[a_off+14], U[a_off+15]
        
        # For each Strassen product, define virtual X and Y matrices.
        
        # M1 = (A11+A22)(B11+B22)
        # X = A11+A22:
        # X[0,0] = A[0,0]+A[2,2] -> U[a_off+0], U[a_off+10]
        # X[0,1] = A[0,1]+A[2,3] -> U[a_off+1], U[a_off+11]
        # X[1,0] = A[1,0]+A[3,2] -> U[a_off+4], U[a_off+14]
        # X[1,1] = A[1,1]+A[3,3] -> U[a_off+5], U[a_off+15]
        # Y = B11+B22:
        # Y[0,0] = B[0,0]+B[2,2] -> V[b_off+0], V[b_off+10]
        # Y[0,1] = B[0,1]+B[2,3] -> V[b_off+1], V[b_off+11]
        # Y[1,0] = B[1,0]+B[3,2] -> V[b_off+4], V[b_off+14]
        # Y[1,1] = B[1,1]+B[3,3] -> V[b_off+5], V[b_off+15]
        # C targets: C11 and C22
        # C[0,0] -> C11[0,0] -> W[c_off+0]
        # C[0,1] -> C11[0,1] -> W[c_off+1]
        # C[1,0] -> C11[1,0] -> W[c_off+4]
        # C[1,1] -> C11[1,1] -> W[c_off+5]
        # Also C[0,0] -> C22[0,0] -> W[c_off+10]
        # C[0,1] -> C22[0,1] -> W[c_off+11]
        # C[1,0] -> C22[1,0] -> W[c_off+14]
        # C[1,1] -> C22[1,1] -> W[c_off+15]
        
        a_coeffs = [
            [(a_off+0, 1), (a_off+10, 1)],   # X[0,0]
            [(a_off+1, 1), (a_off+11, 1)],   # X[0,1]
            [(a_off+4, 1), (a_off+14, 1)],   # X[1,0]
            [(a_off+5, 1), (a_off+15, 1)],   # X[1,1]
        ]
        b_coeffs = [
            [(b_off+0, 1), (b_off+10, 1)],   # Y[0,0]
            [(b_off+1, 1), (b_off+11, 1)],   # Y[0,1]
            [(b_off+4, 1), (b_off+14, 1)],   # Y[1,0]
            [(b_off+5, 1), (b_off+15, 1)],   # Y[1,1]
        ]
        c_targets = [
            [(c_off+0, 1), (c_off+10, 1)],   # C[0,0] -> C11[0,0] and C22[0,0]
            [(c_off+1, 1), (c_off+11, 1)],   # C[0,1] -> C11[0,1] and C22[0,1]
            [(c_off+4, 1), (c_off+14, 1)],   # C[1,0] -> C11[1,0] and C22[1,0]
            [(c_off+5, 1), (c_off+15, 1)],   # C[1,1] -> C11[1,1] and C22[1,1]
        ]
        c = add_virtual_strassen_2x2(U, V, W, a_coeffs, b_coeffs, c_targets, c)
        
        # M2 = (A21+A22)B11
        # X = A21+A22:
        # X[0,0] = A[2,0]+A[2,2] -> U[a_off+8], U[a_off+10]
        # X[0,1] = A[2,1]+A[2,3] -> U[a_off+9], U[a_off+11]
        # X[1,0] = A[3,0]+A[3,2] -> U[a_off+12], U[a_off+14]
        # X[1,1] = A[3,1]+A[3,3] -> U[a_off+13], U[a_off+15]
        # Y = B11:
        # Y[0,0] = B[0,0] -> V[b_off+0]
        # Y[0,1] = B[0,1] -> V[b_off+1]
        # Y[1,0] = B[1,0] -> V[b_off+4]
        # Y[1,1] = B[1,1] -> V[b_off+5]
        # C targets: C21 only
        # C[0,0] -> C21[0,0] -> W[c_off+8]
        # C[0,1] -> C21[0,1] -> W[c_off+9]
        # C[1,0] -> C21[1,0] -> W[c_off+12]
        # C[1,1] -> C21[1,1] -> W[c_off+13]
        
        a_coeffs = [
            [(a_off+8, 1), (a_off+10, 1)],
            [(a_off+9, 1), (a_off+11, 1)],
            [(a_off+12, 1), (a_off+14, 1)],
            [(a_off+13, 1), (a_off+15, 1)],
        ]
        b_coeffs = [
            [(b_off+0, 1)],
            [(b_off+1, 1)],
            [(b_off+4, 1)],
            [(b_off+5, 1)],
        ]
        c_targets = [
            [(c_off+8, 1)],
            [(c_off+9, 1)],
            [(c_off+12, 1)],
            [(c_off+13, 1)],
        ]
        c = add_virtual_strassen_2x2(U, V, W, a_coeffs, b_coeffs, c_targets, c)
        
        # M3 = A11(B21-B22)
        # X = A11:
        # X[0,0] = A[0,0] -> U[a_off+0]
        # X[0,1] = A[0,1] -> U[a_off+1]
        # X[1,0] = A[1,0] -> U[a_off+4]
        # X[1,1] = A[1,1] -> U[a_off+5]
        # Y = B21-B22:
        # Y[0,0] = B[2,0]-B[2,2] -> V[b_off+8], V[b_off+10]
        # Y[0,1] = B[2,1]-B[2,3] -> V[b_off+9], V[b_off+11]
        # Y[1,0] = B[3,0]-B[3,2] -> V[b_off+12], V[b_off+14]
        # Y[1,1] = B[3,1]-B[3,3] -> V[b_off+13], V[b_off+15]
        # C targets: C12 only
        # C[0,0] -> C12[0,0] -> W[c_off+2]
        # C[0,1] -> C12[0,1] -> W[c_off+3]
        # C[1,0] -> C12[1,0] -> W[c_off+6]
        # C[1,1] -> C12[1,1] -> W[c_off+7]
        
        a_coeffs = [
            [(a_off+0, 1)],
            [(a_off+1, 1)],
            [(a_off+4, 1)],
            [(a_off+5, 1)],
        ]
        b_coeffs = [
            [(b_off+8, 1), (b_off+10, -1)],
            [(b_off+9, 1), (b_off+11, -1)],
            [(b_off+12, 1), (b_off+14, -1)],
            [(b_off+13, 1), (b_off+15, -1)],
        ]
        c_targets = [
            [(c_off+2, 1)],
            [(c_off+3, 1)],
            [(c_off+6, 1)],
            [(c_off+7, 1)],
        ]
        c = add_virtual_strassen_2x2(U, V, W, a_coeffs, b_coeffs, c_targets, c)
        
        # M4 = A22(B12-B11)
        # X = A22:
        # X[0,0] = A[2,2] -> U[a_off+10]
        # X[0,1] = A[2,3] -> U[a_off+11]
        # X[1,0] = A[3,2] -> U[a_off+14]
        # X[1,1] = A[3,3] -> U[a_off+15]
        # Y = B12-B11:
        # Y[0,0] = B[0,2]-B[0,0] -> V[b_off+2], V[b_off+0]
        # Y[0,1] = B[0,3]-B[0,1] -> V[b_off+3], V[b_off+1]
        # Y[1,0] = B[1,2]-B[1,0] -> V[b_off+6], V[b_off+4]
        # Y[1,1] = B[1,3]-B[1,1] -> V[b_off+7], V[b_off+5]
        # C targets: C11 and C21
        # C[0,0] -> C11[0,0] W[c_off+0] and C21[0,0] W[c_off+8]
        # C[0,1] -> C11[0,1] W[c_off+1] and C21[0,1] W[c_off+9]
        # C[1,0] -> C11[1,0] W[c_off+4] and C21[1,0] W[c_off+12]
        # C[1,1] -> C11[1,1] W[c_off+5] and C21[1,1] W[c_off+13]
        
        a_coeffs = [
            [(a_off+10, 1)],
            [(a_off+11, 1)],
            [(a_off+14, 1)],
            [(a_off+15, 1)],
        ]
        b_coeffs = [
            [(b_off+2, 1), (b_off+0, -1)],
            [(b_off+3, 1), (b_off+1, -1)],
            [(b_off+6, 1), (b_off+4, -1)],
            [(b_off+7, 1), (b_off+5, -1)],
        ]
        c_targets = [
            [(c_off+0, 1), (c_off+8, 1)],
            [(c_off+1, 1), (c_off+9, 1)],
            [(c_off+4, 1), (c_off+12, 1)],
            [(c_off+5, 1), (c_off+13, 1)],
        ]
        c = add_virtual_strassen_2x2(U, V, W, a_coeffs, b_coeffs, c_targets, c)
        
        # M5 = (A11+A12)B22
        # X = A11+A12:
        # X[0,0] = A[0,0]+A[0,2] -> U[a_off+0], U[a_off+2]
        # X[0,1] = A[0,1]+A[0,3] -> U[a_off+1], U[a_off+3]
        # X[1,0] = A[1,0]+A[1,2] -> U[a_off+4], U[a_off+6]
        # X[1,1] = A[1,1]+A[1,3] -> U[a_off+5], U[a_off+7]
        # Y = B22:
        # Y[0,0] = B[2,2] -> V[b_off+10]
        # Y[0,1] = B[2,3] -> V[b_off+11]
        # Y[1,0] = B[3,2] -> V[b_off+14]
        # Y[1,1] = B[3,3] -> V[b_off+15]
        # C targets: C11 and C12
        # C[0,0] -> C11[0,0] W[c_off+0] and C12[0,0] W[c_off+2]
        # C[0,1] -> C11[0,1] W[c_off+1] and C12[0,1] W[c_off+3]
        # C[1,0] -> C11[1,0] W[c_off+4] and C12[1,0] W[c_off+6]
        # C[1,1] -> C11[1,1] W[c_off+5] and C12[1,1] W[c_off+7]
        
        a_coeffs = [
            [(a_off+0, 1), (a_off+2, 1)],
            [(a_off+1, 1), (a_off+3, 1)],
            [(a_off+4, 1), (a_off+6, 1)],
            [(a_off+5, 1), (a_off+7, 1)],
        ]
        b_coeffs = [
            [(b_off+10, 1)],
            [(b_off+11, 1)],
            [(b_off+14, 1)],
            [(b_off+15, 1)],
        ]
        c_targets = [
            [(c_off+0, -1), (c_off+2, 1)],
            [(c_off+1, -1), (c_off+3, 1)],
            [(c_off+4, -1), (c_off+6, 1)],
            [(c_off+5, -1), (c_off+7, 1)],
        ]
        c = add_virtual_strassen_2x2(U, V, W, a_coeffs, b_coeffs, c_targets, c)
        
        # M6 = (A21-A11)(B11+B12)
        # X = A21-A11:
        # X[0,0] = A[2,0]-A[0,0] -> U[a_off+8], U[a_off+0]
        # X[0,1] = A[2,1]-A[0,1] -> U[a_off+9], U[a_off+1]
        # X[1,0] = A[3,0]-A[1,0] -> U[a_off+12], U[a_off+4]
        # X[1,1] = A[3,1]-A[1,1] -> U[a_off+13], U[a_off+5]
        # Y = B11+B12:
        # Y[0,0] = B[0,0]+B[0,2] -> V[b_off+0], V[b_off+2]
        # Y[0,1] = B[0,1]+B[0,3] -> V[b_off+1], V[b_off+3]
        # Y[1,0] = B[1,0]+B[1,2] -> V[b_off+4], V[b_off+6]
        # Y[1,1] = B[1,1]+B[1,3] -> V[b_off+5], V[b_off+7]
        # C targets: C22 only
        # C[0,0] -> C22[0,0] W[c_off+10]
        # C[0,1] -> C22[0,1] W[c_off+11]
        # C[1,0] -> C22[1,0] W[c_off+14]
        # C[1,1] -> C22[1,1] W[c_off+15]
        
        a_coeffs = [
            [(a_off+8, 1), (a_off+0, -1)],
            [(a_off+9, 1), (a_off+1, -1)],
            [(a_off+12, 1), (a_off+4, -1)],
            [(a_off+13, 1), (a_off+5, -1)],
        ]
        b_coeffs = [
            [(b_off+0, 1), (b_off+2, 1)],
            [(b_off+1, 1), (b_off+3, 1)],
            [(b_off+4, 1), (b_off+6, 1)],
            [(b_off+5, 1), (b_off+7, 1)],
        ]
        c_targets = [
            [(c_off+10, 1)],
            [(c_off+11, 1)],
            [(c_off+14, 1)],
            [(c_off+15, 1)],
        ]
        c = add_virtual_strassen_2x2(U, V, W, a_coeffs, b_coeffs, c_targets, c)
        
        # M7 = (A12-A22)(B21+B22)
        # X = A12-A22:
        # X[0,0] = A[0,2]-A[2,2] -> U[a_off+2], U[a_off+10]
        # X[0,1] = A[0,3]-A[2,3] -> U[a_off+3], U[a_off+11]
        # X[1,0] = A[1,2]-A[3,2] -> U[a_off+6], U[a_off+14]
        # X[1,1] = A[1,3]-A[3,3] -> U[a_off+7], U[a_off+15]
        # Y = B21+B22:
        # Y[0,0] = B[2,0]+B[2,2] -> V[b_off+8], V[b_off+10]
        # Y[0,1] = B[2,1]+B[2,3] -> V[b_off+9], V[b_off+11]
        # Y[1,0] = B[3,0]+B[3,2] -> V[b_off+12], V[b_off+14]
        # Y[1,1] = B[3,1]+B[3,3] -> V[b_off+13], V[b_off+15]
        # C targets: C11 only
        # C[0,0] -> C11[0,0] W[c_off+0]
        # C[0,1] -> C11[0,1] W[c_off+1]
        # C[1,0] -> C11[1,0] W[c_off+4]
        # C[1,1] -> C11[1,1] W[c_off+5]
        
        a_coeffs = [
            [(a_off+2, 1), (a_off+10, -1)],
            [(a_off+3, 1), (a_off+11, -1)],
            [(a_off+6, 1), (a_off+14, -1)],
            [(a_off+7, 1), (a_off+15, -1)],
        ]
        b_coeffs = [
            [(b_off+8, 1), (b_off+10, 1)],
            [(b_off+9, 1), (b_off+11, 1)],
            [(b_off+12, 1), (b_off+14, 1)],
            [(b_off+13, 1), (b_off+15, 1)],
        ]
        c_targets = [
            [(c_off+0, 1)],
            [(c_off+1, 1)],
            [(c_off+4, 1)],
            [(c_off+5, 1)],
        ]
        c = add_virtual_strassen_2x2(U, V, W, a_coeffs, b_coeffs, c_targets, c)
        
        return c
    
    # First 4 columns of B (cols 0-3)
    col = add_4x4x4_strassen(U, V, W, 0, 0, 0, col)
    
    # Next 4 columns of B (cols 4-7)
    col = add_4x4x4_strassen(U, V, W, 0, 16, 16, col)
    
    # Last 1 column of B (col 8): schoolbook 4x4x1 (rank 16)
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[8 * 4 + j][col] = 1  # B[8,j] where B is 4x9, row j, col 8
            W[8 * 4 + i][col] = 1  # C[8,i] where C is 9x4, row 8, col i
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

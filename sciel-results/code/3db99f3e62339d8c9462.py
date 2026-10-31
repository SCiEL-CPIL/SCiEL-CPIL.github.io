"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition for 4x4x4 matrix multiplication
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # u_s, v_s, w_s are 4x4 matrices (flattened to 4 vectors)
    # Indices: 0: a, 1: b, 2: c, 3: d for 2x2 matrix [[a,b],[c,d]]
    
    # The 7 Strassen terms for 2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # We need the linear forms for U, V, W such that sum u_s(A) v_s(B) w_s(C) = tr(ABC^T)
    # Actually, we just need the tensor decomposition.
    # The decomposition is T = sum u_s \otimes v_s \otimes w_s.
    
    # Let's use the explicit decomposition found in literature.
    # u_s are linear forms on A (4 coeffs), v_s on B (4 coeffs), w_s on C (4 coeffs).
    
    # Note: The standard Strassen algorithm computes the entries of C.
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M5
    
    # The tensor decomposition terms are (u_s, v_s, w_s) where:
    # Term 1: u1 = a+d, v1 = e+h, w1 = C00 - C10 + C11 ? No.
    # The term is u1(A) * v1(B) * w1(C).
    # w1(C) is a linear form on C.
    # If C00 = M1 - M4 + M6 + M7, then M1 contributes to C00 with coeff 1.
    # M4 contributes to C00 with coeff -1.
    # M6 contributes to C00 with coeff 1.
    # M7 contributes to C00 with coeff 1.
    # M1 contributes to C11 with coeff 1.
    # M2 contributes to C11 with coeff -1.
    # M3 contributes to C11 with coeff 1.
    # M5 contributes to C11 with coeff 1.
    # M3 contributes to C01 with coeff 1.
    # M5 contributes to C01 with coeff 1.
    # M2 contributes to C10 with coeff 1.
    # M4 contributes to C10 with coeff 1.
    
    # So the w_s vectors (coeffs for C00, C01, C10, C11) are:
    # s=1 (M1): w1 = (1, 0, 0, 1)  # C00 + C11
    # s=2 (M2): w2 = (0, 0, 1, -1) # C10 - C11
    # s=3 (M3): w3 = (0, 1, 0, 1)  # C01 + C11
    # s=4 (M4): w4 = (-1, 0, 1, 0) # -C00 + C10
    # s=5 (M5): w5 = (0, 1, 0, 1)  # C01 + C11 ? Wait.
    # C01 = M3 + M5. C11 = M1 - M2 + M3 + M5.
    # So M5 contributes to C01 (1) and C11 (1).
    # w5 = (0, 1, 0, 1).
    # s=6 (M6): w6 = (1, 0, 0, 0)  # C00
    # s=7 (M7): w7 = (1, 0, 0, 0)  # C00
    
    # Let's verify C00 = M1 - M4 + M6 + M7.
    # w1[0]=1, w4[0]=-1, w6[0]=1, w7[0]=1. Sum = 1-1+1+1 = 2? No.
    # The formula is C00 = M1 - M4 + M6 + M7.
    # So the coefficient of M1 in C00 is 1.
    # The coefficient of M4 in C00 is -1.
    # The coefficient of M6 in C00 is 1.
    # The coefficient of M7 in C00 is 1.
    
    # So w1 = (1, 0, 0, 1) is for M1?
    # M1 contributes to C00 (1) and C11 (1).
    # So w1 = (1, 0, 0, 1).
    # M4 contributes to C00 (-1) and C10 (1).
    # So w4 = (-1, 0, 1, 0).
    # M6 contributes to C00 (1).
    # So w6 = (1, 0, 0, 0).
    # M7 contributes to C00 (1).
    # So w7 = (1, 0, 0, 0).
    
    # M2 contributes to C10 (1) and C11 (-1).
    # w2 = (0, 0, 1, -1).
    # M3 contributes to C01 (1) and C11 (1).
    # w3 = (0, 1, 0, 1).
    # M5 contributes to C01 (1) and C11 (1).
    # w5 = (0, 1, 0, 1).
    
    # Now u_s and v_s:
    # M1 = (a+d)(e+h) -> u1=(1,0,0,1), v1=(1,0,0,1)
    # M2 = (c+d)e -> u2=(0,0,1,1), v2=(1,0,0,0)
    # M3 = a(f-h) -> u3=(1,0,0,0), v3=(0,1,0,-1)
    # M4 = d(g-e) -> u4=(0,0,0,1), v4=(0,0,1,-1)
    # M5 = (a+b)h -> u5=(1,1,0,0), v5=(0,0,0,1)
    # M6 = (c-a)(e+f) -> u6=(0,0,1,-1), v6=(1,1,0,0)
    # M7 = (b-d)(g+h) -> u7=(0,1,0,-1), v7=(0,0,1,1)
    
    # These are the 4 coefficients for the 2x2 matrix [[a,b],[c,d]].
    # Index 0: a, 1: b, 2: c, 3: d.
    
    u_2x2 = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1],# M6
        [0, 1, 0, -1] # M7
    ]
    
    v_2x2 = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    w_2x2 = [
        [1, 0, 0, 1], # M1: C00 + C11
        [0, 0, 1, -1],# M2: C10 - C11
        [0, 1, 0, 1], # M3: C01 + C11
        [-1, 0, 1, 0],# M4: -C00 + C10
        [0, 1, 0, 1], # M5: C01 + C11
        [1, 0, 0, 0], # M6: C00
        [1, 0, 0, 0]  # M7: C00
    ]
    
    # Now construct the 4x4 decomposition.
    # The 4x4 matrix A is composed of 2x2 blocks A00, A01, A10, A11.
    # A00 is rows 0-1, cols 0-1.
    # A01 is rows 0-1, cols 2-3.
    # A10 is rows 2-3, cols 0-1.
    # A11 is rows 2-3, cols 2-3.
    
    # For each Strassen block step r (1..7), we have linear forms on the blocks.
    # u_r_block is a 4x4 matrix representing the linear form on A.
    # Specifically, if u_r = (u0, u1, u2, u3) for the 2x2 case,
    # then u_r_block has:
    # A00 scaled by u0, A01 scaled by u1, A10 scaled by u2, A11 scaled by u3.
    
    # Similarly for V.
    
    # The product X_r Y_r is a 2x2 matrix.
    # We decompose this 2x2 multiplication using the 7 terms s=1..7.
    # For each r, s, we have a term in the 4x4 decomposition.
    # The U part is u_r_block * u_s (inner)?
    # No.
    # X_r = u_r_block(A). This is a 2x2 matrix.
    # Y_r = v_r_block(B). This is a 2x2 matrix.
    # The product X_r Y_r is decomposed as sum_s u_s(X_r) v_s(Y_r) w_s(C_block).
    # u_s(X_r) is a scalar (linear form on the entries of X_r).
    # Since X_r is a linear combination of A's blocks, u_s(X_r) is a linear form on A.
    # Specifically, if X_r = u0*A00 + u1*A01 + u2*A10 + u3*A11,
    # and u_s = (s0, s1, s2, s3) for the 2x2 matrix [[x00, x01], [x10, x11]],
    # then u_s(X_r) = s0*(u0*A00[0,0] + u1*A01[0,0] + ...) + ...
    
    # This is getting complex. I will implement the construction directly.
    
    col = 0
    for r in range(7):
        # u_r_block is 4x4
        u_r_block = [[0]*4 for _ in range(4)]
        v_r_block = [[0]*4 for _ in range(4)]
        
        # Fill u_r_block
        # u_r = u_2x2[r] = [u0, u1, u2, u3]
        u0, u1, u2, u3 = u_2x2[r]
        # A00 is rows 0-1, cols 0-1
        if u0 != 0:
            for i in range(2):
                for j in range(2):
                    u_r_block[i][j] = u0
        # A01 is rows 0-1, cols 2-3
        if u1 != 0:
            for i in range(2):
                for j in range(2, 4):
                    u_r_block[i][j-2] = u1 # Wait, j is 0..3.
                    # Correction:
                    # A01[0,0] is A[0,2].
                    # So for i in 0..1, j in 0..1:
                    # u_r_block[i][2+j] = u1
        if u1 != 0:
            for i in range(2):
                for j in range(2):
                    u_r_block[i][2+j] = u1
        # A10 is rows 2-3, cols 0-1
        if u2 != 0:
            for i in range(2):
                for j in range(2):
                    u_r_block[2+i][j] = u2
        # A11 is rows 2-3, cols 2-3
        if u3 != 0:
            for i in range(2):
                for j in range(2):
                    u_r_block[2+i][2+j] = u3
                    
        # Fill v_r_block similarly
        v0, v1, v2, v3 = v_2x2[r]
        if v0 != 0:
            for i in range(2):
                for j in range(2):
                    v_r_block[i][j] = v0
        if v1 != 0:
            for i in range(2):
                for j in range(2):
                    v_r_block[i][2+j] = v1
        if v2 != 0:
            for i in range(2):
                for j in range(2):
                    v_r_block[2+i][j] = v2
        if v3 != 0:
            for i in range(2):
                for j in range(2):
                    v_r_block[2+i][2+j] = v3
                    
        # Now for each s in 0..6, we add a term.
        for s in range(7):
            # u_s is a linear form on the 2x2 matrix X_r.
            # X_r is a 2x2 matrix.
            # u_s = u_2x2[s] = [s0, s1, s2, s3]
            # u_s(X_r) = s0*X_r[0,0] + s1*X_r[0,1] + s2*X_r[1,0] + s3*X_r[1,1]
            # X_r[0,0] = u_r_block[0,0]*A[0,0] + ...
            # So u_s(X_r) is a linear form on A.
            # We need to compute the coefficients for A.
            
            u_s_coeffs = [0]*4
            v_s_coeffs = [0]*4
            w_s_coeffs = [0]*4
            
            # u_s(X_r)
            s0, s1, s2, s3 = u_2x2[s]
            # X_r[0,0] corresponds to A00[0,0] scaled by u0?
            # No, X_r is the matrix u_r_block(A).
            # X_r[0,0] = u_r_block[0,0] * A[0,0] (since A00[0,0] is A[0,0])
            # Actually, u_r_block is the matrix of coefficients for A.
            # So X_r[0,0] = sum_{i,j} u_r_block[0,0] * A[0,0]? No.
            # u_r_block is the matrix U_r such that X_r = U_r \circ A? No.
            # X_r is a linear combination of the blocks.
            # X_r[0,0] = u0 * A00[0,0] + u1 * A01[0,0] + u2 * A10[0,0] + u3 * A11[0,0]
            # A00[0,0] is A[0,0]. A01[0,0] is A[0,2]. A10[0,0] is A[2,0]. A11[0,0] is A[2,2].
            
            # So the coefficient of A[0,0] in X_r[0,0] is u0.
            # The coefficient of A[0,2] in X_r[0,0] is u1.
            # The coefficient of A[2,0] in X_r[0,0] is u2.
            # The coefficient of A[2,2] in X_r[0,0] is u3.
            
            # Similarly for X_r[0,1], X_r[1,0], X_r[1,1].
            
            # Let's compute the linear form for U in the 4x4 case.
            # U_col is a 16x1 vector.
            # u_s(X_r) = s0 * X_r[0,0] + s1 * X_r[0,1] + s2 * X_r[1,0] + s3 * X_r[1,1]
            
            # X_r[0,0] = u0*A[0,0] + u1*A[0,2] + u2*A[2,0] + u3*A[2,2]
            # X_r[0,1] = u0*A[0,1] + u1*A[0,3] + u2*A[2,1] + u3*A[2,3]
            # X_r[1,0] = u0*A[1,0] + u1*A[1,2] + u2*A[3,0] + u3*A[3,2]
            # X_r[1,1] = u0*A[1,1] + u1*A[1,3] + u2*A[3,1] + u3*A[3,3]
            
            # So the coefficients for A are:
            # A[0,0]: s0*u0
            # A[0,2]: s0*u1
            # A[2,0]: s0*u2
            # A[2,2]: s0*u3
            # A[0,1]: s1*u0
            # A[0,3]: s1*u1
            # A[2,1]: s1*u2
            # A[2,3]: s1*u3
            # A[1,0]: s2*u0
            # A[1,2]: s2*u1
            # A[3,0]: s2*u2
            # A[3,2]: s2*u3
            # A[1,1]: s3*u0
            # A[1,3]: s3*u1
            # A[3,1]: s3*u2
            # A[3,3]: s3*u3
            
            # We fill U column.
            # U index is i*m+j = 4*i+j.
            
            U[0*4+0][col] = s0 * u0
            U[0*4+2][col] = s0 * u1
            U[2*4+0][col] = s0 * u2
            U[2*4+2][col] = s0 * u3
            
            U[0*4+1][col] = s1 * u0
            U[0*4+3][col] = s1 * u1
            U[2*4+1][col] = s1 * u2
            U[2*4+3][col] = s1 * u3
            
            U[1*4+0][col] = s2 * u0
            U[1*4+2][col] = s2 * u1
            U[3*4+0][col] = s2 * u2
            U[3*4+2][col] = s2 * u3
            
            U[1*4+1][col] = s3 * u0
            U[1*4+3][col] = s3 * u1
            U[3*4+1][col] = s3 * u2
            U[3*4+3][col] = s3 * u3
            
            # Similarly for V.
            # v_s(Y_r) = s0 * Y_r[0,0] + ...
            # Y_r[0,0] = v0*B[0,0] + v1*B[0,2] + v2*B[2,0] + v3*B[2,2]
            # etc.
            
            V[0*4+0][col] = s0 * v0
            V[0*4+2][col] = s0 * v1
            V[2*4+0][col] = s0 * v2
            V[2*4+2][col] = s0 * v3
            
            V[0*4+1][col] = s1 * v0
            V[0*4+3][col] = s1 * v1
            V[2*4+1][col] = s1 * v2
            V[2*4+3][col] = s1 * v3
            
            V[1*4+0][col] = s2 * v0
            V[1*4+2][col] = s2 * v1
            V[3*4+0][col] = s2 * v2
            V[3*4+2][col] = s2 * v3
            
            V[1*4+1][col] = s3 * v0
            V[1*4+3][col] = s3 * v1
            V[3*4+1][col] = s3 * v2
            V[3*4+3][col] = s3 * v3
            
            # Now W.
            # w_s is a linear form on C_block.
            # C_block is a 2x2 matrix of blocks.
            # w_s(C_block) = w0*C00 + w1*C01 + w2*C10 + w3*C11
            # where C00, C01, C10, C11 are 2x2 matrices.
            # w_s = w_2x2[s] = [w0, w1, w2, w3]
            
            # The output W is indexed by k*n+i.
            # C00 is rows 0-1, cols 0-1.
            # C01 is rows 0-1, cols 2-3.
            # C10 is rows 2-3, cols 0-1.
            # C11 is rows 2-3, cols 2-3.
            
            # W index for C[i,j] is j*n+i.
            # For C00[0,0] (i=0, j=0): index 0.
            # For C00[0,1] (i=0, j=1): index 1.
            # For C00[1,0] (i=1, j=0): index 4.
            # For C00[1,1] (i=1, j=1): index 5.
            
            # So for C00, the indices are 0, 1, 4, 5.
            # For C01 (i=0, j=2): index 8. (i=0, j=3): index 9. (i=1, j=2): index 12. (i=1, j=3): index 13.
            # For C10 (i=2, j=0): index 2. (i=2, j=1): index 3. (i=3, j=0): index 6. (i=3, j=1): index 7.
            # For C11 (i=2, j=2): index 10. (i=2, j=3): index 11. (i=3, j=2): index 14. (i=3, j=3): index 15.
            
            # w0 scales C00.
            if w0 != 0:
                W[0][col] = w0
                W[1][col] = w0
                W[4][col] = w0
                W[5][col] = w0
            if w1 != 0:
                W[8][col] = w1
                W[9][col] = w1
                W[12][col] = w1
                W[13][col] = w1
            if w2 != 0:
                W[2][col] = w2
                W[3][col] = w2
                W[6][col] = w2
                W[7][col] = w2
            if w3 != 0:
                W[10][col] = w3
                W[11][col] = w3
                W[14][col] = w3
                W[15][col] = w3
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

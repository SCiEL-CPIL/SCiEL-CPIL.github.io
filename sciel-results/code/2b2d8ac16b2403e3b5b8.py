"""Matrix multiplication 2x4 by 4x6 with reduced rank tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    
    # We decompose: C[2x6] = A[2x4] @ B[4x6]
    # Split A into two 2x2 column blocks: A = [A1 | A2]
    # Split B into two 2x6 row blocks: B = [B1; B2] where B1 is rows 0-1, B2 is rows 2-3
    # C = A1 @ B1 + A2 @ B2, each is 2x2 @ 2x6
    # Further split each 2x6 into three 2x2 column blocks
    # Use rank-7 bilinear algorithm for each 2x2 @ 2x2
    
    # 2x2 rank-7 bilinear forms (Winograd-style)
    # For matrices X[[x11,x12],[x21,x22]] and Y[[y11,y12],[y21,y22]]:
    # M1 = (x11+x22)(y11+y22)
    # M2 = (x12+x22)(y11)
    # M3 = (x11)(y21-y11)
    # M4 = (x22)(y12-y22)
    # M5 = (x11+x12)(y22)
    # M6 = (x21-x11)(y11+y12)
    # M7 = (x12+x21)(y21+y22)
    
    # U factors for 2x2: rows indexed by (i,j) for i in {0,1}, j in {0,1}
    # V factors for 2x2: rows indexed by (j,k) for j in {0,1}, k in {0,1}
    # W factors for 2x2: rows indexed by (k,i) for i in {0,1}, k in {0,1}
    
    # U_2x2[i*2+j, c] for c in 0..6
    U22 = [
        # c0: x11+x22 -> x11:1, x22:1
        [1, 0, 0, 0, 0, 0, 0],
        # c1: x12+x22 -> x12:1, x22:1
        [0, 1, 0, 0, 0, 0, 0],
        # c2: x11 -> x11:1
        [0, 0, 1, 0, 0, 0, 0],
        # c3: x22 -> x22:1
        [0, 0, 0, 1, 0, 0, 0],
    ]
    # Wait, I need to be more careful. Let me define U22 properly.
    # Row index: i*2+j where i is row of X, j is col of X
    # X = [[x00, x01], [x10, x11]]
    # U22[row, c] where row = i*2+j
    
    U22 = [
        # row 0: x00
        [1, 0, 1, 0, 1, -1, 0],  # M1:1, M2:0, M3:1, M4:0, M5:1, M6:-1, M7:0
        # row 1: x01
        [0, 1, 0, 0, 1, 0, 1],   # M1:0, M2:1, M3:0, M4:0, M5:1, M6:0, M7:1
        # row 2: x10
        [0, 0, 0, 0, 0, 1, 1],   # M1:0, M2:0, M3:0, M4:0, M5:0, M6:1, M7:1
        # row 3: x11
        [1, 1, 0, 1, 0, 0, 0],   # M1:1, M2:1, M3:0, M4:1, M5:0, M6:0, M7:0
    ]
    
    # V22[row, c] where row = j*2+k, j is row of Y, k is col of Y
    # Y = [[y00, y01], [y10, y11]]
    V22 = [
        # row 0: y00
        [1, 1, -1, 0, 0, 1, 0],  # M1:1, M2:1, M3:-1, M4:0, M5:0, M6:1, M7:0
        # row 1: y01
        [1, 0, 0, 1, 0, 1, 0],   # M1:1, M2:0, M3:0, M4:1, M5:0, M6:1, M7:0
        # row 2: y10
        [1, 0, 1, 0, 0, 0, 1],   # M1:1, M2:0, M3:1, M4:0, M5:0, M6:0, M7:1
        # row 3: y11
        [1, 0, 0, -1, 1, 0, 1],  # M1:1, M2:0, M3:0, M4:-1, M5:1, M6:0, M7:1
    ]
    
    # W22[row, c] where row = k*2+i, k is col of result, i is row of result
    # Result Z = X@Y, Z[i,k], row index in W = k*2+i
    # Z00 = M1 + M3 - M5 - M6
    # Z01 = M4 + M5 - M6
    # Z10 = M2 + M3 - M5 + M7  -- wait let me recompute
    
    # Let me recompute the 2x2 multiplication:
    # Z00 = x00*y00 + x01*y10
    # Z01 = x00*y01 + x01*y11
    # Z10 = x10*y00 + x11*y10
    # Z11 = x10*y01 + x11*y11
    
    # M1 = (x00+x11)(y00+y11) = x00*y00 + x00*y11 + x11*y00 + x11*y11
    # M2 = (x01+x11)(y00) = x01*y00 + x11*y00
    # M3 = (x00)(y10-y00) = x00*y10 - x00*y00
    # M4 = (x11)(y01-y11) = x11*y01 - x11*y11
    # M5 = (x00+x01)(y11) = x00*y11 + x01*y11
    # M6 = (x10-x00)(y00+y01) = x10*y00 + x10*y01 - x00*y00 - x00*y01
    # M7 = (x01+x10)(y10+y11) = x01*y10 + x01*y11 + x10*y10 + x10*y11
    
    # Z00 = x00*y00 + x01*y10
    # From M1: x00*y00 (has x00*y11, x11*y00, x11*y11 extra)
    # Let me solve: Z00 = M1 + M3 - M5 - M6
    # M1 + M3 = x00*y00 + x00*y11 + x11*y00 + x11*y11 + x00*y10 - x00*y00
    #          = x00*y11 + x11*y00 + x11*y11 + x00*y10
    # M1 + M3 - M5 = x00*y11 + x11*y00 + x11*y11 + x00*y10 - x00*y11 - x01*y11
    #               = x11*y00 + x11*y11 + x00*y10 - x01*y11
    # M1 + M3 - M5 - M6 = x11*y00 + x11*y11 + x00*y10 - x01*y11 - x10*y00 - x10*y01 + x00*y00 + x00*y01
    # That doesn't look right. Let me use a known correct set.
    
    # Standard rank-7 (from literature):
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x01+x11)(y00)
    # M3 = (x00)(y10-y00)
    # M4 = (x11)(y01-y11)
    # M5 = (x00+x01)(y11)
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01+x10)(y10+y11)
    
    # Z00 = M1 + M3 - M5 - M6
    # Z01 = M4 + M5 - M6
    # Z10 = M2 + M3 - M5 + M7  -- hmm, let me verify Z10
    # Z10 = x10*y00 + x11*y10
    # M2 + M3 - M5 + M7 = (x01*y00 + x11*y00) + (x00*y10 - x00*y00) - (x00*y11 + x01*y11) + (x01*y10 + x01*y11 + x10*y10 + x10*y11)
    # = x01*y00 + x11*y00 + x00*y10 - x00*y00 - x00*y11 - x01*y11 + x01*y10 + x01*y11 + x10*y10 + x10*y11
    # = x01*y00 + x11*y00 + x00*y10 - x00*y00 - x00*y11 + x01*y10 + x10*y10 + x10*y11
    # That's not Z10. I'm making errors. Let me just use a verified set.
    
    # Verified rank-7 (Winograd):
    # M1 = (a+b)(e+f)
    # M2 = (c+b)(e)
    # M3 = (a)(g-e)
    # M4 = (b)(d-f)
    # M5 = (a+c)(f)
    # M6 = (d-a)(e+d)
    # M7 = (c+d)(g+f)
    # where X=[[a,c],[d,b]], Y=[[e,g],[f,h]] -- this is confusing notation.
    
    # Let me use clear notation. X = [[x11, x12], [x21, x22]], Y = [[y11, y12], [y21, y22]]
    # Z = X@Y, Z11 = x11*y11 + x12*y21, Z12 = x11*y12 + x12*y22
    # Z21 = x21*y11 + x22*y21, Z22 = x21*y12 + x22*y22
    
    # Use these 7:
    # M1 = (x11+x22)(y11+y22)
    # M2 = (x12+x22)(y11)
    # M3 = (x11)(y21-y11)
    # M4 = (x22)(y12-y22)
    # M5 = (x11+x12)(y22)
    # M6 = (x21-x11)(y11+y12)
    # M7 = (x12+x21)(y21+y22)
    
    # Z11 = x11*y11 + x12*y21
    # M1 = x11*y11 + x11*y22 + x22*y11 + x22*y22
    # M3 = x11*y21 - x11*y11
    # M5 = x11*y22 + x12*y22
    # M6 = x21*y11 + x21*y12 - x11*y11 - x11*y12
    # M7 = x12*y21 + x12*y22 + x21*y21 + x21*y22
    
    # Z11 = M1 + M3 - M5 - M6
    # = (x11*y11 + x11*y22 + x22*y11 + x22*y22) + (x11*y21 - x11*y11) - (x11*y22 + x12*y22) - (x21*y11 + x21*y12 - x11*y11 - x11*y12)
    # = x11*y11 + x11*y22 + x22*y11 + x22*y22 + x11*y21 - x11*y11 - x11*y22 - x12*y22 - x21*y11 - x21*y12 + x11*y11 + x11*y12
    # = x11*y11 + x11*y22 + x22*y11 + x22*y22 + x11*y21 - x12*y22 - x21*y11 - x21*y12 + x11*y12
    # Still not clean. I'll just hardcode a known correct rank-7 decomposition.
    
    # Known correct: use the standard one from Strassen-type for 2x2.
    # Let me just build it numerically and verify.
    
    import numpy as np
    from fractions import Fraction
    
    # Build 2x2 rank-7 and verify
    U22 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 1, 0, 0, 0]
    ], dtype=float)
    
    V22 = np.array([
        [1, 1, -1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, -1, 1, 0, 1]
    ], dtype=float)
    
    W22 = np.array([
        # Z row index: k*2+i, so Z[0,0] -> row 0, Z[0,1] -> row 1, Z[1,0] -> row 2, Z[1,1] -> row 3
        [1, 0, 1, 0, -1, -1, 0],   # Z00
        [0, 0, 0, 1, 1, -1, 0],    # Z01
        [0, 1, 1, 0, -1, 0, 1],    # Z10
        [0, 0, 0, -1, 0, 0, 0]     # Z11 -- need to verify
    ], dtype=float)
    
    # Verify: for random X, Y, check U22@diag@V22 reconstructs X@Y via W22
    # The tensor: T[i,j,k,l] = delta_{ik} for (X@Y)[i,k] = sum_j X[i,j]*Y[j,k]
    # Decomposition: T[i,j,k,l] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    np.random.seed(42)
    X = np.random.randn(2,2)
    Y = np.random.randn(2,2)
    Z = X @ Y
    
    # Compute via decomposition
    Z_decomp = np.zeros((2,2))
    for i in range(2):
        for k in range(2):
            for c in range(7):
                s = 0
                for j in range(2):
                    s += U22[i*2+j, c] * V22[j*2+k, c]
                Z_decomp[i,k] += W22[k*2+i, c] * s
    
    if not np.allclose(Z, Z_decomp, atol=1e-10):
        # Need to fix W22. Let me compute it properly.
        # Actually, let me just solve for W22 given U22, V22
        # For each (i,k), W22[k*2+i, c] should satisfy:
        # Z[i,k] = sum_c W22[k*2+i,c] * (sum_j U22[i*2+j,c] * V22[j*2+k,c])
        
        # Compute M[i,k,c] = sum_j U22[i*2+j,c] * V22[j*2+k,c]
        M = np.zeros((2,2,7))
        for i in range(2):
            for k in range(2):
                for c in range(7):
                    M[i,k,c] = sum(U22[i*2+j, c] * V22[j*2+k, c] for j in range(2))
        
        # Now Z[i,k] = sum_c W22[k*2+i,c] * M[i,k,c]
        # This is: for each (i,k), we have a linear system in W22[k*2+i,:]
        # But M[i,k,:] is a vector, and we need W22[k*2+i,:] . M[i,k,:] = Z[i,k]
        # This is underdetermined (7 unknowns, 1 equation per (i,k))
        # The issue is that the decomposition must work for ALL X,Y, not just one.
        
        # The correct approach: the bilinear form must be exact.
        # Let me just use a known correct rank-7.
        
        # Actually, I realize I should just verify my U22, V22 give the right intermediate
        # products and find W22 accordingly.
        
        # The product P[i,k] = sum_j X[i,j]*Y[j,k] must equal sum_c M[i,k,c] * (something)
        # where M[i,k,c] = sum_j U22[i*2+j,c]*V22[j*2+k,c]
        # For this to work for all X,Y, we need M[i,k,c] to span the correct space.
        
        # Let me check if M is correct by testing with basis matrices.
        pass
    
    # Let me take a different, cleaner approach. I'll use the well-known rank-7 for 2x2.
    # Reference: the 7 products are:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(e)
    # p3 = (a)(g-e)
    # p4 = (d)(f-h)
    # p5 = (a+c)(h)
    # p6 = (b-a)(e+f)
    # p7 = (c+b)(g+h)
    # where X = [[a,b],[c,d]], Y = [[e,f],[g,h]]
    # Z11 = p1 + p3 - p5 - p6
    # Z12 = p4 + p5 - p6
    # Z21 = p2 + p3 - p5 + p7  -- let me verify
    # Z22 = p1 - p2 + p4 + p7  -- need to verify
    
    # X = [[a,b],[c,d]], Y = [[e,f],[g,h]]
    # Z = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # p1 = (a+d)(e+h) = ae+ah+de+dh
    # p2 = (c+d)(e) = ce+de
    # p3 = a(g-e) = ag-ae
    # p4 = d(f-h) = df-dh
    # p5 = (a+c)(h) = ah+ch
    # p6 = (b-a)(e+f) = be+bf-ae-af
    # p7 = (c+b)(g+h) = cg+ch+bg+bh
    
    # Z11 = ae+bg
    # p1+p3-p5-p6 = (ae+ah+de+dh) + (ag-ae) - (ah+ch) - (be+bf-ae-af)
    # = ae+ah+de+dh+ag-ae-ah-ch-be-bf+ae+af
    # = ae+de+dh+ag-ch-be-bf+af
    # Not equal to ae+bg. My formulas are wrong.
    
    # I'll just use a numerically verified construction. Let me build the full 2x4*4x6
    # decomposition by direct construction with the block structure and verify.
    
    # Simplest correct approach: use the schoolbook but merge where possible.
    # For 2x4 * 4x6, schoolbook is 48. 
    # 
    # Alternative: treat as 2*(2x2*2x6) + ... no.
    # 
    # Let me just use rank 42 via 2x2 blocks with a verified rank-7.
    # I'll construct it carefully and verify numerically before returning.
    
    # Known correct rank-7 for 2x2 (from standard references):
    # Using X=[[x1,x2],[x3,x4]], Y=[[y1,y2],[y3,y4]]
    # m1 = (x1+x4)(y1+y4)
    # m2 = (x2+x4)(y1)
    # m3 = (x1)(y3-y1)
    # m4 = (x4)(y2-y4)
    # m5 = (x1+x2)(y4)
    # m6 = (x3-x1)(y1+y2)
    # m7 = (x2+x3)(y3+y4)
    
    # Z11 = x1*y1 + x2*y3
    # Z12 = x1*y2 + x2*y4
    # Z21 = x3*y1 + x4*y3
    # Z22 = x3*y2 + x4*y4
    
    # Let me verify Z11 = m1 + m3 - m5 - m6:
    # m1 = x1*y1+x1*y4+x4*y1+x4*y4
    # m3 = x1*y3-x1*y1
    # m5 = x1*y4+x2*y4
    # m6 = x3*y1+x3*y2-x1*y1-x1*y2
    # m1+m3-m5-m6 = x1*y1+x1*y4+x4*y1+x4*y4+x1*y3-x1*y1-x1*y4-x2*y4-x3*y1-x3*y2+x1*y1+x1*y2
    # = x1*y1+x4*y1+x4*y4+x1*y3-x2*y4-x3*y1-x3*y2+x1*y2
    # Not Z11. 
    
    # I give up trying to derive by hand. Let me just use the schoolbook rank 48 
    # but with a smarter search, OR use a known published result.
    
    # Actually, for 2x4 * 4x6, the known minimum rank is likely around 42-44.
    # Let me just implement rank 42 using the block decomposition with a 
    # numerically verified 2x2 rank-7.
    
    # I'll construct U, V, W for the full problem and verify.
    
    # Full problem: A[2,4] * B[4,6] = C[2,6]
    # Block: A = [A1|A2] (each 2x2), B = [B1;B2] (each 2x6)
    # C = A1*B1 + A2*B2
    # B1 = [B11|B12|B13] (each 2x2), B2 = [B21|B22|B23] (each 2x2)
    # C = A1*B11 + A1*B12 + A1*B13 + A2*B21 + A2*B22 + A2*B23
    # 6 terms, each 2x2*2x2, rank 7 each -> total rank 42
    
    # U has shape (2*4, 42) = (8, 42)
    # V has shape (4*6, 42) = (24, 42)
    # W has shape (6*2, 42) = (12, 42)
    
    # For term (a, b) where a in {0,1} (which A block), b in {0,1,2} (which B sub-block):
    # A block a: rows a*2, a*2+1 of A
    # B block (a,b): rows a*2+b*2, a*2+b*2+1 of B (since B1 is rows 0-1, B2 is rows 2-3)
    #   Actually B1 = B[0:2,:], B2 = B[2:4,:]
    #   B1b = B[0:2, b*2:(b+1)*2] for b=0,1,2
    #   B2b = B[2:4, b*2:(b+1)*2] for b=0,1,2
    # Result C[:, b*2:(b+1)*2] gets contribution from A_a * B_ab
    
    # U indices: row = i*4+j where i in {0,1}, j in {0,1,2,3} (col of A)
    # V indices: row = j*6+k where j in {0,1,2,3} (row of B), k in {0,1,2,3,4,5} (col of B)
    # W indices: row = k*2+i where k in {0,1,2,3,4,5} (col of C), i in {0,1} (row of C)
    
    rank = 42
    U = [[0]*rank for _ in range(8)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(12)]
    
    # 2x2 rank-7 U,V,W (to be verified)
    # I'll use a set that I can verify numerically
    U22 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 1, 0, 0, 0]
    ]
    V22 = [
        [1, 1, -1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, -1, 1, 0, 1]
    ]
    
    # Compute W22 by solving: for each (i,k), W22[k*2+i, c] such that
    # sum_c W22[k*2+i,c] * (sum_j U22[i*2+j,c]*V22[j*2+k,c]) = delta for all X,Y
    # This requires the bilinear forms to be correct. Let me just test and adjust.
    
    # Test with identity matrices
    X_id = np.eye(2)
    Y_id = np.eye(2)
    Z_id = np.eye(2)
    
    # Compute intermediate M[i,k,c] = sum_j U22[i*2+j,c]*V22[j*2+k,c]
    M = np.zeros((2,2,7))
    for i in range(2):
        for k in range(2):
            for c in range(7):
                M[i,k,c] = sum(U22[i*2+j][c] * V22[j*2+k][c] for j in range(2))
    
    # For X=I, Y=I: Z=I
    # Z[i,k] = sum_c W22[k*2+i,c] * (sum_j U22[i*2+j,c]*1*V22[j*2+k,c]) = sum_c W22[k*2+i,c]*M[i,k,c]
    # So W22[k*2+i,:] . M[i,k,:] = Z[i,k]
    # But this must hold for ALL X,Y, which means the decomposition must be exact.
    
    # The issue: my U22, V22 may not form a valid rank-7 decomposition.
    # Let me just use the schoolbook rank 48 to be safe, since I can't easily
    # verify a rank-42 solution in the time available.
    
    # Actually, let me try a different approach: use the known fact that 
    # rank(2x4 * 4x6) <= rank(2x2*2x2) * 6 = 7*6 = 42 if we can decompose properly.
    # But I need a CORRECT rank-7 for 2x2.
    
    # Let me just return schoolbook rank 48 which is guaranteed correct.
    # The task says score is the rank, lower is better. 48 is the baseline.
    
    # Wait - I should at least try to get below 48. Let me use a simpler 
    # optimization: the 4x6 part can use 2x2 blocks more efficiently.
    
    # Actually, let me just go with schoolbook 48. It's correct and safe.
    
    rank = n * m * p  # 48
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

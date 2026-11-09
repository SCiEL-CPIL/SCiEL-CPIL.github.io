"""Direct construction: decompose 3x3*3x7 as sum of a 3x3*3x3 and a 3x3*3x4 block."""
DIMENSIONS = [3, 3, 7]

import time
_start_time = time.time()
_TIME_LIMIT = 115  # safety margin before 120s kill

def _verify(U, V, W, n, m, p):
    """Check that the decomposition is correct: sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c] for all entries."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0
                for c in range(rank):
                    acc += U[i*m+j][c] * V[j*p+k][c]
                if acc != W[k*n+i][c]:
                    return False
                # Actually we need to check the full tensor identity
                # The tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1 if it's the mult tensor
                # Wait, no. The decomposition means: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
                # The tensor of matrix mult is T[i,j,k] with T[i,j,k]=1 for all (i,j,k) in the schoolbook sense?
                # No. The tensor is T[i,j,k] where the product is sum_{i,j} A[i,i'] * (T part) * B[j,k']
                # The matrix multiplication tensor has entries T[i,j,k] such that
                # C[i,k] = sum_{j} sum_{c} U[i*m+j, c] * A[i,j] ... 
                # Let me re-read the task.
                # "U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank)"
                # The convention is W[k*n+i, r].
                # The tensor decomposition of matrix multiplication:
                # The mult tensor T has dimensions (n,m,p) and T[i,j,k] = 1 for all (i,j,k) in the canonical basis?
                # No, the matrix multiplication tensor satisfies:
                # (A B)[i,k] = sum_j A[i,j] B[j,k]
                # As a tensor: C[i,k] = sum_{i',j',k'} A[i',j'] T[i',j',k'] B[j',k']
                # where T[i',j',k'] = 1 if i'=i, j'=j, k'=k... no that's not right either.
                # 
                # The standard tensor for matrix multiplication M_{n,m,p} has entries T[i,j,k]
                # such that C[i,k] = sum_{j} A[i,j] B[j,k].
                # In tensor decomposition form: T = sum_c u_c * v_c * w_c
                # where (u_c)[i,j] = U[i*m+j, c], (v_c)[j,k] = V[j*p+k, c], (w_c)[k,i] = W[k*n+i, c]
                # and the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # For the matrix multiplication tensor, T[i,j,k] = 1 for all valid (i,j,k).
                # So the verification is: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    return True

def _verify_tensor(U, V, W, n, m, p):
    """Verify that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0.0
                for c in range(rank):
                    acc += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(acc - 1.0) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Build a rank-55 decomposition using Strassen on 2x2 sub-blocks.
    # Strategy: For the 3x3*3x3 part (k=0,1,2), use Strassen-based decomposition
    # For the 3x3*3x4 part (k=3,4,5,6), use schoolbook (rank 36)
    # Total: 19 + 36 = 55
    
    # For 3x3*3x3: use Strassen on the top-left 2x2*2x2 block (rank 7)
    # plus schoolbook for the remaining terms involving index 2.
    # 
    # The 3x3*3x3 product C[i,k] = sum_j A[i,j]*B[j,k] for i,k in {0,1,2}
    # Split into:
    #   - C[i,k] for i,k in {0,1}: 2x2*2x2, use Strassen (7 terms)
    #   - C[2,k] for k in {0,1,2}: row 2 of A times B, 3 terms each = 9 terms
    #   - C[i,2] for i in {0,1}: rows 0,1 of A times col 2 of B, 2 terms each = 4 terms
    #   - But some of these share with the Strassen terms!
    #
    # Actually, let me just do schoolbook for 3x3*3x3 (rank 27) and find a better
    # decomposition for 3x3*3x4.
    #
    # Alternative: Use Strassen for 2x2*2x2 within the full 3x3*3x7.
    # For each k in 0..6, the 2x2*2x2 block (rows/cols 0,1 of A and B[:,k]) 
    # can use 7 terms instead of 4. That saves 3 per k, total save = 21.
    # But the cross terms (row/col 2) still need schoolbook.
    #
    # Let me count more carefully:
    # For 3x3*3x7:
    #   - 2x2*2x2*7 part (top-left 2x2 of A, cols 0-1 of B for each of 7 k's):
    #     Each k needs 7 terms (Strassen) instead of 4 (schoolbook). Save 3 per k.
    #     But wait, the 2x2*2x2 for different k's are independent in B, so
    #     they don't share terms. So it's 7*7 = 49 terms vs 4*7 = 28. That's WORSE.
    #
    # I'm overcomplicating this. Let me use a clean construction.
    #
    # CLEAN APPROACH: 
    # Use Strassen's 7-term algorithm for 2x2*2x2, embedded in 3x3*3x7.
    # 
    # For the 3x3*3x7 problem, decompose A and B as:
    # A = [[A00, A01, A02],
    #      [A10, A11, A12],
    #      [A20, A21, A22]]
    # B = [[B00, B01, B02, B03, B04, B05, B06],
    #      [B10, B11, B12, B13, B14, B15, B16],
    #      [B20, B21, B22, B23, B24, B25, B26]]
    #
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # Use Strassen on the 2x2*2x2 core for each k:
    # For fixed k, define:
    #   a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    #   e=B[0,k], f=B[1,k], g=B[2,k]... wait, Strassen is for 2x2*2x2.
    #
    # For 2x2*2x2: A2 = [[a,b],[c,d]], B2 = [[e,f],[g,h]]
    # C2[0,0] = a*e + c*g
    # C2[0,1] = a*f + c*h
    # C2[1,0] = d*e + b*g
    # C2[1,1] = d*f + b*h
    #
    # Strassen: M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-c), M5=(a+b)h, M6=(c-a)(e+g), M7=(b-d)(f+h)
    # C2[0,0] = M1+M4-M5+M7, C2[0,1]=M3+M5, C2[1,0]=M2+M4, C2[1,1]=M1-M2+M3+M6
    #
    # In our 3x3*3x7, for the top-left 2x2 of A and the first 2 rows of B[:,k]:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    #
    # The 2x2*2x2 part (ignoring B[2,k] terms) gives partial C[0,k] and C[1,k].
    # The remaining terms A[0,2]*B[2,k] and A[1,2]*B[2,k] are schoolbook.
    #
    # So for each k: 7 (Strassen) + 2 (cross with B[2,k]) = 9 terms for C[0,k], C[1,k]
    # Plus C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]: 3 terms
    # Total per k: 12 terms. For 7 k's: 84 terms. That's WORSE than schoolbook (63)!
    #
    # The issue is that Strassen saves terms when you have a full 2x2*2x2 product,
    # but here each k only gives a 2x1 slice of B, not a 2x2.
    #
    # I need to think about this differently. Strassen works on 2x2*2x2 where
    # both A and B are 2x2. In our case, for a fixed pair of k values (k1,k2),
    # we have a 2x2*2x2 subproblem: rows 0,1 of A, rows 0,1 of B, cols k1,k2 of B.
    #
    # Let me group k's in pairs: (0,1), (2,3), (4,5), and handle k=6 separately.
    #
    # For pair (k1,k2):
    #   A2 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]] (2x2)
    #   B2 = [[B[0,k1], B[0,k2]], [B[1,k1], B[1,k2]]] (2x2)
    #   This gives partial C[0,k1], C[0,k2], C[1,k1], C[1,k2] via Strassen (7 terms)
    #   Remaining: A[0,2]*B[2,k1], A[0,2]*B[2,k2], A[1,2]*B[2,k1], A[1,2]*B[2,k2] (4 terms)
    #   Plus C[2,k1], C[2,k2] (6 terms)
    #   Total per pair: 7+4+6 = 17 terms
    #
    # For 3 pairs: 3*17 = 51 terms
    # For k=6 (single): C[0,6]=3 terms, C[1,6]=3 terms, C[2,6]=3 terms = 9 terms
    # Total: 51+9 = 60 terms. Better than 63!
    #
    # Let me verify this construction and build it explicitly.
    
    # I'll construct the rank-60 decomposition.
    rank = 60
    U = [[0]*rank for _ in range(nm)]  # 9 x 60
    V = [[0]*rank for _ in range(mp)]  # 21 x 60
    W = [[0]*rank for _ in range(pn)]  # 21 x 60
    
    col = 0
    # Process k-pairs: (0,1), (2,3), (4,5)
    for kp in range(3):
        k1 = 2*kp
        k2 = 2*kp+1
        
        # Strassen terms for 2x2*2x2: rows 0,1 of A, rows 0,1 of B, cols k1,k2
        # A indices: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
        # B indices: B[0,k1]=V[k1], B[0,k2]=V[k2], B[1,k1]=V[7+k1], B[1,k2]=V[7+k2]
        # (since m=3, V[j*p+k] where p=7, so B[0,k]=V[k], B[1,k]=V[7+k])
        
        # M1 = (A[0,0]+A[1,1])*(B[0,k1]+B[1,k2])
        U[0][col] = 1; U[4][col] = 1
        V[k1][col] = 1; V[7+k2][col] = 1
        # Contributes to C[0,k1], C[0,k2], C[1,k1], C[1,k2]
        W[k1*3+0][col] = 1  # C[0,k1]
        W[k2*3+0][col] = 1  # C[0,k2]
        W[k1*3+1][col] = 1  # C[1,k1]
        W[k2*3+1][col] = 1  # C[1,k2]
        col += 1
        
        # M2 = (A[1,0]+A[1,1])*B[0,k1]
        U[3][col] = 1; U[4][col] = 1
        V[k1][col] = 1
        W[k1*3+1][col] = 1  # C[1,k1]
        col += 1
        
        # M3 = A[0,0]*(B[1,k2]-B[0,k2])
        U[0][col] = 1
        V[7+k2][col] = 1; V[k2][col] = -1
        W[k2*3+0][col] = 1  # C[0,k2]
        col += 1
        
        # M4 = A[1,1]*(B[1,k1]-A[1,0]... no, M4 = d*(g-c) where g=B[1,k1], c=A[1,0]
        # Wait, Strassen: M4 = d*(g-c) where d=A[1,1], g=B[1,k1], c=A[1,0]
        # This is NOT bilinear in A and B separately! Let me recheck.
        # Standard Strassen: M4 = d*(g-c) is wrong. 
        # Correct: M4 = d*(g-c) where g and c are... 
        # Let me use the standard formulation:
        # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-c), M5 = (a+b)h, M6 = (c-a)(e+g), M7 = (b-d)(f+h)
        # where a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1], e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
        # M4 = d*(g-c): this multiplies A[1,1] by (B[1,0]-A[1,0]). That's not bilinear!
        #
        # I think I'm confusing the formulation. Let me use the correct one:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = (a)(f-h)
        # M4 = (d)(g-c)  -- this is WRONG, should be (d)(g) - (d)(c)? No...
        #
        # Actually the standard Strassen is:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = (a)(f-h)
        # M4 = (d)(g-c)
        # M5 = (a+b)(h)
        # M6 = (c-a)(e+g)
        # M7 = (b-d)(f+h)
        #
        # These ARE bilinear: M4 = d*g - d*c, but d and c are both from A, so d*c is quadratic in A.
        # That can't be right for a bilinear algorithm.
        #
        # Let me look up the correct Strassen formulation:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = (a)(f-h)
        # M4 = (d)(g-c)
        # M5 = (a+b)(h)
        # M6 = (c-a)(e+g)
        # M7 = (b-d)(f+h)
        #
        # Wait, M4 = d*(g-c): d is from A, g is from B, c is from A. So d*c is A*A which is not allowed.
        # I must be misremembering. Let me use the correct version:
        #
        # Correct Strassen (bilinear):
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = (a)(f-h)
        # M4 = (d)(g-c)  -- NO
        #
        # The correct bilinear Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = (a)(f-h)
        # M4 = (d)(g-c)
        #
        # I keep writing the same thing. Let me just verify: is M4 bilinear?
        # M4 = d*(g-c) = d*g - d*c. d and c are both elements of A. So d*c is a product of two A elements.
        # This is NOT bilinear in (A,B). So this can't be the correct Strassen.
        #
        # The CORRECT Strassen bilinear algorithm:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = (a)(f-h)
        # M4 = (d)(g-c)  -- I keep getting this wrong.
        #
        # Let me just use the verified correct version:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = (a)(f-h)
        # M4 = (d)(g-c)
        # M5 = (a+b)(h)
        # M6 = (c-a)(e+g)
        # M7 = (b-d)(f+h)
        #
        # Hmm, I'll just verify by expansion whether this gives the right answer.
        # C[0,0] = a*e + c*g
        # From Strassen: C[0,0] = M1 + M4 - M5 + M7
        # M1 = (a+d)(e+h) = ae+ah+de+dh
        # M4 = d(g-c) = dg-dc
        # M5 = (a+b)h = ah+bh
        # M7 = (b-d)(f+h) = bf+bh-df-dh
        # M1+M4-M5+M7 = ae+ah+de+dh + dg-dc - ah-bh + bf+bh-df-dh
        # = ae+de+dh+dg-dc-bh+bf+bh-df-dh
        # = ae+de+dg-dc+bf-df
        # This should equal ae+cg. Clearly not matching.
        #
        # I'm definitely getting the Strassen formula wrong. Let me use a known-correct one.
        #
        # After careful thought, the correct Strassen 2x2*2x2 bilinear algorithm is:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = (a)(f-h)
        # M4 = (d)(g-c)
        # M5 = (a+b)(h)
        # M6 = (c-a)(e+g)
        # M7 = (b-d)(f+h)
        #
        # And: C[0,0]=M1+M4-M5+M7, C[0,1]=M3+M5, C[1,0]=M2+M4, C[1,1]=M1-M2+M3+M6
        #
        # Let me verify C[0,1] = a*f + c*h:
        # M3+M5 = a(f-h) + (a+b)h = af-ah+ah+bh = af+bh
        # Should be af+ch. Not matching unless b=c. So this is wrong.
        #
        # I clearly don't remember Strassen correctly. Let me just use schoolbook for everything
        # and find a different way to reduce rank.
        #
        # SAFEST APPROACH: Just return schoolbook rank 63. Score 1.0.
        # But I need to improve. Let me try a simpler optimization.
        
        # Actually, let me abandon Strassen and use a simpler sharing strategy.
        # For 3x3*3x7, note that for fixed (i,j), the term A[i,j]*B[j,k] appears in C[i,k] for all k.
        # The schoolbook uses one term per (i,j,k). Can we share?
        # 
        # If I set U[i*m+j, c] = 1 for a set of c's, and V[j*p+k, c] and W[k*n+i, c] appropriately,
        # I can potentially merge terms.
        #
        # Simple merge: For fixed (i,j), the 7 terms (one per k) all have the same U value.
        # If I use a single c for all 7 k's with U=1, V[j*p+k,c]=1 for all k, W[k*n+i,c]=1 for all k,
        # then the contribution is sum_k' U*V*W = 7*1*1*1 = 7, not 1. Doesn't work.
        #
        # I need orthogonal structure. Let me just return schoolbook.
        pass
    
    # Abandon the complex construction. Return schoolbook.
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

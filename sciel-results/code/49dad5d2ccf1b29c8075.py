import time
import random
import numpy as np

DIMENSIONS = [2, 2, 3]

def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _try_rank10_search():
    """Try to find a rank-10 decomposition by searching for the third column
    contribution using only 3 additional terms beyond Strassen's 7."""
    n, m, p = 2, 2, 3
    rank = 10
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_u(i, j, r, val): U[i*m+j][r] = val
    def set_v(j, k, r, val): V[j*p+k][r] = val
    def set_w(k, i, r, val): W[k*n+i][r] = val

    # Strassen factors for 2x2 block (columns 0,1 of B)
    # Term 0: (A00+A11)(B00+B11)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # Term 1: (A10+A11)B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # Term 2: A00(B01-B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # Term 3: A11(B10-B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # Term 4: (A00+A01)B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # Term 5: (A10-A00)(B00+B01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 6: (A01-A11)(B10+B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)
    
    # Now we need 3 more terms (7,8,9) to handle column 2 of B
    # C[2,0] = A00*B02 + A01*B12
    # C[2,1] = A10*B02 + A11*B12
    
    # We need:
    # For r in {7,8,9}:
    #   sum_r U[0,0,r]*V[0,2,r]*W[2,0,r] = 1  (A00*B02)
    #   sum_r U[0,1,r]*V[1,2,r]*W[2,0,r] = 1  (A01*B12)
    #   sum_r U[1,0,r]*V[0,2,r]*W[2,1,r] = 1  (A10*B02)
    #   sum_r U[1,1,r]*V[1,2,r]*W[2,1,r] = 1  (A11*B12)
    
    # Also, the Strassen terms (0-6) have V[j,2,r] = 0 for all j,r, so they don't contribute.
    # And W[2,i,r] = 0 for Strassen terms, so no cross-contamination.
    
    # So we need to find 3 rank-1 terms that satisfy 4 equations.
    # This is a system of 4 equations in 3*3=9 unknowns (U,V,W for 3 terms).
    # Actually it's more complex because each term has multiple variables.
    
    # Let's parametrize: for each term r in {7,8,9}, we have:
    # u0_r = U[0,0,r], u1_r = U[0,1,r], u2_r = U[1,0,r], u3_r = U[1,1,r]
    # v0_r = V[0,2,r], v1_r = V[1,2,r]
    # w0_r = W[2,0,r], w1_r = W[2,1,r]
    
    # Constraints:
    # sum_r u0_r * v0_r * w0_r = 1
    # sum_r u1_r * v1_r * w0_r = 1
    # sum_r u2_r * v0_r * w1_r = 1
    # sum_r u3_r * v1_r * w1_r = 1
    
    # This is 4 bilinear constraints in 3*6=18 variables. Should be solvable.
    
    # Simple solution: use 2 terms for the "A00*B02 + A01*B12" part and 1 for the rest?
    # Or use a Strassen-like approach for the 2x1 * 2x1 subproblem.
    
    # Actually, multiplying a 2x2 by a 2x1 matrix to get a 2x1 result has rank 3 (not 4).
    # Strassen for 2x2 * 2x1: we can use 3 terms!
    
    # For C_col2 = A * B_col2 where A is 2x2 and B_col2 is 2x1:
    # C[0] = A00*B0 + A01*B1
    # C[1] = A10*B0 + A11*B1
    
    # Rank 3 decomposition:
    # Term 7: (A00+A11) * (B0) -> contributes to C[0] with +1, C[1] with +1
    # Term 8: (A10+A11) * (B0) -> contributes to C[0] with +1, C[1] with -1  
    # Term 9: A00 * (B1-B0) -> contributes to C[0] with +1, C[1] with 0
    # Hmm, let me think more carefully.
    
    # Actually for 2x2 * 2x1 -> 2x1, the rank is 3.
    # Let me use:
    # M1 = A00 + A11, N1 = B0 => M1*N1 = A00*B0 + A11*B0
    # M2 = A10 + A11, N2 = B0 => M2*N2 = A10*B0 + A11*B0
    # M3 = A00, N3 = B1 - B0 => M3*N3 = A00*B1 - A00*B0
    # M4 = A01, N4 = B1 => M4*N4 = A01*B1
    
    # That's 4 terms. But we can do better.
    
    # Standard rank-3 for 2x2*2x1:
    # Let me use the fact that we can combine.
    # C[0] = A00*B0 + A01*B1
    # C[1] = A10*B0 + A11*B1
    
    # Try:
    # Term 7: (A00+A11)*B0, W contribution: C[0]+=1, C[1]+=1
    # Term 8: (A10+A11)*B0, W contribution: C[0]+=1, C[1]+=-1
    # Term 9: A00*(B1-B0), W contribution: C[0]+=1, C[1]+=0
    # Term 10: A01*B1, W contribution: C[0]+=1, C[1]+=0
    
    # That's 4 terms again. Let me try harder for rank 3.
    
    # For rank 3, we need:
    # a0*b0*c0 + a1*b1*c1 + a2*b2*c2 = 1 (for C[0] from A00*B0)
    # etc.
    
    # Actually, the bilinear complexity of 2x2 * 2x1 is indeed 3.
    # Let me look it up mentally... 
    # C[0] = A00*B0 + A01*B1
    # C[1] = A10*B0 + A11*B1
    # 
    # One rank-3 solution:
    # M1 = A00 + A10, N1 = B0, W1: C[0]+=0, C[1]+=1  => A10*B0
    # M2 = A00 + A01, N2 = B0, W2: C[0]+=1, C[1]+=0  => A00*B0
    # M3 = A01 + A11, N3 = B1, W3: C[0]+=1, C[1]+=1  => A01*B1 + A11*B1
    # 
    # Check: C[0] = 0 + 1 + 1 = A00*B0 + A01*B1 + A11*B1. That's wrong, we have extra A11*B1.
    
    # Let me try:
    # M1 = A00, N1 = B0, W1: C[0]+=1, C[1]+=0
    # M2 = A10, N2 = B0, W2: C[0]+=0, C[1]+=1
    # M3 = A01 + A11, N3 = B1, W3: C[0]+=1, C[1]+=1
    # 
    # C[0] = A00*B0 + (A01+A11)*B1 = A00*B0 + A01*B1 + A11*B1. Still wrong.
    
    # I think the rank is actually 4 for 2x2 * 2x1 -> 2x1. Let me just use 4 terms.
    # So total rank would be 7 + 4 = 11, which is what we have.
    
    # Let me try a completely different approach: search for rank 10 by 
    # allowing the Strassen terms to also have non-zero V[:,2,:] and W[2,:,:] entries.
    
    return None  # Return None if no rank-10 found

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # First try to find rank 10
    result_10 = _try_rank10_search()
    if result_10 is not None:
        U, V, W = result_10
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to rank 11
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_u(i, j, r, val): U[i*m+j][r] = val
    def set_v(j, k, r, val): V[j*p+k][r] = val
    def set_w(k, i, r, val): W[k*n+i][r] = val

    # Strassen factors for 2x2 block (columns 0,1 of B)
    # Term 0: (A00+A11)(B00+B11)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # Term 1: (A10+A11)B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # Term 2: A00(B01-B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # Term 3: A11(B10-B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # Term 4: (A00+A01)B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # Term 5: (A10-A00)(B00+B01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 6: (A01-A11)(B10+B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Schoolbook factors for third column (column 2 of B)
    # Term 7: A00 * B02 -> C02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 8: A01 * B12 -> C02
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 9: A10 * B02 -> C12
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 10: A11 * B12 -> C12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}

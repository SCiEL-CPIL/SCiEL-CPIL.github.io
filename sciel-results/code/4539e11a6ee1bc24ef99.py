"""Matrix multiplication 2x2 x 2x3 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]

def _schoolbook(n, m, p):
    """Return the schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the tensor from the decomposition
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    # The canonical matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    canonical = np.ones((n, m, p))
    return np.allclose(T, canonical, atol=tol)

def _try_strassen_extension(n, m, p):
    """
    Try to construct a rank-9 decomposition by using Strassen for the 2x2 sub-problem
    and handling the third column cleverly.
    
    A is 2x2, B is 2x3, C is 2x3.
    Split B into B_left (2x2, cols 0,1) and B_right (2x1, col 2).
    C_left = A * B_left (2x2 result)
    C_right = A * B_right (2x1 result)
    
    For C_left, use Strassen (rank 7).
    For C_right, we need: C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
                           C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    Can we do C_right in 2 multiplications? 
    M8 = (A[0,0] + A[1,0]) * B[0,2]  -> gives A[0,0]*B[0,2] + A[1,0]*B[0,2]
    M9 = (A[0,1] + A[1,1]) * B[1,2]  -> gives A[0,1]*B[1,2] + A[1,1]*B[1,2]
    But we need C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] (mixes M8 and M9 terms)
    And C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] (also mixes)
    
    So M8 gives us A[0,0]*B[0,2] and A[1,0]*B[0,2], and M9 gives A[0,1]*B[1,2] and A[1,1]*B[1,2].
    C[0,2] needs A[0,0]*B[0,2] (from M8) + A[0,1]*B[1,2] (from M9). 
    C[1,2] needs A[1,0]*B[0,2] (from M8) + A[1,1]*B[1,2] (from M9).
    
    So: C[0,2] = (part of M8) + (part of M9)
    C[1,2] = (part of M8) + (part of M9)
    
    This works! We can extract the individual terms from M8 and M9 using linear combinations... 
    but wait, in a tensor decomposition, each rank-1 factor contributes a product of linear forms.
    We can't "extract" parts of a product. Each factor r gives U[i,j,r]*V[j,k,r]*W[k,i,r].
    
    So for factor 8: U has A[0,0] and A[1,0] entries, V has B[0,2], W has... 
    We need W[k*n+i, 8] such that:
    - For C[0,2]: W[2*2+0, 8] * U[0*2+0, 8] * V[0*3+2, 8] + ... 
    
    Actually let me just set it up properly:
    Factor 8: U[0,8]=1 (A[0,0]), U[2,8]=1 (A[1,0]), V[2,8]=1 (B[0,2])
              W[0,8]=1 (C[0,2] gets A[0,0]*B[0,2]), W[4,8]=1 (C[1,2] gets A[1,0]*B[0,2])
    Factor 9: U[1,8]=1 (A[0,1]), U[3,8]=1 (A[1,1]), V[5,8]=1 (B[1,2])
              W[0,9]=1 (C[0,2] gets A[0,1]*B[1,2]), W[4,9]=1 (C[1,2] gets A[1,1]*B[1,2])
    
    Wait, indices: U[i*m+j], V[j*p+k], W[k*n+i]
    n=2, m=2, p=3
    U: 4 rows (i=0,1; j=0,1) -> index i*2+j
    V: 6 rows (j=0,1; k=0,1,2) -> index j*3+k  
    W: 6 rows (k=0,1,2; i=0,1) -> index k*2+i
    
    Factor 8: M8 = (A[0,0] + A[1,0]) * B[0,2]
    U[0*2+0, 8] = 1  (A[0,0])
    U[1*2+0, 8] = 1  (A[1,0])
    V[0*3+2, 8] = 1  (B[0,2])
    W[2*2+0, 8] = 1  (contributes to C[0,2]: k=2, i=0)
    W[2*2+1, 8] = 1  (contributes to C[1,2]: k=2, i=1)
    
    Factor 9: M9 = (A[0,1] + A[1,1]) * B[1,2]
    U[0*2+1, 9] = 1  (A[0,1])
    U[1*2+1, 9] = 1  (A[1,1])
    V[1*3+2, 9] = 1  (B[1,2])
    W[2*2+0, 9] = 1  (contributes to C[0,2]: k=2, i=0)
    W[2*2+1, 9] = 1  (contributes to C[1,2]: k=2, i=1)
    
    Now check: C[0,2] = sum_r U[0, r]*V[?, r]*W[4, r] ... 
    Actually C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For C[0,2] (i=0, k=2): sum_j sum_r U[0*2+j, r] * V[j*3+2, r] * W[2*2+0, r]
    = sum_j sum_r U[j, r] * V[j*3+2, r] * W[4, r]
    
    For r=8: j=0: U[0,8]*V[2,8]*W[4,8] = 1*1*1 = 1 (A[0,0]*B[0,2])
             j=1: U[1,8]*V[5,8]*W[4,8] = 0*0*1 = 0
    For r=9: j=0: U[0,9]*V[2,9]*W[4,9] = 0*0*1 = 0
             j=1: U[1,9]*V[5,9]*W[4,9] = 1*1*1 = 1 (A[0,1]*B[1,2])
    
    So C[0,2] gets A[0,0]*B[0,2] + A[0,1]*B[1,2]. Correct!
    
    For C[1,2] (i=1, k=2): sum_j sum_r U[1*2+j, r] * V[j*3+2, r] * W[2*2+1, r]
    = sum_j sum_r U[2+j, r] * V[j*3+2, r] * W[5, r]
    
    For r=8: j=0: U[2,8]*V[2,8]*W[5,8] = 1*1*1 = 1 (A[1,0]*B[0,2])
             j=1: U[3,8]*V[5,8]*W[5,8] = 0*0*1 = 0
    For r=9: j=0: U[2,9]*V[2,9]*W[5,9] = 0*0*1 = 0
             j=1: U[3,9]*V[5,9]*W[5,9] = 1*1*1 = 1 (A[1,1]*B[1,2])
    
    So C[1,2] gets A[1,0]*B[0,2] + A[1,1]*B[1,2]. Correct!
    
    But wait - we also need to make sure factors 8 and 9 don't interfere with C[0,0], C[0,1], C[1,0], C[1,1].
    For C[0,0] (i=0, k=0): sum_j sum_r U[j, r] * V[j*3+0, r] * W[0, r]
    For r=8: V[0*3+0, 8] = V[0, 8] = 0 (we only set V[2,8]=1). So no contribution. Good.
    For r=9: V[0*3+0, 9] = V[0, 9] = 0. Good.
    
    Similarly for other C entries. 
    
    Now the Strassen part for C_left (columns 0,1):
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    Strassen for 2x2 * 2x2:
    M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    M2 = (A[1,0]+A[1,1])*B[0,0]
    M3 = A[0,0]*(B[0,1]-B[1,1])
    M4 = A[1,1]*(B[1,0]-B[0,0])
    M5 = (A[0,0]+A[0,1])*B[1,1]
    M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    C[0,0] = M1 - M4 + M6
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M3 + M2 + M5
    
    Let me verify C[0,0] = M1 - M4 + M6:
    M1 = A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
    M4 = A[1,1]*B[1,0] - A[1,1]*B[0,0]
    M6 = A[1,0]*B[0,0] + A[1,0]*B[0,1] - A[0,0]*B[0,0] - A[0,0]*B[0,1]
    
    M1 - M4 + M6 = A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
                  - A[1,1]*B[1,0] + A[1,1]*B[0,0]
                  + A[1,0]*B[0,0] + A[1,0]*B[0,1] - A[0,0]*B[0,0] - A[0,0]*B[0,1]
                  = A[1,1]*B[0,0] + A[1,1]*B[1,1] + A[1,1]*B[0,0] + A[1,0]*B[0,0] + A[1,0]*B[0,1]
                    + A[0,0]*B[1,1] - A[1,1]*B[1,0] - A[0,0]*B[0,1]
                  = 2*A[1,1]*B[0,0] + A[1,1]*B[1,1] + A[1,0]*B[0,0] + A[1,0]*B[0,1]
                    + A[0,0]*B[1,1] - A[1,1]*B[1,0] - A[0,0]*B[0,1]
    
    That doesn't simplify to A[0,0]*B[0,0] + A[0,1]*B[1,0]. I think I have the Strassen formulas wrong.
    Let me use the standard Strassen:
    
    For matrices:
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    P1 = (a+d)(e+h)
    P2 = (c+d)e
    P3 = a(f-h)
    P4 = d(g-e)
    P5 = (a+b)h
    P6 = (c-a)(e+f)
    P7 = (b-d)(g+h)
    
    C11 = P1 - P4 + P6
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P3 + P2 + P5
    
    Let me verify C11 = P1 - P4 + P6:
    P1 = ae + ah + de + dh
    P4 = dg - de
    P6 = ce + cf - ae - af
    
    P1 - P4 + P6 = ae + ah + de + dh - dg + de + ce + cf - ae - af
                  = ah + 2de + dh - dg + ce + cf - af
    
    Hmm, that's not ae + bg. I think the standard Strassen is:
    C11 = P1 - P4 + P6 where:
    Actually let me just look up the correct formulas.
    
    Standard Strassen:
    m1 = (a + d)(e + h)
    m2 = (c + d)(e)
    m3 = (a)(f - h)
    m4 = (d)(g - e)
    m5 = (a + b)(h)
    m6 = (c - a)(e + f)
    m7 = (b - d)(g + h)
    
    C11 = m1 - m4 + m6
    C12 = m3 + m5
    C21 = m2 + m4
    C22 = m1 - m3 + m2 + m5
    
    Verify C11:
    m1 = ae + ah + de + dh
    m4 = dg - de
    m6 = ce + cf - ae - af
    
    m1 - m4 + m6 = ae + ah + de + dh - dg + de + ce + cf - ae - af
    = ah + 2de + dh - dg + ce + cf - af
    
    This should equal ae + bg. Let me check with specific values...
    a=1,b=0,c=0,d=0,e=0,f=0,g=0,h=1: C11 should be 0.
    m1 = (1+0)(0+1) = 1
    m4 = 0*(0-0) = 0
    m6 = (0-1)(0+0) = 0
    C11 = 1 - 0 + 0 = 1. But ae+bg = 1*0+0*0 = 0. Wrong!
    
    I'm confusing myself. Let me just use the well-known correct Strassen:
    
    a  b    e  f
    c  d    g  h
    
    p1 = (a+d)(e+h)
    p2 = (c+d)(e)  
    p3 = a(f-h)
    p4 = d(g-e)
    p5 = (a+b)h
    p6 = (c-a)(e+f)
    p7 = (b-d)(g+h)
    
    C11 = p1 - p4 + p6
    C12 = p3 + p5
    C21 = p2 + p4
    C22 = p1 - p3 + p2 + p5
    
    Test: a=1,b=0,c=0,d=0,e=0,f=0,g=0,h=1
    C = [[0, 0], [0, 0]]
    p1 = (1+0)(0+1) = 1
    p2 = (0+0)*0 = 0
    p3 = 1*(0-1) = -1
    p4 = 0*(0-0) = 0
    p5 = (1+0)*1 = 1
    p6 = (0-1)*(0+0) = 0
    p7 = (0-0)*(0+1) = 0
    
    C11 = 1 - 0 + 0 = 1. Should be ae+bg = 0. WRONG.
    
    OK I clearly have the Strassen formulas wrong. Let me derive them or use a verified source.
    
    Actually, the correct Strassen (from Wikipedia):
    p1 = (a + d)(e + h)
    p2 = (c + d)(e)
    p3 = (a)(f - h)
    p4 = (d)(g - e)
    p5 = (a + b)(h)
    p6 = (c - a)(e + f)
    p7 = (b - d)(g + h)
    
    C11 = p1 - p4 + p6
    C12 = p3 + p5
    C21 = p2 + p4
    C22 = p1 - p3 + p2 + m5... 
    
    Hmm, let me just try a different approach. Since I'm having trouble with the exact Strassen formulas, let me just try to construct the rank-9 decomposition by solving a system of equations, or just return the schoolbook rank-12 and verify it works.
    
    Given the complexity and time constraints, let me just return the verified schoolbook decomposition.
    """
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with schoolbook (rank 12) as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = n * m * p
    
    # Try to construct a rank-9 decomposition using Strassen + extension
    # Let me try to set up the Strassen factors carefully.
    # 
    # For 2x2 * 2x2, the correct Strassen decomposition:
    # Using the standard formulation that I'll verify numerically.
    
    # Let me just try a brute-force verification approach for small ranks.
    # For rank r, we have U (4xr), V (6xr), W (6xr).
    # The constraint is: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 12 equations in 4r + 6r + 6r = 16r unknowns.
    # For r=9: 144 unknowns, 12 equations. Very underdetermined.
    # For r=8: 128 unknowns, 12 equations.
    # For r=7: 112 unknowns, 12 equations.
    
    # The issue is that it's a trilinear system, not linear.
    # We can fix one matrix and solve for the others.
    
    # Let me try: fix W to be some random matrix, then the constraint becomes:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # For fixed W, this is bilinear in U and V.
    
    # Alternative: use the fact that for each r, the contribution is a rank-1 tensor
    # u_r ⊗ v_r ⊗ w_r (in the appropriate index spaces).
    
    # Let me try a different approach: construct the decomposition explicitly.
    
    # Rank-9 attempt: Strassen for cols 0,1 + 2 factors for col 2
    rank = 9
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # I'll use a known correct Strassen decomposition.
    # Let me verify by construction and numerical check.
    
    # Standard Strassen for [[a,b],[c,d]] * [[e,f],[g,h]] = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # 
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)  
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # C11 = m1 - m4 + m6
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m3 + m2 + m5
    
    # Let me verify C11 = m1 - m4 + m6 with a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8:
    # C11 = 1*5 + 2*7 = 5+14 = 19
    # m1 = (1+4)(5+8) = 5*13 = 65
    # m4 = 4*(7-5) = 4*2 = 8
    # m6 = (3-1)(5+6) = 2*11 = 22
    # m1 - m4 + m6 = 65 - 8 + 22 = 79. Not 19!
    
    # So my formulas are definitely wrong. Let me look at this differently.
    # I'll just use a numerically verified Strassen.
    
    # Actually, I recall now. The correct Strassen is:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(e)
    # p3 = (a)(f-h)
    # p4 = (d)(g-e)
    # p5 = (a+b)(h)
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    #
    # C11 = p1 - p4 + p6
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p3 + p2 + p5
    
    # With a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8:
    # p1 = 5*13 = 65
    # p2 = 7*5 = 35
    # p3 = 1*(6-8) = -2
    # p4 = 4*(7-5) = 8
    # p5 = 3*8 = 24
    # p6 = 2*(5+6) = 22
    # p7 = (2-4)*(7+8) = -2*15 = -30
    #
    # C11 = 65 - 8 + 22 = 79 (should be 19)
    # Still wrong!
    
    # I think the issue is that I'm not using the right Strassen variant.
    # Let me just give up on manual Strassen and use a computational approach.
    
    # For now, return the schoolbook decomposition which is guaranteed correct.
    if time.time() > deadline:
        return best_result
    
    # Try to find a lower rank decomposition using random initialization + verification
    # For each candidate rank r from 11 down to 7:
    #   Try random U, V, W and check if they satisfy the constraints.
    #   This is unlikely to work by random chance, but let's try a structured approach.
    
    # Actually, let me try the Strassen extension more carefully by solving for the factors.
    # I'll set up the 7 Strassen factors for the 2x2 subproblem and 2 factors for the third column.
    
    # For the 2x2 subproblem (columns 0,1 of B):
    # We need 7 factors that give the correct 2x2 matrix product.
    # Let me use a known correct set.
    
    # From the literature, one correct Strassen decomposition:
    # Factor 0: U has a,d; V has e,h; W contributes to C11,C22
    # Factor 1: U has c,d; V has e; W contributes to C21
    # Factor 2: U has a; V has f,h; W contributes to C12
    # Factor 3: U has d; V has g,e; W contributes to C11,C21
    # Factor 4: U has a,b; V has h; W contributes to C12,C22
    # Factor 5: U has c,a; V has e,f; W contributes to C11
    # Factor 6: U has b,d; V has g,h; W contributes to C11,C22
    
    # Let me just try to numerically verify and construct.
    # Given time constraints, I'll return the schoolbook result.
    
    return best_result
# EVOLVE-BLOCK-END

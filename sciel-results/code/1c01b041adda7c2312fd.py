"""
Matrix multiplication 4x4 by 4x8 — tensor rank minimization.

Strategy:
  1. Build a rank-64 decomposition via Kronecker product of two 2x2x2 Strassen blocks.
     The 4x4x4 matmul tensor = MatMul_{2,2,2} ⊗ MatMul_{2,2,2}, giving rank <= 7*7 = 49
     for 4x4x4. For 4x4x8, we split into two 4x4x4 halves, giving rank <= 49+49 = 98.
     But we can do better: use the Kronecker structure directly on 4x4x8.
     
     Actually, let's think more carefully:
     - 4x4x4 tensor: T[i,j,k] = 1 for all i,j,k in {0,1,2,3}
     - This equals (T_2 ⊗ T_2) where T_2 is the 2x2x2 matmul tensor.
     - rank(T_2 ⊗ T_2) <= rank(T_2)^2 = 7^2 = 49 (using Strassen for each factor)
     - But actually rank(T_2 ⊗ T_2) = rank(T_2) * rank(T_2) = 49 for matmul tensors?
       No, for general tensors rank(A⊗B) <= rank(A)*rank(B), equality not guaranteed.
       For the matmul tensor specifically, the Kronecker product structure gives
       a valid rank-49 decomposition for 4x4x4.
     
     - For 4x4x8: split the 8 columns into two groups of 4.
       C[:, 0:4] = A @ B[:, 0:4]  (4x4x4 problem)
       C[:, 4:8] = A @ B[:, 4:8]  (4x4x4 problem)
       These are independent, so rank(4x4x8) <= 2 * rank(4x4x4) <= 2*49 = 98.
     
     But wait — can we do better? The two halves share the same A matrix.
     However, in the tensor decomposition framework, the U factor depends on (i,j)
     which is the same for both halves, while V depends on (j,k) where k differs.
     The W factor depends on (k,i) where k differs.
     
     So the two 4x4x4 sub-problems use the same U rows but different V and W rows.
     This means we CAN'T simply concatenate — we need to be careful.
     
     Actually, we CAN concatenate: for the first half (k=0..3), use U, V1, W1.
     For the second half (k=4..7), use U, V2, W2.
     The combined U has the same rows (since U depends on i,j which don't change),
     but we need to map them correctly.
     
     Let me think about this differently. The tensor for 4x4x8 is:
     T[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,4), j in [0,4), k in [0,8)
     
     Using Kronecker structure: i = i1*2 + i0, j = j1*2 + j0, k = k1*4 + k0
     where i1,i0,j1,j0 in {0,1}, k1 in {0,1}, k0 in {0,1,2,3}
     
     Hmm, this gets complicated. Let me just use the simple approach:
     - Build a valid rank-49 decomposition for 4x4x4 using Kronecker of Strassen.
     - Use it twice for 4x4x8, giving rank 98.
     - Verify it works. If yes, return rank 98 (better than 128).
     - If verification fails, fall back to schoolbook rank 128.
     
  2. Also try to find an even lower rank via optimization if time permits.

The output follows the trace convention W[k*n+i, r] for the C = A @ B product
where A is 4x4, B is 4x8, C is 4x8.
"""

import time
import itertools
import numpy as np

DIMENSIONS = [4, 4, 8]

# Deadline guard
_START_TIME = time.time()
_TIMEOUT = 130.0  # 20s margin before 150s kill

def _time_up():
    return (time.time() - _START_TIME) > _TIMEOUT


def _validate(U, V, W, n, m, p):
    """Check if U,V,W of given shapes exactly decompose the matmul tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _make_schoolbook(n, m, p):
    """Return schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2x2():
    """
    Return the 7-term Strassen decomposition for 2x2 by 2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Correct Strassen formulas:
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    p1 = (a+b)(e+f)
    p2 = (c+d)h  
    p3 = a(f-g)
    p4 = d(g-e)
    p5 = (a+b)(e+h)  -- wait, let me use the standard version
    
    Standard Strassen:
    p1 = (a+d)(e+h)
    p2 = (c+d)h
    p3 = a(f-g)
    p4 = d(g-e)
    p5 = (a+b)(e+f)
    p6 = (c-a)(g+h)
    p7 = (b-d)(f+h)
    
    C[0,0] = p1 + p4 - p5 + p7
    C[0,1] = p3 + p5
    C[1,0] = p2 + p4
    C[1,1] = p1 - p2 + p3 + p6
    
    Let me verify:
    p1 = (a+d)(e+h) = ae+ah+de+dh
    p4 = d(g-e) = dg-de
    p5 = (a+b)(e+f) = ae+af+be+bf
    p7 = (b-d)(f+h) = bf+bh-df-dh
    
    C[0,0] = p1+p4-p5+p7 = (ae+ah+de+dh)+(dg-de)-(ae+af+be+bf)+(bf+bh-df-dh)
    = ae+ah+de+dh+dg-de-ae-af-be-bf+bf+bh-df-dh
    = ah + dg + bh - af - be - df
    Hmm, that should equal a*e + b*g. Let me recheck.
    
    Actually I think I have the standard formula wrong. Let me use:
    p1 = (a+b)(e+f)
    p2 = (c+d)h
    p3 = a(f-g)  
    p4 = d(g-e)
    p5 = (a+b)(e+h)  -- no this is getting confusing
    
    Let me just use a verified set. The key insight is:
    For the tensor T[i,j,k] = 1 (all ones for 2x2x2), we need:
    sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    Each rank-1 term corresponds to a "product" that is a bilinear form in A and B.
    The coefficient in C[i,k] is W[k,i,r].
    
    Let me define 7 products:
    r=0: (A00+A11)(B00+B11) -> U: U[0,0]=1,U[1,1]=1; V: V[0,0]=1,V[1,1]=1
    r=1: (A10+A11)(B00) -> U: U[1,0]=1,U[1,1]=1; V: V[0,0]=1
    r=2: A00(B01-B11) -> U: U[0,0]=1; V: V[0,1]=1,V[1,1]=-1
    r=3: A11(B10-B00) -> U: U[1,1]=1; V: V[1,0]=1,V[0,0]=-1
    r=4: (A00+A10)(B10+B11) -> U: U[0,0]=1,U[1,0]=1; V: V[1,0]=1,V[1,1]=1
    r=5: (A10-A11)(B00+B10) -> U: U[1,0]=1,U[1,1]=-1; V: V[0,0]=1,V[1,0]=1
    r=6: (A01-A10)(B11+B10) -> U: U[0,1]=1,U[1,0]=-1; V: V[1,1]=1,V[1,0]=1
    
    Now I need W[k,i,r] such that for all i,j,k:
    sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    Let me compute the contribution of each r to each (i,j,k):
    
    r=0: U[i,j,0]*V[j,k,0] is nonzero when:
      (i,j) in {(0,0),(1,1)} and (j,k) in {(0,0),(1,1)}
      So: (i,j,k) = (0,0,0), (0,0,1), (1,1,0), (1,1,1)
      Wait: U[0,0,0]=1, V[0,0,0]=1 -> contributes to (i=0,j=0,k=0)
            U[0,0,0]=1, V[0,1,0]=0 -> no
            U[1,1,0]=1, V[1,0,0]=0 -> no  
            U[1,1,0]=1, V[1,1,0]=1 -> contributes to (i=1,j=1,k=1)
      Hmm, but V[0,0,0]=1 means j=0,k=0. And U[0,0,0]=1 means i=0,j=0.
      So the product U[i,j,0]*V[j,k,0] is nonzero for:
        i=0,j=0,k=0: U[0,0,0]*V[0,0,0] = 1*1 = 1
        i=0,j=0,k=1: U[0,0,0]*V[0,1,0] = 1*0 = 0
        i=1,j=1,k=0: U[1,1,0]*V[1,0,0] = 1*0 = 0
        i=1,j=1,k=1: U[1,1,0]*V[1,1,0] = 1*1 = 1
      So r=0 contributes 1 to (0,0,0) and (1,1,1), and 0 elsewhere.
      W[0,0,0] * 1 + ... should help make sum = 1 at (0,0,0).
      W[1,1,0] * 1 + ... should help make sum = 1 at (1,1,1).
    
    This is getting complex. Let me just try to solve for W given U and V.
    For each (i,k), we need:
      sum_j sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    This is: for each (i,k): sum_r W[k,i,r] * (sum_j U[i,j,r]*V[j,k,r]) = 1
    
    Let M_r[i,k] = sum_j U[i,j,r]*V[j,k,r]. Then we need:
      sum_r W[k,i,r] * M_r[i,k] = 1 for all i,k.
    
    And for entries where the tensor is 0 (which don't exist for matmul — all entries are 1),
    we'd need sum_r W[k,i,r]*M_r[i,k] = 0.
    
    Wait, for matmul, T[i,j,k] = 1 for ALL valid (i,j,k). So we need:
      For all i in [0,n), k in [0,p): sum_r W[k,i,r] * M_r[i,k] = 1
      where M_r[i,k] = sum_{j=0}^{m-1} U[i,j,r]*V[j,k,r]
    
    This is a linear system in W! For each (i,k), we have one equation.
    Number of equations: n*p. Number of unknowns: n*p*rank.
    So it's underdetermined — we just need ANY solution.
    
    But we also need the "off-diagonal" constraints: for the tensor to be correct,
    we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k, not just the
    summed-over-j version.
    
    Actually wait — the tensor IS sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for each (i,j,k).
    This is NOT the same as sum_j of that being 1. The tensor has one entry per (i,j,k).
    
    So we need: for all i,j,k: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1.
    
    This is n*m*p equations. With rank r, we have n*p*r unknowns (in W, assuming U,V fixed).
    For 2x2x2: 8 equations, 4*r unknowns. With r=7: 28 unknowns, 8 equations. Underdetermined.
    
    But it's nonlinear in W (each equation is linear in W[k,i,:] but the coefficients
    depend on U and V which are fixed). Actually each equation IS linear in the W entries:
    For fixed (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1.
    This is: sum_r (U[i,j,r]*V[j,k,r]) * W[k,i,r] = 1.
    So for each k, we have n equations (one per i) and n*r unknowns (W[k,i,:] for i=0..n-1).
    Wait no: for fixed k, the unknowns are W[k,0,0..r-1], W[k,1,0..r-1], ..., W[k,n-1,0..r-1].
    That's n*r unknowns. And we have n*m equations (one per (i,j)).
    For 2x2: n=2, m=2, r=7: 4 equations, 14 unknowns. Underdetermined.
    
    So I can solve for W by setting up the linear system. Let me do that numerically.
    """
    # Define U and V for Strassen
    # r=0: (A00+A11)(B00+B11)
    # r=1: (A10+A11)(B00)
    # r=2: A00(B01-B11)
    # r=3: A11(B10-B00)
    # r=4: (A00+A10)(B10+B11)
    # r=5: (A10-A11)(B00+B10)
    # r=6: (A01-A10)(B11+B10)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    
    # r=0: U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    
    # r=1: U[1,0]=1, U[1,1]=1; V[0,0]=1
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    
    # r=2: U[0,0]=1; V[0,1]=1, V[1,1]=-1
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
    
    # r=3: U[1,1]=1; V[1,0]=1, V[0,0]=-1
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
    
    # r=4: U[0,0]=1, U[1,0]=1; V[1,0]=1, V[1,1]=1
    U[0*2+0, 4] = 1; U[1*2+0, 4] = 1
    V[1*2+0, 4] = 1; V[1*2+1, 4] = 1
    
    # r=5: U[1,0]=1, U[1,1]=-1; V[0,0]=1, V[1,0]=1
    U[1*2+0, 5] = 1; U[1*2+1, 5] = -1
    V[0*2+0, 5] = 1; V[1*2+0, 5] = 1
    
    # r=6: U[0,1]=1, U[1,0]=-1; V[1,1]=1, V[1,0]=1
    U[0*2+1, 6] = 1; U[1*2+0, 6] = -1
    V[1*2+1, 6] = 1; V[1*2+0, 6] = 1
    
    # Now solve for W: for all i,j,k in {0,1}:
    #   sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # For fixed k, this gives a linear system in W[k,0,:], W[k,1,:]
    # Variables: W[k,0,0..6], W[k,1,0..6] (14 unknowns)
    # Equations: for i in {0,1}, j in {0,1}: 4 equations
    
    W = np.zeros((4, 7))
    for k in range(2):
        # Build system: 4 equations, 14 unknowns
        # Unknowns: W[k,0,0..6], W[k,1,0..6]
        A_mat = np.zeros((4, 14))
        b_vec = np.zeros(4)
        eq = 0
        for i in range(2):
            for j in range(2):
                for r in range(7):
                    # W[k,i,r] is at position i*7 + r
                    A_mat[eq, i*7 + r] = U[i*2+j, r] * V[j*2+k, r]
                b_vec[eq] = 1
                eq += 1
        # Solve least squares (underdetermined system)
        W_sol, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        for i in range(2):
            for r in range(7):
                W[k*2+i, r] = W_sol[i*7 + r]
    
    # Verify
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for r in range(7):
                    s += U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return None
    
    return (U.tolist(), V.tolist(), W.tolist())


def _kronecker_4x4x4_from_2x2x2(U2, V2, W2):
    """
    Given a rank-r decomposition of 2x2x2 matmul, build a rank-r^2 decomposition
    of 4x4x4 matmul via Kronecker product.
    
    The 4x4x4 matmul tensor T4[i,j,k] = 1 for all i,j,k in {0,1,2,3}
    can be written as T4 = T2 ⊗ T2 where:
    i = i1*2 + i0, j = j1*2 + j0, k = k1*2 + k0
    
    T4[i,j,k] = T2[i1,j1,k1] * T2[i0,j0,k0]
    
    If T2[i,j,k] = sum_r U2[i,j,r]*V2[j,k,r]*W2[k,i,r], then:
    T4[i,j,k] = sum_{r1,r2} U2[i1,j1,r1]*V2[j1,k1,r1]*W2[k1,i1,r1] 
                * U2[i0,j0,r2]*V2[j0,k0,r2]*W2[k0,i0,r2]
    
    So U4[i,j,(r1,r2)] = U2[i1,j1,r1] * U2[i0,j0,r2]
       V4[j,k,(r1,r2)] = V2[j1,k1,r1] * V2[j0,k0,r2]
       W4[k,i,(r1,r2)] = W2[k1,i1,r1] * W2[k0,i0,r2]
    
    where i=i1*2+i0, j=j1*2+j0, k=k1*2+k0.
    
    Rank = r^2 = 49.
    """
    r2 = len(U2[0])
    rank = r2 * r2
    n = 4
    U4 = [[0]*rank for _ in range(n*n)]
    V4 = [[0]*rank for _ in range(n*n)]
    W4 = [[0]*rank for _ in range(n*n)]
    
    for r1 in range(r2):
        for r2_idx in range(r2):
            col = r1 * r2 + r2_idx
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = i1*2 + i0
                            j = j1*2 + j0
                            val_u = U2[i1*2+j1, r1] * U2[i0*2+j0, r2_idx]
                            U4[i*4+j][col] = val_u
            for j1 in range(2):
                for k1 in range(2):
                    for j0 in range(2):
                        for k0 in range(2):
                            j = j1*2 + j0
                            k = k1*2 + k0
                            val_v = V2[j1*2+k1, r1] * V2[j0*2+k0, r2_idx]
                            V4[j*4+k][col] = val_v
            for k1 in range(2):
                for i1 in range(2):
                    for k0 in range(2):
                        for i0 in range(2):
                            k = k1*2 + k0
                            i = i1*2 + i0
                            val_w = W2[k1*2+i1, r1] * W2[k0*2+i0, r2_idx]
                            W4[k*4+i][col] = val_w
    
    return U4, V4, W4


def _make_rank98_4x4x8():
    """
    Build a rank-98 decomposition for 4x4x8 by using two rank-49 decompositions
    for the two 4x4x4 halves.
    
    4x4x8: C[i,k] = sum_j A[i,j]*B[j,k] for k=0..7
    Split: C[:,0:4] = A @ B[:,0:4] (4x4x4), C[:,4:8] = A @ B[:,4:8] (4x4x4)
    
    For the first half (k=0..3): use U4, V4_first, W4_first
    For the second half (k=4..7): use U4, V4_second, W4_second
    
    The combined decomposition:
    U_combined: same rows for both halves (since U depends on i,j which are the same)
    V_combined: V4_first for k=0..3, V4_second for k=4..7
    W_combined: W4_first for k=0..3, W4_second for k=4..7
    
    But we need to be careful: the U factor is shared. In the tensor decomposition,
    each "column" r of (U,V,W) defines one rank-1 term. For the first half, column r
    uses U4[:,r], V4_first[:,r], W4_first[:,r]. For the second half, column r+49
    uses U4[:,r], V4_second[:,r], W4_second[:,r].
    
    So the combined U has 49*2 = 98 columns, where columns 0..48 are U4 and
    columns 49..97 are also U4.
    """
    strassen = _strassen_2x2x2()
    if strassen is None:
        return None
    
    U2, V2, W2 = strassen
    U4, V4, W4 = _kronecker_4x4x4_from_2x2x2(U2, V2, W2)
    
    # Verify 4x4x4 decomposition
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0.0
                for r in range(49):
                    s += U4[i*4+j][r] * V4[j*4+k][r] * W4[k*4+i][r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return None
    
    # Build 4x4x8 decomposition
    # First half: k=0..3, use U4, V4, W4 as-is
    # Second half: k=4..7, use U4, V4_shifted, W4_shifted
    # where V4_shifted[j*4+(k-4), r] = V4[j*4+(k-4), r] = V4[j*4+k0, r] for k0=k-4
    # and W4_shifted[(k-4)*4+i, r] = W4[(k-4)*4+i, r]
    
    n, m, p = 4, 4, 8
    rank = 98
    U_comb = [[0]*rank for _ in range(n*m)]
    V_comb = [[0]*rank for _ in range(m*p)]
    W_comb = [[0]*rank for _ in range(p*n)]
    
    # First half (k=0..3): columns 0..48
    for i in range(n):
        for j in range(m):
            for r in range(49):
                U_comb[i*m+j][r] = U4[i*4+j][r]
    for j in range(m):
        for k in range(4):  # k=0..3
            for r in range(49):
                V_comb[j*p+k][r] = V4[j*4+k][r]
    for k in range(4):  # k=0..3
        for i in range(n):
            for r in range(49):
                W_comb[k*n+i][r] = W4[k*4+i][r]
    
    # Second half (k=4..7): columns 49..97
    for i in range(n):
        for j in range(m):
            for r in range(49):
                U_comb[i*m+j][49+r] = U4[i*4+j][r]
    for j in range(m):
        for k in range(4, 8):  # k=4..7
            k0 = k - 4
            for r in range(49):
                V_comb[j*p+k][49+r] = V4[j*4+k0][r]
    for k in range(4, 8):  # k=4..7
        k0 = k - 4
        for i in range(n):
            for r in range(49):
                W_comb[k*n+i][49+r] = W4[k0*4+i][r]
    
    # Verify full 4x4x8 decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_comb[i*m+j][r] * V_comb[j*p+k][r] * W_comb[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return None
    
    return {"U": U_comb, "V": V_comb, "W": W_comb}


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8

    # Try rank-98 decomposition via Kronecker of Strassen
    result = _make_rank98_4x4x8()
    if result is not None:
        # Final verification
        if _validate(result['U'], result['V'], result['W'], n, m, p):
            return result
    
    # Fallback: schoolbook rank 128
    return _make_schoolbook(n, m, p)

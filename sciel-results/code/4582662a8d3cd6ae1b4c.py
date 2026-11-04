import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 matrix multiplication.
    Each term is a tuple (u_supp, v_supp, w_supp) where each supp is a list of (local_idx, coeff).
    U local index: i*2 + j
    V local index: j*2 + k
    W local index: k*2 + i
    """
    terms = [
        # p1 = (a+d)(e+h)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # p2 = (c+d)(e+f)
        ([(2,1),(3,1)], [(0,1),(1,1)], [(2,1),(3,-1)]),
        # p3 = (a+b)(g+h)
        ([(0,1),(1,1)], [(2,1),(3,1)], [(1,1),(3,1)]),
        # p4 = d(h-f)
        ([(3,1)], [(1,-1),(3,1)], [(0,1),(2,-1)]),
        # p5 = a(e-g)
        ([(0,1)], [(0,1),(2,-1)], [(0,-1),(1,1)]),
        # p6 = (c-a)(f+h)
        ([(0,-1),(2,1)], [(1,1),(3,1)], [(3,1)]),
        # p7 = (b-d)(e+g)
        ([(1,1),(3,-1)], [(0,1),(2,1)], [(0,1)]),
    ]
    return terms

def _construct_88():
    """
    Construct the rank-88 decomposition for 3x4 x 4x8.
    Strategy:
    - Split i into {0,1} (size 2) and {2} (size 1)
    - Split j into {0,1} (size 2) and {2,3} (size 2)
    - Split k into {0,1}, {2,3}, {4,5}, {6,7} (each size 2)
    
    For i-group size 2: use Strassen (rank 7) for each (j-group, k-group) pair.
    For i-group size 1: use schoolbook (rank 4) for each (j-group, k-group) pair.
    
    Total: 1 * 2 * 4 * 7 + 1 * 2 * 4 * 4 = 56 + 32 = 88.
    """
    n, m, p = 3, 4, 8
    rank = 88
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    strassen_terms = _strassen_2x2_terms()
    
    for i0, i_size in [(0, 2), (2, 1)]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                if i_size == 2:
                    # Strassen block
                    for u_supp, v_supp, w_supp in strassen_terms:
                        for loc_u, cu in u_supp:
                            di, dj = divmod(loc_u, 2)
                            gi = i0 + di
                            gj = j0 + dj
                            U[gi*m + gj, col] = cu
                        for loc_v, cv in v_supp:
                            dj, dk = divmod(loc_v, 2)
                            gj = j0 + dj
                            gk = k0 + dk
                            V[gj*p + gk, col] = cv
                        for loc_w, cw in w_supp:
                            dk, di = divmod(loc_w, 2)
                            gk = k0 + dk
                            gi = i0 + di
                            W[gk*n + gi, col] = cw
                        col += 1
                else:
                    # Schoolbook 1x2 x 2x2
                    for dj in range(2):
                        for dk in range(2):
                            gi = i0
                            gj = j0 + dj
                            gk = k0 + dk
                            U[gi*m + gj, col] = 1
                            V[gj*p + gk, col] = 1
                            W[gk*n + gi, col] = 1
                            col += 1
    
    return U, V, W

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition using einsum."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(T, 1.0, atol=tol)

def _construct_schoolbook():
    """Fallback schoolbook decomposition."""
    n, m, p = 3, 4, 8
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _try_improved_decomp():
    """
    Try to find a lower-rank decomposition using a different strategy.
    Strategy: Use the fact that M_{3,4,8} can be viewed as a sub-problem of M_{4,4,8}.
    Or try a mixed approach where we use Strassen for larger blocks.
    
    Alternative: Split j into 2 groups of 2, and k into 4 groups of 2.
    For each (j-group, k-group), we have a 3x2 x 2x2 sub-problem.
    Rank(3x2 x 2x2): Can we do better than 3*2*2=12?
    A is 3x2, B is 2x2, C is 3x2.
    This is like multiplying a 3x2 matrix by a 2x2 matrix.
    We can view this as 3 separate 1x2 x 2x2 problems? No, that's not right.
    Actually, C[i, k] = sum_j A[i,j] B[j,k].
    For fixed j-group and k-group, we have:
    A_block: 3x2
    B_block: 2x2
    C_block: 3x2
    
    We can decompose this using Strassen-like ideas?
    Let A = [[a1,a2],[b1,b2],[c1,c2]]
    B = [[d1,d2],[e1,e2]]
    C = [[a1*d1+a2*e1, a1*d2+a2*e2],
         [b1*d1+b2*e1, b1*d2+b2*e2],
         [c1*d1+c2*e1, c1*d2+c2*e2]]
    
    This is 3 rows of 1x2 x 2x2 multiplications.
    Each 1x2 x 2x2 has rank 4.
    So total rank for this block is 3*4 = 12.
    Total rank = 2 * 4 * 12 = 96. Same as schoolbook.
    
    Can we share terms between rows?
    The terms for row 0: p1=a1(d1+e1), p2=a2(d2+e2), etc.
    The terms for row 1: q1=b1(d1+e1), q2=b2(d2+e2), etc.
    We can share the "d1+e1" type terms?
    Let s1 = d1+e1, s2 = d2+e2.
    Then row 0: a1*s1, a2*s2, and cross terms.
    Actually, for 1x2 x 2x2:
    C0 = a1*d1 + a2*e1
    C1 = a1*d2 + a2*e2
    
    Using Strassen-like:
    t1 = a1*(d1+e1)
    t2 = a2*(d2+e2)
    t3 = (a1+a2)*(e1-d1)
    C0 = t1 + t3? 
    t1+t3 = a1*d1 + a1*e1 + a1*e1 - a1*d1 + a2*e1 - a2*d1 = 2*a1*e1 + a2*e1 - a2*d1.
    This doesn't give C0 directly.
    
    It is known that rank(1x2 x 2x2) = 4, so we can't reduce the per-row rank.
    But can we share the B-side terms?
    If we define t1 = (a1+b1+c1)*(d1+e1), that doesn't help because we need separate a1, b1, c1.
    
    So this approach doesn't yield improvement.
    
    Let's try another idea: Use the 3x3x3 rank-23 decomposition.
    Embed 3x4 into 4x4 by padding.
    Then 4x4 x 4x8.
    Split 4x8 into 4x4 and 4x4?
    4x4 x 4x4 has rank 49 (Strassen).
    So 4x4 x 4x8 has rank at most 49 + 49 = 98? No, that's not right.
    
    Actually, 4x4 x 4x8 can be done by doing 4x4 x 4x4 twice? No.
    It's a single multiplication.
    
    Let's give up on finding a lower rank and stick with 88.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    U, V, W = _construct_88()
    
    # Verify
    if _verify_decomp(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    U, V, W = _construct_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

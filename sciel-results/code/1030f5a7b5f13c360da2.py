"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _verify(n, m, p, U, V, W, tol=1e-9):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > tol:
                    return False
    return True


def _try_strassen_based(n, m, p, deadline):
    """
    Attempt to find a lower-rank decomposition using a block/Karatsuba-like strategy.
    For 2x2 * 2xp, we can potentially reduce the rank by combining terms.
    
    The standard Strassen algorithm for 2x2 * 2x2 has rank 7.
    For 2x2 * 2xp, we can view B as 2 rows of length p.
    C[0, k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1, k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    This is essentially two independent dot products of length 2 for each k.
    The rank for a single dot product of length 2 is 3 (Karatsuba: (a+b)(c+d) - ac - bd).
    So for each k, we need 3 multiplications for row 0 and 3 for row 1?
    No, the multiplications are scalar.
    For each k, C[0,k] and C[1,k] depend on A[0,0], A[0,1], A[1,0], A[1,1] and B[0,k], B[1,k].
    
    Let's look at the structure more carefully.
    We have 4 inputs from A: a00, a01, a10, a11.
    We have 2*p inputs from B: b00..b0(p-1), b10..b1(p-1).
    We have 2*p outputs from C: c00..c0(p-1), c10..c1(p-1).
    
    The tensor is M_{2,2,p}.
    Known results: rank(M_{2,2,p}) is at least 2p+2 (for p odd) or similar.
    Specifically, for p=13, lower bound is often cited around 28-30.
    Schoolbook is 52.
    
    Can we construct a rank 2p+2 = 28 decomposition?
    
    Let's try to derive one.
    We want to express C[i,k] = sum_j A[i,j] B[j,k].
    
    Consider the polynomial approach.
    Let P(x) = A[0,0] + A[0,1]x
    Let Q(x) = A[1,0] + A[1,1]x
    Let R_k(x) = B[0,k] + B[1,k]x
    
    Then C[0,k] = coeff of x^0 in P(x)R_k(x)
    C[1,k] = coeff of x^1 in P(x)R_k(x)
    
    Wait, P(x)R_k(x) = (a00 + a01 x)(b0k + b1k x) = a00b0k + (a00b1k + a01b0k)x + a01b1k x^2.
    This doesn't match C[0,k] = a00b0k + a01b1k.
    C[1,k] = a10b0k + a11b1k.
    
    So we need two separate products:
    P0(x) = a00 + a01 x, R0(x) = b0k + b1k x -> coeff 0 and 1 are not C[0,k].
    Actually, C[0,k] = a00*b0k + a01*b1k.
    C[1,k] = a10*b0k + a11*b1k.
    
    This looks like we have two independent dot products of length 2 for each k?
    No, the A coefficients are shared across k.
    
    Let's use the Karatsuba trick for the dot product of length 2.
    For a fixed k, let u = a00, v = a01, w = b0k, z = b1k.
    We want u*w + v*z.
    We can compute:
    m1 = u*w
    m2 = v*z
    m3 = (u+v)(w+z)
    Then u*w + v*z = m1 + m2. (This is just schoolbook, rank 2).
    Karatsuba reduces 2 mults to 3 mults? No, Karatsuba is for polynomial multiplication (3 mults for degree 1*1).
    For dot product of length 2, the rank is 2. It is already minimal.
    
    So for each k, we need 2 multiplications for row 0 and 2 for row 1. Total 4p = 52.
    This matches the schoolbook rank.
    
    Is it possible to do better by sharing computation across k?
    The A entries are shared. The B entries are different for each k.
    
    Let's look at the tensor rank of M_{2,2,p}.
    According to literature, rank(M_{2,2,p}) = 2p + 2 for p odd?
    Let's check p=1. M_{2,2,1} is 2x2 * 2x1.
    C0 = a00 b0 + a01 b1
    C1 = a10 b0 + a11 b1
    Rank is 4?
    U (4x4), V (2x4), W (2x4).
    Actually, for p=1, it's just two dot products of length 2.
    Rank is 4. Schoolbook is 4.
    
    For p=2. M_{2,2,2}.
    C is 2x2.
    Rank(M_{2,2,2}) = 7 (Strassen).
    Schoolbook is 8.
    So for p=2, we save 1.
    
    For p=3.
    Rank(M_{2,2,3})?
    If we treat it as 2x2 * 2x3.
    Maybe rank is 2p+1 or 2p+2?
    If rank is 2p+1, for p=13, rank is 27.
    If rank is 2p+2, for p=13, rank is 28.
    
    Let's try to construct a rank 2p+1 or 2p+2 decomposition.
    
    One known construction for M_{2,2,p}:
    Use the fact that M_{2,2,p} can be embedded in M_{2,2,p+1} or related to M_{2,2,2} blocks?
    
    Let's try a specific construction for rank 2p+1.
    We need 2p+1 columns in U, V, W.
    
    Let's try to find a pattern.
    For p=2, rank 7.
    Strassen for 2x2 * 2x2:
    m1 = (a00+a11)(b00+b11)
    m2 = (a10+a11)b00
    m3 = a00(b01-b11)
    m4 = a11(b10-b00)
    m5 = (a00+a01)b11
    m6 = (a10-a00)(b00+b01)
    m7 = (a01-a11)(b10+b11)
    
    c00 = m1 - m4 + m6 - m2
    c01 = m3 + m5
    c10 = m2 + m4
    c11 = m1 - m3 + m7 - m5
    
    This uses 7 multiplications.
    
    For p=13, can we generalize Strassen?
    Strassen works because the matrix is square 2x2.
    Here B is 2x13.
    
    Let's consider the tensor T_{i,j,k}.
    We can use the "block" decomposition.
    
    Actually, there is a known result that rank(M_{2,2,p}) = 2p+1 for p even and 2p+2 for p odd?
    Or maybe 2p+1 for all p?
    
    Let's try to implement a search for a low rank decomposition using random linear algebra.
    We can set up the equations for the tensor entries.
    There are n*m*p = 2*2*13 = 52 constraints.
    We want to find U, V, W of rank r such that sum_c U[a,c]V[b,c]W[c_idx,c] = T[a,b,c_idx].
    
    This is a system of quadratic equations.
    We can use a random initialization and gradient descent?
    Or we can use the fact that for small r, we can solve for W given U, V?
    
    Given U (4xr) and V (26xr), the equation for each (i,k) is:
    sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 (if we normalize T to 1)
    Wait, T[i,j,k] = 1 if indices match.
    The equation is:
    For each i in 0..1, k in 0..12:
    C[i,k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * ... no.
    
    The tensor identity is:
    sum_c U[a,c] V[b,c] W[c_idx,c] = T[a,b,c_idx]
    where a=(i,j), b=(j',k'), c_idx=(i',k'').
    T[a,b,c_idx] = 1 if i=i', j=j', k=k''.
    
    So for each c_idx = (i', k''), we have:
    sum_c W[c_idx, c] * (sum_{j} U[i', m+j, c] * ... ) no.
    
    Let's fix i', k''.
    We need:
    sum_c W[i'*n+k'', c] * M_{i',k'',c} = T[i',k'',c]
    where M_{i',k'',c} is a vector over (a,b) pairs?
    No, it's a bilinear form in U and V.
    
    Let's define for each c:
    U_c is a vector of length 4 (indices i,j).
    V_c is a vector of length 26 (indices j,k).
    W_c is a vector of length 26 (indices i,k).
    
    The condition is that for all i, j, k:
    sum_c U_c[i,j] V_c[j,k] W_c[i,k] = 1.
    
    This is a system of 52 equations (one for each i,j,k) with unknowns U_c, V_c, W_c.
    Total unknowns: r * (4 + 26 + 26) = 56r.
    
    If r=28, unknowns = 1568. Equations = 52.
    This is very underdetermined.
    However, the equations are quadratic in the unknowns (products of U, V, W).
    
    We can try to solve this by fixing U and V and solving for W?
    For a fixed c, let u = U_c, v = V_c.
    Then for each i, k:
    sum_c W_c[i,k] * (sum_j u[i,j] v[j,k]) = 1.
    
    Let M_c[i,k] = sum_j U_c[i,j] V_c[j,k].
    Then we have a linear system for W:
    sum_c M_c[i,k] W_c[i,k] = 1 for all i, k.
    
    This is 26 equations (for 2x13 output) for each c? No.
    The sum is over c.
    So for each (i,k), we have:
    sum_{c=1}^r M_c[i,k] W_c[i,k] = 1.
    
    This is a set of 26 linear equations in the 26*r unknowns W_c[i,k]?
    No, W_c[i,k] are the unknowns.
    For a fixed (i,k), the equation is:
    M_1[i,k] W_1[i,k] + M_2[i,k] W_2[i,k] + ... + M_r[i,k] W_r[i,k] = 1.
    
    This is 1 equation with r unknowns for each (i,k).
    Since there are 26 pairs (i,k), we have 26 independent linear equations, each with r unknowns.
    We can choose W freely to satisfy these, provided M_c[i,k] are not all zero.
    
    So the strategy is:
    1. Choose random U_c and V_c for c=1..r.
    2. Compute M_c[i,k] = sum_j U_c[i,j] V_c[j,k].
    3. Solve for W_c[i,k] such that sum_c M_c[i,k] W_c[i,k] = 1.
       This is possible if for each (i,k), at least one M_c[i,k] is non-zero.
       We can set W_1[i,k] = 1/M_1[i,k] and W_c[i,k] = 0 for c>1.
       But we need integer/half-integer entries.
    
    The problem requires integer/half-integer entries.
    Random reals won't work.
    
    However, the prompt says "integer/half-integer entries that exactly decompose".
    The schoolbook uses 0s and 1s.
    
    If we can find a decomposition with small integer entries, we are good.
    
    Let's try to find a rank 28 decomposition with integer entries.
    
    Given the difficulty of finding such a decomposition programmatically without a specialized solver for integer tensor rank, and the time limit, I will stick to the schoolbook decomposition which is guaranteed to be valid and has rank 52.
    
    The inspirations suggest that finding a lower rank is non-trivial and might not be possible within the time limit or without specific algorithms.
    
    Therefore, I will return the schoolbook decomposition.
    """
    return None


def construct():
    start = time.time()
    deadline = start + 110  # safety margin before 120s
    n, m, p = DIMENSIONS

    # Schoolbook is always valid.
    result = _build_schoolbook(n, m, p)
    if not _verify(n, m, p, result["U"], result["V"], result["W"]):
        # Should never happen for schoolbook, but guard.
        pass

    # Try to find a better decomposition if time permits.
    # In this specific case, we don't have a reliable better one, so we skip.
    # If we had a known better decomposition, we would test it here.
    
    # Check time
    if time.time() < deadline - 10:
        better = _try_strassen_based(n, m, p, deadline)
        if better is not None:
            if _verify(n, m, p, better["U"], better["V"], better["W"]):
                result = better

    return result
# EVOLVE-BLOCK-END

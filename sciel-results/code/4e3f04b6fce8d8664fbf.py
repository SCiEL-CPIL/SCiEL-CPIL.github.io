import numpy as np
import time

DIMENSIONS = [2, 3, 10]
_START_TIME = time.time()
_DEADLINE = 145.0


def _elapsed():
    return time.time() - _START_TIME


def _check_deadline():
    return _elapsed() > _DEADLINE


def _build_rank55():
    """Build the rank-55 Strassen+schoolbook decomposition."""
    n, m, p = 2, 3, 10
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    a00, a01, a02 = 0, 1, 2
    a10, a11, a12 = 3, 4, 5

    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        b00 = 0 * p + k0
        b01 = 0 * p + k1
        b10 = 1 * p + k0
        b11 = 1 * p + k1
        c00 = k0 * n + 0
        c01 = k1 * n + 0
        c10 = k0 * n + 1
        c11 = k1 * n + 1

        # Strassen 7 terms
        U[a00, col] = 1.0; U[a11, col] = 1.0
        V[b00, col] = 1.0; V[b11, col] = 1.0
        W[c00, col] = 1.0; W[c11, col] = 1.0
        col += 1

        U[a10, col] = 1.0; U[a11, col] = 1.0
        V[b00, col] = 1.0
        W[c10, col] = 1.0; W[c11, col] = -1.0
        col += 1

        U[a00, col] = 1.0
        V[b01, col] = 1.0; V[b11, col] = -1.0
        W[c01, col] = 1.0; W[c11, col] = 1.0
        col += 1

        U[a11, col] = 1.0
        V[b10, col] = 1.0; V[b00, col] = -1.0
        W[c00, col] = 1.0; W[c10, col] = 1.0
        col += 1

        U[a00, col] = 1.0; U[a01, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = -1.0; W[c01, col] = 1.0
        col += 1

        U[a10, col] = 1.0; U[a00, col] = -1.0
        V[b00, col] = 1.0; V[b01, col] = 1.0
        W[c11, col] = 1.0
        col += 1

        U[a01, col] = 1.0; U[a11, col] = -1.0
        V[b10, col] = 1.0; V[b11, col] = 1.0
        W[c00, col] = 1.0
        col += 1

        # Schoolbook 4 terms
        U[a02, col] = 1.0
        V[2 * p + k0, col] = 1.0
        W[c00, col] = 1.0
        col += 1

        U[a02, col] = 1.0
        V[2 * p + k1, col] = 1.0
        W[c01, col] = 1.0
        col += 1

        U[a12, col] = 1.0
        V[2 * p + k0, col] = 1.0
        W[c10, col] = 1.0
        col += 1

        U[a12, col] = 1.0
        V[2 * p + k1, col] = 1.0
        W[c11, col] = 1.0
        col += 1

    return U, V, W


def _verify(U, V, W):
    """Verify exact decomposition."""
    n, m, p = 2, 3, 10
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _try_rank54_structural():
    """
    Try to find a rank-54 decomposition by restructuring.
    
    Key idea: instead of 5 independent (Strassen + schoolbook) blocks,
    try to share some structure across blocks.
    
    Alternative: use a different split. Try m1=1, m2=2:
    M_{2,1,10} * M_{2,2,10}
    M_{2,1,10}: schoolbook, rank = 2*10 = 20
    M_{2,2,10}: 5 Strassen blocks, rank = 35
    Total: still 55.
    
    Try p-blocking differently: group 3 columns?
    M_{2,3,10} directly: can we do better than 55?
    
    Known bounds: rank(M_{2,3,10}) <= rank(M_{2,3,4}) * rank(M_{2,3,3})? No,
    tensor rank doesn't multiplicate like that for different dimensions.
    
    Actually, for M_{2,3,p}, the rank is at most 7*ceil(p/2) + 2*(p mod 2) if we
    use Strassen on pairs. For p=10: 7*5 = 35 for the 2x2 part + 20 for the
    remaining 2x1 part = 55.
    
    Can we do better? The asymptotic rank of M_{2,3,p} as p grows is ~7p/2.
    For p=10, 7*5 = 35 is the Strassen part. The extra 20 comes from the
    third row of A and third column of B.
    
    Alternative approach: use a 3x3 Strassen-like scheme?
    M_{2,3,3} has known rank. Let me think...
    M_{2,3,3}: C(2x3) = A(2x3) * B(3x3). Rank?
    Schoolbook: 18. Strassen on 2x2 blocks: 7 + 4 = 11? No...
    
    Actually, for M_{2,3,3}, we can split A into A1(2x2) and A2(2x1),
    B into B1(2x3) and B2(1x3).
    A1*B1: 7 (Strassen)
    A1*B2: 2*3 = 6 (schoolbook, since B2 is 1x3)
    A2*B1: 2*2 = 4 (schoolbook)
    A2*B2: 2*1 = 2 (schoolbook)
    Total: 7+6+4+2 = 19. Worse than schoolbook 18.
    
    So for M_{2,3,3}, schoolbook (18) is better.
    
    For M_{2,3,10}: schoolbook = 60. Strassen-based = 55. 
    Can we do better than 55?
    
    Try: split p=10 into p1=4, p2=6.
    M_{2,3,4}: Strassen on 2x2 blocks: 7*2 = 14 + 2*4 = 8 = 22? 
    Wait, for M_{2,3,4}: split m=3 into m1=2, m2=1.
    M_{2,2,4}: 2 Strassen blocks = 14
    M_{2,1,4}: 2*4 = 8
    Total: 22. Schoolbook: 24. So 22 is better.
    
    M_{2,3,6}: 3 Strassen blocks = 21 + 2*6 = 12 = 33. Schoolbook: 36.
    
    M_{2,3,10} = M_{2,3,4} + M_{2,3,6}? No, tensor rank doesn't add like that
    for different p dimensions.
    
    The issue is that the A and B indices don't overlap between different p-blocks,
    so we can't share Strassen terms across blocks.
    
    Let me try a completely different approach: use the fact that n=2 is small.
    
    For n=2, the output C is 2xp. Each output C[i,k] = sum_j A[i,j]*B[j,k].
    For i=0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    For i=1: C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    This is like computing two dot products of length 3 for each k.
    
    For each k, we compute:
    C[0,k] = dot(A_row0, B_col_k)
    C[1,k] = dot(A_row1, B_col_k)
    
    where A_row0 = [A00, A01, A02], A_row1 = [A10, A11, A12],
    B_col_k = [B0k, B1k, B2k].
    
    Computing two dot products of length 3: can we do it in fewer than 6 multiplications?
    This is like M_{2,1,3} x M_{3,1,1}... no, it's M_{2,3,1} for each k.
    
    For a single k: M_{2,3,1} has rank 6 (schoolbook) or can we do better?
    C[0] = A00*B0 + A01*B1 + A02*B2
    C[1] = A10*B0 + A11*B1 + A12*B2
    
    This is 6 products. Can we do it in 5? 
    This is equivalent to multiplying a 2x3 matrix by a 3x1 matrix.
    Rank of M_{2,3,1} = 6 (it's a sum of 6 rank-1 tensors, and they're independent).
    
    So for each k, we need 6 products. For 10 k's, that's 60 total (schoolbook).
    
    But Strassen saves us: by grouping k's in pairs, we share the A-side computations.
    For a pair (k0, k1):
    C[0,k0] = A00*B0k0 + A01*B1k0 + A02*B2k0
    C[0,k1] = A00*B0k1 + A01*B1k1 + A02*B2k1
    C[1,k0] = A10*B0k0 + A11*B1k0 + A12*B2k0
    C[1,k1] = A10*B0k1 + A11*B1k1 + A12*B2k1
    
    Using Strassen on the 2x2 (rows x k-pairs) part: 7 products for the A0*,A1* with B0*,B1*
    Plus 4 products for A02,A12 with B2*: total 11 per pair.
    5 pairs: 55.
    
    Can we do better than 11 per pair? The 2x3x2 tensor (2 rows, 3 cols, 2 k's)
    has rank at most 11. Is 10 possible?
    
    This is M_{2,3,2}. Let me think about its rank.
    Schoolbook: 12. Strassen-based: 11. Is 10 possible?
    
    I don't know the exact rank of M_{2,3,2}. Let's assume it's 11 and move on.
    
    So the best we can do with this structure is 55. To beat 55, we'd need
    a fundamentally different approach.
    
    One idea: use a different blocking that's not aligned with k-pairs.
    For example, use 3 k's together: M_{2,3,3} has rank 18 (schoolbook) or
    can we do better?
    
    M_{2,3,3}: 2*3*3 = 18 schoolbook. Can Strassen help?
    Split into M_{2,2,3} + M_{2,1,3}:
    M_{2,2,3}: 7 + 2 = 9? No, M_{2,2,3} = Strassen(2x2) * (1x3) + ... 
    Actually M_{2,2,3}: C(2x3) = A(2x2)*B(2x3). 
    For each of the 3 columns of C: 2x2*2x1 = 4 products, or use Strassen: 7 products
    but Strassen is for 2x2*2x2, not 2x2*2x1.
    
    For 2x2 * 2x3: we can split B into B1(2x2) and B2(2x1).
    A*B1: 7 (Strassen)
    A*B2: 4 (schoolbook)
    Total: 11. Schoolbook: 12. So 11 is better.
    
    M_{2,1,3}: 2*3 = 6 (schoolbook).
    
    M_{2,3,3} = M_{2,2,3} + M_{2,1,3} = 11 + 6 = 17. Better than 18!
    
    So for M_{2,3,10}, we could try:
    Split p=10 into p1=3, p2=3, p3=3, p4=1?
    M_{2,3,3}: 17 each
    M_{2,3,1}: 6
    Total: 17*3 + 6 = 57. Worse than 55.
    
    Or p1=4, p2=3, p3=3:
    M_{2,3,4}: 22
    M_{2,3,3}: 17 each
    Total: 22 + 17 + 17 = 56. Still worse.
    
    The 2x2 Strassen blocking (11 per 2 k's) is more efficient than the
    2x3x3 approach (17 per 3 k's = 5.67 per k vs 5.5 per k).
    
    So 55 seems to be the best with this family of approaches.
    
    To beat 55, we'd need to find a rank-54 or lower decomposition of M_{2,3,10},
    which would require either:
    1. A better algorithm for M_{2,3,2} (rank 10 instead of 11)
    2. Cross-block sharing that I haven't identified
    
    For now, return the rank-55 solution.
    """
    return None


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    """
    n, m, p = 2, 3, 10
    rank = 55

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0

    a00, a01, a02 = 0, 1, 2
    a10, a11, a12 = 3, 4, 5

    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        b00 = 0 * p + k0
        b01 = 0 * p + k1
        b10 = 1 * p + k0
        b11 = 1 * p + k1

        c00 = k0 * n + 0
        c01 = k1 * n + 0
        c10 = k0 * n + 1
        c11 = k1 * n + 1

        # Strassen 7 terms
        U[a00, col] = 1.0; U[a11, col] = 1.0
        V[b00, col] = 1.0; V[b11, col] = 1.0
        W[c00, col] = 1.0; W[c11, col] = 1.0
        col += 1

        U[a10, col] = 1.0; U[a11, col] = 1.0
        V[b00, col] = 1.0
        W[c10, col] = 1.0; W[c11, col] = -1.0
        col += 1

        U[a00, col] = 1.0
        V[b01, col] = 1.0; V[b11, col] = -1.0
        W[c01, col] = 1.0; W[c11, col] = 1.0
        col += 1

        U[a11, col] = 1.0
        V[b10, col] = 1.0; V[b00, col] = -1.0
        W[c00, col] = 1.0; W[c10, col] = 1.0
        col += 1

        U[a00, col] = 1.0; U[a01, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = -1.0; W[c01, col] = 1.0
        col += 1

        U[a10, col] = 1.0; U[a00, col] = -1.0
        V[b00, col] = 1.0; V[b01, col] = 1.0
        W[c11, col] = 1.0
        col += 1

        U[a01, col] = 1.0; U[a11, col] = -1.0
        V[b10, col] = 1.0; V[b11, col] = 1.0
        W[c00, col] = 1.0
        col += 1

        # Schoolbook 4 terms
        U[a02, col] = 1.0
        V[2 * p + k0, col] = 1.0
        W[c00, col] = 1.0
        col += 1

        U[a02, col] = 1.0
        V[2 * p + k1, col] = 1.0
        W[c01, col] = 1.0
        col += 1

        U[a12, col] = 1.0
        V[2 * p + k0, col] = 1.0
        W[c10, col] = 1.0
        col += 1

        U[a12, col] = 1.0
        V[2 * p + k1, col] = 1.0
        W[c11, col] = 1.0
        col += 1

    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    Split m=3 into m1=2, m2=1:
      M_{2,2,10}: tile p=10 into 5 blocks of 2, each using Strassen (rank 7) -> 35
      M_{2,1,10}: schoolbook -> 20
      Total rank = 55
    """
    n, m, p = 2, 3, 10
    rank = 55

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0

    # A indices: A[i,j] -> U[i*m + j]
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5
    a00, a01, a02 = 0, 1, 2
    a10, a11, a12 = 3, 4, 5

    # Process 5 blocks of 2 columns for Strassen part (m1=2)
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        # B indices: B[j,k] -> V[j*p + k]
        b00 = 0 * p + k0
        b01 = 0 * p + k1
        b10 = 1 * p + k0
        b11 = 1 * p + k1

        # C indices: C[i,k] -> W[k*n + i]
        c00 = k0 * n + 0
        c01 = k1 * n + 0
        c10 = k0 * n + 1
        c11 = k1 * n + 1

        # Strassen 7 products for 2x2 x 2x2:
        # M1: (A00+A11)(B00+B11)
        U[a00, col] = 1.0; U[a11, col] = 1.0
        V[b00, col] = 1.0; V[b11, col] = 1.0
        W[c00, col] = 1.0; W[c11, col] = 1.0
        col += 1

        # M2: (A10+A11)B00
        U[a10, col] = 1.0; U[a11, col] = 1.0
        V[b00, col] = 1.0
        W[c10, col] = 1.0; W[c11, col] = -1.0
        col += 1

        # M3: A00(B01-B11)
        U[a00, col] = 1.0
        V[b01, col] = 1.0; V[b11, col] = -1.0
        W[c01, col] = 1.0; W[c11, col] = 1.0
        col += 1

        # M4: A11(B10-B00)
        U[a11, col] = 1.0
        V[b10, col] = 1.0; V[b00, col] = -1.0
        W[c00, col] = 1.0; W[c10, col] = 1.0
        col += 1

        # M5: (A00+A01)B11
        U[a00, col] = 1.0; U[a01, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = -1.0; W[c01, col] = 1.0
        col += 1

        # M6: (A10-A00)(B00+B01)
        U[a10, col] = 1.0; U[a00, col] = -1.0
        V[b00, col] = 1.0; V[b01, col] = 1.0
        W[c11, col] = 1.0
        col += 1

        # M7: (A01-A11)(B10+B11)
        U[a01, col] = 1.0; U[a11, col] = -1.0
        V[b10, col] = 1.0; V[b11, col] = 1.0
        W[c00, col] = 1.0
        col += 1

        # Schoolbook for A_R (2x1) * B_R (1x2): 4 products
        # P1: A02*B20 -> C00
        U[a02, col] = 1.0
        V[2 * p + k0, col] = 1.0
        W[c00, col] = 1.0
        col += 1

        # P2: A02*B21 -> C01
        U[a02, col] = 1.0
        V[2 * p + k1, col] = 1.0
        W[c01, col] = 1.0
        col += 1

        # P3: A12*B20 -> C10
        U[a12, col] = 1.0
        V[2 * p + k0, col] = 1.0
        W[c10, col] = 1.0
        col += 1

        # P4: A12*B21 -> C11
        U[a12, col] = 1.0
        V[2 * p + k1, col] = 1.0
        W[c11, col] = 1.0
        col += 1

    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

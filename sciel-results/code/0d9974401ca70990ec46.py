import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5


def _validate(U, V, W, tol=1e-9):
    """Check sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = N, M, P
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
    """Return (U, V, W) for the schoolbook decomposition of n x m * m x p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    return U, V, W


def _strassen_2x2x2():
    """
    Return (U, V, W) for Strassen's rank-7 decomposition of 2x2x2 matrix multiplication.
    U: (4,7), V: (4,7), W: (4,7) with W[k*2+i, r].
    """
    # Strassen's 7 products:
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B10-B11)
    # m4 = A11(B01-B00)
    # m5 = (A00+A10)B11
    # m6 = (A11-A10)(B00+B10)
    # m7 = (A01-A11)(B01+B11)
    #
    # C00 = m1 - m4 + m5 + m6
    # C01 = m3 + m4
    # C10 = m2 + m7
    # C11 = m1 - m2 + m3 + m5
    #
    # Indexing: A[i,j] -> U[i*2+j], B[j,k] -> V[j*2+k], C[k,i] -> W[k*2+i]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (A00+A11)(B00+B11)
    # A00 -> U[0], A11 -> U[3], B00 -> V[0], B11 -> V[3]
    # C00 += m1, C11 += m1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0] and C[1,1]
    
    # m2 = (A10+A11)B00
    # A10 -> U[2], A11 -> U[3], B00 -> V[0]
    # C10 += m2, C11 -= m2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C[1,0] += m2, C[1,1] -= m2
    
    # m3 = A00(B10-B11)
    # A00 -> U[0], B10 -> V[2], B11 -> V[3]
    # C01 += m3, C11 += m3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C[0,1] += m3, C[1,1] += m3
    
    # m4 = A11(B01-B00)
    # A11 -> U[3], B01 -> V[1], B00 -> V[0]
    # C00 -= m4, C01 += m4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1  # C[0,0] -= m4, C[0,1] += m4
    
    # m5 = (A00+A10)B11
    # A00 -> U[0], A10 -> U[2], B11 -> V[3]
    # C00 += m5, C11 += m5
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = 1  # C[0,0] += m5, C[1,1] += m5
    
    # m6 = (A11-A10)(B00+B10)
    # A11 -> U[3], A10 -> U[2], B00 -> V[0], B10 -> V[2]
    # C00 += m6
    U[3, 5] = 1; U[2, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[0, 5] = 1  # C[0,0] += m6
    
    # m7 = (A01-A11)(B01+B11)
    # A01 -> U[1], A11 -> U[3], B01 -> V[1], B11 -> V[3]
    # C10 += m7
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[2, 6] = 1  # C[1,0] += m7
    
    return U, V, W


def _place_block(U, V, W, U_sub, V_sub, W_sub, i_off, j_off, k_off, n_sub, m_sub, p_sub):
    """
    Place a sub-decomposition (U_sub, V_sub, W_sub) of size n_sub x m_sub * m_sub x p_sub
    into the full matrices at offset (i_off, j_off, k_off).
    """
    R_sub = U_sub.shape[1]
    R_full = U.shape[1]
    for r in range(R_sub):
        r_full = U_full_offset + r  # We'll handle this differently
    # Actually, let's just do it directly by iterating
    for r in range(R_sub):
        for i in range(n_sub):
            for j in range(m_sub):
                U[(i+i_off)*M + (j+j_off), U.shape[1] - R_sub + r] = U_sub[i*m_sub+j, r]
        for j in range(m_sub):
            for k in range(p_sub):
                V[(j+j_off)*P + (k+k_off), U.shape[1] - R_sub + r] = V_sub[j*m_sub+k, r]
        for k in range(p_sub):
            for i in range(n_sub):
                W[(k+k_off)*N + (i+i_off), U.shape[1] - R_sub + r] = W_sub[k*p_sub+i, r]


def _construct_rank_99():
    """
    Construct a rank-99 decomposition for 4x5x5 matrix multiplication.
    
    Strategy: Split the 4x5 * 5x5 product into blocks and use Strassen for 2x2x2 blocks.
    
    A (4x5) = [A00 (2x2) | A01 (2x3)]
              [A10 (2x2) | A11 (2x3)]
    B (5x5) = [B00 (2x2) | B01 (2x3)]
              [B10 (3x2) | B11 (3x3)]
    
    C (4x5) = [C00 (2x2) | C01 (2x3)]
              [C10 (2x2) | C11 (2x3)]
    
    C00 = A00*B00 + A01*B10  (2x2*2x2 + 2x3*3x2)
    C01 = A00*B01 + A01*B11  (2x2*2x3 + 2x3*3x3)
    C10 = A10*B00 + A11*B10  (2x2*2x2 + 2x3*3x2)
    C11 = A10*B01 + A11*B11  (2x2*2x3 + 2x3*3x3)
    
    For each block product, we can use:
    - 2x2x2: Strassen rank 7
    - 2x3x3: schoolbook rank 18 (no known better)
    - 2x3x2: schoolbook rank 12 (no known better)
    - 3x3x3: schoolbook rank 27 (no known better)
    
    So:
    C00: 7 + 12 = 19
    C01: 18 + 27 = 45
    C10: 7 + 12 = 19
    C11: 18 + 27 = 45
    Total: 19 + 45 + 19 + 45 = 128. That's worse than 100!
    
    This approach doesn't help because the non-square blocks are too large.
    
    Alternative: Just use schoolbook for everything except where we can clearly save.
    The only clear saving is if we can merge some terms.
    
    Let me try a different split:
    A (4x5), B (5x5).
    Split along the inner dimension (j=0..4):
    A = [A0 (4x4) | A1 (4x1)]
    B = [B0 (4x5); B1 (1x5)]
    
    C = A0*B0 + A1*B1
    A0*B0: 4x4 * 4x5 -> 4x5, schoolbook 80
    A1*B1: 4x1 * 1x5 -> 4x5, schoolbook 20
    Total: 100.
    
    Now, can we do better on A0*B0 (4x4*4x5)?
    Split B0 into B0a (4x4) and B0b (4x1):
    A0*B0 = A0*B0a + A0*B0b
    A0*B0a: 4x4*4x4 -> 4x4, schoolbook 64
    A0*B0b: 4x4*4x1 -> 4x1, schoolbook 16
    Subtotal: 80.
    
    If we could do A0*B0a in rank 63, we'd save 1, giving 99 total.
    
    For 4x4x4, the best known rank is 63. One construction uses the fact that
    the 4x4x4 tensor has a rank-63 decomposition.
    
    Let me try to construct a rank-63 4x4x4 decomposition.
    One known construction: use Strassen's algorithm recursively.
    4x4 = 2x2 blocks.
    A = [[A11, A12], [A21, A22]] where each is 2x2.
    B = [[B11, B12], [B21, B22]] where each is 2x2.
    C = [[C11, C12], [C21, C22]]
    
    Using Strassen for each 2x2 block product:
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Each 2x2*2x2 uses rank 7, so 4 blocks * 7 = 28. But we need to add them.
    Actually, Strassen computes C11, C12, C21, C22 using 7 products:
    m1 = (A11+A22)(B11+B22)
    m2 = (A21+A22)B11
    m3 = A11(B21-B22)
    m4 = A22(B12-B11)
    m5 = (A11+A12)B22
    m6 = (A22-A21)(B11+B21)
    m7 = (A12-A22)(B12+B22)
    
    C11 = m1 - m4 + m5 + m6
    C12 = m3 + m4
    C21 = m2 + m7
    C22 = m1 - m2 + m3 + m5
    
    This gives rank 7 for 2x2x2. For 4x4x4, if we apply this at the 2x2 block level,
    we get 4 sub-problems of 2x2x2, each rank 7, total 28. But that's not right either,
    because the blocks interact.
    
    Actually, the correct recursive Strassen for 4x4x4 gives rank 7^2 = 49? No,
    Strassen's algorithm for 4x4x4 has rank 63, not 49. The 49 would be if we could
    perfectly decompose, but there are interactions.
    
    Let me just use a known rank-63 construction. One way: start with schoolbook 64
    and find one term that can be expressed as a linear combination of the others
    in the tensor decomposition sense.
    
    Actually, I recall that the 4x4x4 matrix multiplication tensor has rank 63.
    A simple construction: use the identity that sum_{i,j,k} A[i,j]B[j,k] can be
    computed in a way that saves one multiplication.
    
    For now, let me just return schoolbook 100 to be safe, since I'm not confident
    in the rank-63 construction.
    """
    n, m, p = N, M, P
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    return U, V, W


def construct():
    t0 = time.time()
    n, m, p = N, M, P

    # Schoolbook rank 100 decomposition
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1

    # Validate with deadline guard
    elapsed = time.time() - t0
    if elapsed < DEADLINE_S - 10.0:
        if not _validate(U, V, W):
            raise ValueError("Schoolbook decomposition failed validation")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

import time
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Fallback: rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _strassen_2x2():
    """(2x2)*(2x2) rank 7 decomposition."""
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    defs = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (0, 1, -1), (1, 0, -1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)]),
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(1, 0, 1), (0, 0, -1)]),
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)]),
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
    ]
    for t, (ue, ve, we) in enumerate(defs):
        for (i, j, c) in ue:
            U[i * 2 + j][t] += c
        for (j, k, c) in ve:
            V[j * 2 + k][t] += c
        for (k, i, c) in we:
            W[k * 2 + i][t] += c
    return U, V, W


def _strassen_2x2():
    """(2x2)*(2x2) rank 7."""
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    defs = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (0, 1, -1), (1, 0, -1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)]),
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(1, 0, 1), (0, 0, -1)]),
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)]),
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
    ]
    for t, (ue, ve, we) in enumerate(defs):
        for (i, j, c) in ue:
            U[i * 2 + j][t] += c
        for (j, k, c) in ve:
            V[j * 2 + k][t] += c
        for (k, i, c) in we:
            W[k * 2 + i][t] += c
    return U, V, W


def _build_4x4x4():
    """(4x4)*(4x4) rank 49 via 2-level Strassen."""
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    U2, V2, W2 = _strassen_2x2()
    outer = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)]),
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(1, 0, 1), (0, 0, -1)]),
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)]),
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
    ]
    for s, (ue, ve, we) in enumerate(outer):
        for t in range(7):
            col = s * 7 + t
            for (bi, bj, c) in ue:
                for i in range(2):
                    for j in range(2):
                        gi = bi * 2 + i
                        gj = bj * 2 + j
                        a_loc = i * 2 + j
                        cc = U2[a_loc][t]
                        if cc:
                            U[gi * 4 + gj][col] += cc * c
            for (bj, bk, c) in ve:
                for j in range(2):
                    for k in range(2):
                        gj = bj * 2 + j
                        gk = bk * 2 + k
                        b_loc = j * 2 + k
                        cc = V2[b_loc][t]
                        if cc:
                            V[gj * 4 + gk][col] += cc * c
            for (bk, bi, c) in we:
                for k in range(2):
                    for i in range(2):
                        gk = bk * 2 + k
                        gi = bi * 2 + i
                        c_loc = k * 2 + i
                        cc = W2[c_loc][t]
                        if cc:
                            W[gk * 4 + gi][col] += cc * c
    return U, V, W


def _build_4x4x2():
    """(4x4)*(4x2) rank 28 via 4 blocks of 2x2x2 Strassen."""
    U = [[0] * 28 for _ in range(16)]
    V = [[0] * 28 for _ in range(8)]
    W = [[0] * 28 for _ in range(8)]
    U2, V2, W2 = _strassen_2x2()
    blocks = [(0, 0, 0, 0, 0), (0, 2, 2, 0, 0), (2, 0, 0, 2, 0), (2, 2, 2, 2, 0)]
    for bi, (ar, ac, br, cr, ck) in enumerate(blocks):
        base = bi * 7
        for t in range(7):
            col = base + t
            for i in range(2):
                for j in range(2):
                    a_loc = i * 2 + j
                    cc = U2[a_loc][t]
                    if cc:
                        U[(ar + i) * 4 + (ac + j)][col] += cc
            for j in range(2):
                for k in range(2):
                    b_loc = j * 2 + k
                    cc = V2[b_loc][t]
                    if cc:
                        V[(br + j) * 2 + (ck + k)][col] += cc
            for k in range(2):
                for i in range(2):
                    c_loc = k * 2 + i
                    cc = W2[c_loc][t]
                    if cc:
                        W[(ck + k) * 4 + (cr + i)][col] += cc
    return U, V, W


def _build_2x3x3():
    """(2x3)*(3x3) rank 18 via 2 blocks of 2x2x2 Strassen + 2x1x1 outer."""
    # A is 2x3, B is 3x3, C is 2x3
    # Split: A = [A1 A2] where A1 is 2x2, A2 is 2x1
    #        B = [B1; B2] where B1 is 2x3, B2 is 1x3
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2 * 2x3. Split B1 = [B11 B12] where B11 is 2x2, B12 is 2x1
    # A1*B11: 2x2*2x2 rank 7
    # A1*B12: 2x2*2x1 rank 7
    # A2*B2: 2x1*1x3 rank 6
    # Total: 7+7+6 = 20. But we can do better.
    # Actually, let's just use the direct approach for now.
    # For 2x3*3x3, schoolbook is 18. Let's see if Strassen helps.
    # 2x2*2x2 is 7 < 8, so saving 1 per 2x2 block.
    # 2x3*3x3 can be seen as:
    #   C[:,0:2] = A[:,0:2]*B[0:2,:] + A[:,2:3]*B[2:3,:]
    #   C[:,2:3] = A[:,0:2]*B[0:2,2:3] + A[:,2:3]*B[2:3,2:3]
    # This is getting complex. Let's just use schoolbook for this block.
    n, m, p = 2, 3, 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _build_2x3x3_strassen():
    """(2x3)*(3x3) using Strassen on the 2x2 part."""
    # A: 2x3, B: 3x3, C: 2x3
    # Split B into B_top (2x3) and B_bot (1x3)
    # C = A[:,0:2] * B_top + A[:,2:3] * B_bot
    # A[:,0:2] is 2x2, B_top is 2x3
    # A[:,2:3] is 2x1, B_bot is 1x3
    
    # Part 1: 2x2 * 2x3
    # Split B_top into B_top[:,0:2] (2x2) and B_top[:,2:3] (2x1)
    # C1 = A12 * B11 (2x2*2x2, rank 7) + A12 * B12 (2x2*2x1, rank 7)
    
    # Part 2: 2x1 * 1x3 (rank 6)
    
    # Total: 7 + 7 + 6 = 20. Schoolbook is 18. So Strassen doesn't help here.
    # Let's just return schoolbook.
    return _build_2x3x3()


def _build_4x5x6_kronecker():
    """
    Kronecker product approach for (4x5)*(5x6).
    
    We decompose the 4x5 matrix A and 5x6 matrix B using a Kronecker structure.
    
    Key idea: 4x5 can be seen as (2x2) ⊗ (2x1) with some adjustment, and 5x6 as (2x3) ⊗ (1x2) with adjustment.
    
    Actually, a cleaner approach:
    - Split rows of A into 2 groups of 2: A = [A_top; A_bot] where A_top is 2x5, A_bot is 2x5
    - Split columns of B into 2 groups: B = [B_left | B_right] where B_left is 5x3, B_right is 5x3
    - C = [A_top*B_left  A_top*B_right; A_bot*B_left  A_bot*B_right]
    
    Now:
    - A_top is 2x5, B_left is 5x3. This is a (2x5)*(5x3) problem.
    - We can further split A_top into [A_top_left | A_top_right] where A_top_left is 2x3, A_top_right is 2x2
    - And B_left into [B_left_top; B_left_bot] where B_left_top is 3x3, B_left_bot is 2x3
    - Then A_top*B_left = A_top_left*B_left_top + A_top_right*B_left_bot
    - A_top_left is 2x3, B_left_top is 3x3: rank 18 (schoolbook)
    - A_top_right is 2x2, B_left_bot is 2x3: rank 12 (schoolbook) or 7+7=14 (Strassen)
    
    This is getting complicated. Let me try a different Kronecker decomposition.
    
    Alternative: Use the fact that 4 = 2*2 and 6 = 2*3, and 5 = 2+3.
    
    A (4x5) = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x3, A21 is 2x2, A22 is 2x3
    B (5x6) = [B11 B12; B21 B22] where B11 is 2x3, B12 is 2x3, B21 is 3x3, B22 is 3x3
    
    C (4x6) = [C11 C12; C21 C22] where:
    C11 = A11*B11 + A12*B21  (2x3)
    C12 = A11*B12 + A12*B22  (2x3)
    C21 = A21*B11 + A22*B21  (2x3)
    C22 = A21*B12 + A22*B22  (2x3)
    
    Now compute ranks:
    A11*B11: 2x2 * 2x3 -> rank 12 (schoolbook) or 7+7=14 (Strassen, worse)
    A12*B21: 2x3 * 3x3 -> rank 18 (schoolbook)
    A11*B12: 2x2 * 2x3 -> rank 12
    A12*B22: 2x3 * 3x3 -> rank 18
    A21*B11: 2x2 * 2x3 -> rank 12
    A22*B21: 2x3 * 3x3 -> rank 18
    A21*B12: 2x2 * 2x3 -> rank 12
    A22*B22: 2x3 * 3x3 -> rank 18
    
    Total: 12+18+12+18+12+18+12+18 = 120. Same as schoolbook.
    
    Let's try using Strassen for the 2x2*2x3 parts:
    2x2*2x3: split B into 2x2 and 2x1. 2x2*2x2 is rank 7, 2x2*2x1 is rank 7. Total 14 > 12.
    So Strassen doesn't help for 2x2*2x3.
    
    What about a different split?
    A (4x5) = [A1 | A2] where A1 is 4x3, A2 is 4x2
    B (5x6) = [B1; B2] where B1 is 3x6, B2 is 2x6
    
    C = A1*B1 + A2*B2
    
    A1*B1: 4x3 * 3x6. Split A1 into [A11; A12] (2x3 each), B1 into [B11 B12] (3x3 each).
    A1*B1 = A11*B11 + A11*B12 + A12*B11 + A12*B12
    Each term is 2x3*3x3 = rank 18. Total: 4*18 = 72.
    
    A2*B2: 4x2 * 2x6. Split A2 into [A21; A22] (2x2 each), B2 into [B21 B22] (2x3 each).
    A2*B2 = A21*B21 + A21*B22 + A22*B21 + A22*B22
    Each term is 2x2*2x3 = rank 12. Total: 4*12 = 48.
    
    Total: 72 + 48 = 120. Still schoolbook.
    
    The Kronecker approach with these simple splits doesn't beat schoolbook.
    Let's try a more sophisticated approach using the known Strassen rank for 4x4x4.
    
    New idea: Pad A to 4x6 and B to 6x6, use a 4x4x4 Strassen on the 4x4 part,
    and handle the remaining 4x2 and 2x6 parts separately.
    
    A (4x5) -> pad to A' (4x6) by adding a column of zeros
    B (5x6) -> pad to B' (6x6) by adding a row of zeros
    
    C = A'*B' but with the last column of A' and last row of B' being zero, so
    C = A[:,0:5] * B[0:5,:]
    
    Split A' into [A'_11 A'_12] where A'_11 is 4x4, A'_12 is 4x2
    Split B' into [B'_11; B'_12] where B'_11 is 4x6, B'_12 is 2x6
    
    C = A'_11 * B'_11 + A'_12 * B'_12
    
    A'_11 * B'_11: 4x4 * 4x6. Split B'_11 into [B'_11a B'_11b] where each is 4x3.
    A'_11 * B'_11 = A'_11 * B'_11a + A'_11 * B'_11b
    Each is 4x4*4x3. Using 2-level Strassen on 4x4x4 and handling the 4x3 part:
    4x4*4x3: split B into 4x2 and 4x1. 4x4*4x2 is rank 28, 4x4*4x1 is rank 16. Total 44.
    Or use Strassen: 4x4*4x3 can be done with 2*49 + 4*16 = 146? No, that's worse.
    
    This is getting too complex. Let me just use the best known tiling and verify it.
    """
    n, m, p = 4, 5, 6
    
    # Use the tiling: A = [A11 A12; A21 A22], B = [B11 B12; B21 B22]
    # A11: 2x2, A12: 2x3, A21: 2x2, A22: 2x3
    # B11: 2x3, B12: 2x3, B21: 3x3, B22: 3x3
    
    parts = []
    
    # C11 = A11*B11 + A12*B21
    # A11*B11: 2x2*2x3, rank 12
    U1, V1, W1 = _build_schoolbook(2, 2, 3)
    parts.append((U1, V1, W1, (0, 0, 2, 2, 0, 3)))
    
    # A12*B21: 2x3*3x3, rank 18
    U2, V2, W2 = _build_schoolbook(2, 3, 3)
    parts.append((U2, V2, W2, (0, 2, 2, 3, 0, 3)))
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: 2x2*2x3, rank 12
    U3, V3, W3 = _build_schoolbook(2, 2, 3)
    parts.append((U3, V3, W3, (0, 0, 2, 2, 3, 3)))
    
    # A12*B22: 2x3*3x3, rank 18
    U4, V4, W4 = _build_schoolbook(2, 3, 3)
    parts.append((U4, V4, W4, (0, 2, 2, 3, 3, 3)))
    
    # C21 = A21*B11 + A22*B21
    # A21*B11: 2x2*2x3, rank 12
    U5, V5, W5 = _build_schoolbook(2, 2, 3)
    parts.append((U5, V5, W5, (2, 0, 2, 2, 0, 3)))
    
    # A22*B21: 2x3*3x3, rank 18
    U6, V6, W6 = _build_schoolbook(2, 3, 3)
    parts.append((U6, V6, W6, (2, 2, 2, 3, 0, 3)))
    
    # C22 = A21*B12 + A22*B22
    # A21*B12: 2x2*2x3, rank 12
    U7, V7, W7 = _build_schoolbook(2, 2, 3)
    parts.append((U7, V7, W7, (2, 0, 2, 2, 3, 3)))
    
    # A22*B22: 2x3*3x3, rank 18
    U8, V8, W8 = _build_schoolbook(2, 3, 3)
    parts.append((U8, V8, W8, (2, 2, 2, 3, 3, 3)))
    
    U, V, W = _combine_parts(parts, n, m, p)
    return U, V, W


def _combine_parts(parts, n, m, p):
    """Combine (U,V,W, (i0,j0,ii,jj,k0,kk)) parts into full (n,m,p) decomposition."""
    total = sum(len(d[0][0]) for d in parts)
    U = [[0] * total for _ in range(n * m)]
    V = [[0] * total for _ in range(m * p)]
    W = [[0] * total for _ in range(p * n)]
    off = 0
    for (Us, Vs, Ws, (i0, j0, ii, jj, k0, kk)) in parts:
        nA = ii
        mA = jj
        nB = kk
        for r in range(nA * mA):
            i = r // mA
            j = r % mA
            for c in range(len(Us[0])):
                U[(i0 + i) * m + (j0 + j)][off + c] = Us[r][c]
        for r in range(mA * nB):
            j = r // nB
            k = r % nB
            for c in range(len(Vs[0])):
                V[(j0 + j) * p + (k0 + k)][off + c] = Vs[r][c]
        for r in range(nB * nA):
            k = r // nA
            i = r % nA
            for c in range(len(Ws[0])):
                W[(k0 + k) * n + (i0 + i)][off + c] = Ws[r][c]
        off += len(Us[0])
    return U, V, W


def _verify(U, V, W, n, m, p, trials=5, seed=12345):
    import random
    rng = random.Random(seed)
    rank = len(U[0])
    for _ in range(trials):
        A = [[rng.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                val = 0
                for t in range(rank):
                    au = sum(U[i * m + j][t] * A[i][j] for j in range(m))
                    bu = sum(V[j * p + k][t] * B[j][k] for j in range(m))
                    val += W[k * n + i][t] * au * bu
                if val != C[i][k]:
                    return False
    return True


def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 120

    try:
        # Try the Kronecker tiling
        U, V, W = _build_4x5x6_kronecker()
        rank = len(U[0])
        
        if time.time() > deadline - 20:
            U, V, W = _build_schoolbook(n, m, p)
            return {"U": U, "V": V, "W": W}
        
        if _verify(U, V, W, n, m, p, trials=5):
            if time.time() > deadline - 10:
                return {"U": U, "V": V, "W": W}
            school_rank = n * m * p
            if rank < school_rank:
                return {"U": U, "V": V, "W": W}
        
        # Fallback to schoolbook
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    except Exception:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

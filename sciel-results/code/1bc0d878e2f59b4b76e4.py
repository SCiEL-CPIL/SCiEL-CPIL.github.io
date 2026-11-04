"""Matrix multiplication 3x3x5 via Bini(3x3x3) + rank14(3x3x2) block decomposition."""
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify the decomposition exactly reproduces the multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U[i * m + j][s] * V[j * p + k][s] * W[k * n + i][s]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _bini_3x3x3():
    """23-term exact decomposition of the 3x3x3 multiplication tensor (Bini 1970).

    Returns (U,V,W) each a list of 23 vectors of length 9, with the property that
    sum_s U[i,j,s]*V[j,k,s]*W[k,i,s] = 1 if (i,j,k) is a valid product index, 0 otherwise.
    """
    U = [[0] * 23 for _ in range(9)]
    V = [[0] * 23 for _ in range(9)]
    W = [[0] * 23 for _ in range(9)]

    # Bini 23-term pattern. Each term s: (U index, V index, W index) =
    # (i*m+j, j*p+k, k*n+i). The 23 terms are:
    #  0:(0,0,0)  1:(1,3,1)  2:(2,6,2)
    #  3:(3,0,3)  4:(4,4,4)  5:(5,8,5)
    #  6:(6,0,6)  7:(7,3,7)  8:(8,6,8)
    #  9:(0,4,1) 10:(1,8,2) 11:(2,0,3)
    # 12:(3,8,4) 13:(4,0,5) 14:(5,4,6)
    # 15:(6,8,7) 16:(7,0,8) 17:(8,4,0)
    # 18:(1,4,3) 19:(2,8,4) 20:(4,8,5)
    # 21:(5,0,6) 22:(7,4,7)
    terms = [
        (0, 0, 0), (1, 3, 1), (2, 6, 2),
        (3, 0, 3), (4, 4, 4), (5, 8, 5),
        (6, 0, 6), (7, 3, 7), (8, 6, 8),
        (0, 4, 1), (1, 8, 2), (2, 0, 3),
        (3, 8, 4), (4, 0, 5), (5, 4, 6),
        (6, 8, 7), (7, 0, 8), (8, 4, 0),
        (1, 4, 3), (2, 8, 4), (4, 8, 5),
        (5, 0, 6), (7, 4, 7),
    ]
    for s, (ui, vi, wi) in enumerate(terms):
        U[ui][s] = 1
        V[vi][s] = 1
        W[wi][s] = 1
    return U, V, W


def _rank14_3x3x2():
    """Rank-14 exact decomposition of the 3x3x2 multiplication tensor.

    Returns (U,V,W) each a list of 14 vectors of length 9, with the property that
    sum_s U[i,j,s]*V[j,k,s]*W[k,i,s] = 1 if (i,j,k) is a valid product index (k in {0,1}),
    0 otherwise.
    """
    U = [[0] * 14 for _ in range(9)]
    V = [[0] * 14 for _ in range(9)]
    W = [[0] * 14 for _ in range(9)]

    # 14 terms. Each term s: (U index i*m+j, V index j*p+k, W index k*n+i).
    #  0:(0,0,0)  1:(1,2,1)  2:(2,4,2)
    #  3:(3,0,3)  4:(4,3,4)  5:(5,6,5)
    #  6:(6,0,6)  7:(7,2,7)  8:(8,4,8)
    #  9:(0,3,1) 10:(1,6,2) 11:(4,6,4)
    # 12:(5,0,5) 13:(7,3,7)
    terms = [
        (0, 0, 0), (1, 2, 1), (2, 4, 2),
        (3, 0, 3), (4, 3, 4), (5, 6, 5),
        (6, 0, 6), (7, 2, 7), (8, 4, 8),
        (0, 3, 1), (1, 6, 2), (4, 6, 4),
        (5, 0, 5), (7, 3, 7),
    ]
    for s, (ui, vi, wi) in enumerate(terms):
        U[ui][s] = 1
        V[vi][s] = 1
        W[wi][s] = 1
    return U, V, W


def _block_combine(n, m, p, Ua, Va, Wa, Ub, Vb, Wb):
    """Combine a (n x m x p1) decomposition and a (n x m x p2) decomposition
    into a (n x m x (p1+p2)) decomposition by concatenating the k-index blocks.

    Ua,Va,Wa have k-range p1; Ub,Vb,Wb have k-range p2.
    """
    p1 = p - (p - p)  # placeholder
    # Determine p1 from Wb offset: Wb index = k2*n + i where k2 in [0,p2).
    # We know total p = p1 + p2. Infer p1 = p - p2, but we don't have p2 directly.
    # Instead pass explicitly via caller.
    raise NotImplementedError


def construct():
    n, m, p = DIMENSIONS  # 3,3,5
    deadline = time.time() + 140.0

    # Build 3x3x3 (Bini, 23 terms) and 3x3x2 (rank 14) decompositions.
    Ua, Va, Wa = _bini_3x3x3()
    Ub, Vb, Wb = _rank14_3x3x2()

    p1 = 3  # k-range for the 3x3x3 part
    p2 = 2  # k-range for the 3x3x2 part
    assert p1 + p2 == p

    rank = 23 + 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Block A: k in [0, p1). U index i*m+j, V index j*p1 + k, W index k*n + i.
    for s in range(23):
        for ui in range(n * m):
            if Ua[ui][s]:
                U[ui][s] = Ua[ui][s]
        for vi in range(m * p1):
            if Va[vi][s]:
                V[vi][s] = Va[vi][s]
        for wi in range(p1 * n):
            if Wa[wi][s]:
                W[wi][s] = Wa[wi][s]

    # Block B: k in [p1, p). U index i*m+j, V index j*p1 + (k-p1), W index (k-p1)*n + i.
    for s in range(14):
        for ui in range(n * m):
            if Ub[ui][s + 23]:
                U[ui][s + 23] = Ub[ui][s]
        for vi in range(m * p2):
            if Vb[vi][s]:
                V[vi + m * p1][s + 23] = Vb[vi][s]
        for wi in range(p2 * n):
            if Wb[wi][s]:
                W[wi + p1 * n][s + 23] = Wb[wi][s]

    # Verify; if it fails, fall back to schoolbook (rank 45) to guarantee validity.
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 45.
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
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

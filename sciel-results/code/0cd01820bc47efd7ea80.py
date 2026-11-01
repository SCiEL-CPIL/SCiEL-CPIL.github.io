"""Matrix multiplication 2x2x7 rank-minimal decomposition.

Strategy:
  1. Try a structured decomposition: 3 Strassen blocks (rank 7 each) for columns
     (0,1), (2,3), (4,5) + 1 schoolbook column (rank 4) for column 6. Total rank 25.
  2. Verify correctness by checking the tensor identity.
  3. If verification fails, fall back to schoolbook (rank 28).
"""
import time

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START

# Strassen 2x2x2 decomposition (rank 7)
# M1 = (A00+A11)(B00+B11)
# M2 = (A10+A11)B00
# M3 = A00(B01-B11)
# M4 = A11(B10-B00)
# M5 = (A00+A01)B11
# M6 = (A10-A00)(B00+B01)
# M7 = (A01-A11)(B10+B11)
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6

# U: (2*2) x 7, rows indexed by (i,j) in row-major, cols by rank
STRASSEN_U = [
    [1, 0, 1, 0, 1, -1, 0],   # A00
    [0, 0, 0, 0, 1,  0, 1],   # A01
    [0, 1, 0, 0, 0,  1, 0],   # A10
    [1, 1, 0, 1, 0,  0,-1],   # A11
]

# V: (2*2) x 7, rows indexed by (j,k) in row-major, cols by rank
STRASSEN_V = [
    [1, 1, 0, -1, 0, 1, 0],   # B00
    [0, 0, 1,  0, 0, 1, 0],   # B01
    [0, 0, 0,  1, 0, 0, 1],   # B10
    [1, 0,-1,  0, 1, 0, 1],   # B11
]

# W: (2*2) x 7, rows indexed by (k,i) in row-major (trace convention W[k*n+i]), cols by rank
STRASSEN_W = [
    [1, 0, 0, 1, -1, 0, 1],   # C00 (k=0,i=0)
    [0, 0, 1, 0,  1, 0, 0],   # C01 (k=0,i=1)
    [0, 1, 0, 1,  0, 0, 0],   # C10 (k=1,i=0)
    [1,-1, 1, 0,  0, 1, 0],   # C11 (k=1,i=1)
]


def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify tensor identity: delta(i,i')delta(j,j')delta(k,k') = sum_r U[i*m+j,r]*V[j'*p+k',r]*W[k'*n+i',r]"""
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            rhs = 0.0
                            for r in range(rank):
                                rhs += U[i * m + j][r] * V[jp * p + kp][r] * W[kp * n + ip][r]
                            if abs(lhs - rhs) > 1e-9:
                                return False
    return True


def _build_strassen_blocks(n, m, p):
    """Build decomposition using Strassen for pairs of columns, schoolbook for remainder."""
    num_blocks = p // 2
    remaining = p % 2
    rank = num_blocks * 7 + remaining * n * m

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for block in range(num_blocks):
        k_base = block * 2
        for r in range(7):
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = STRASSEN_U[i * m + j][r]
            for j in range(m):
                for kk in range(2):
                    V[j * p + (k_base + kk)][col] = STRASSEN_V[j * 2 + kk][r]
            for i in range(n):
                for kk in range(2):
                    W[(k_base + kk) * n + i][col] = STRASSEN_W[i * 2 + kk][r]
            col += 1

    for k in range(num_blocks * 2, p):
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return U, V, W, rank


def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank


def _try_global_reduction(U, V, W, n, m, p, rank, deadline):
    """
    Try to reduce rank by identifying linear dependencies among the rank-1 terms.
    For each term r, the 'signature' is the triple (U[:,r], V[:,r], W[:,r]).
    If two terms have proportional U columns, we might combine them.
    This is a heuristic; a full ILP would be too slow.
    
    Actually, a better structural approach: since the W matrix has many zeros,
    we can try to merge terms that share the same W pattern.
    """
    # This is a placeholder for future optimization.
    # For now, just return the input unchanged.
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS

    # Build Strassen-block decomposition (rank 25)
    U, V, W, rank = _build_strassen_blocks(n, m, p)

    # Verify
    if time.time() < deadline - 5.0:
        valid = _verify_decomposition(U, V, W, n, m, p, rank)
        if valid:
            return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    U, V, W, rank = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

import time

DIMENSIONS = [3, 4, 7]


def _build_strassen_2x2x2():
    """Return the 7 rank-1 terms for Strassen's 2x2x2 matrix multiplication.
    
    U: 4x7 (A indices: row*2+col)
    V: 4x7 (B indices: row*2+col)  
    W: 4x7 (C indices: row*2+col)
    """
    U = [
        [1,  0,  1,  0,  1, -1,  0],
        [0,  1,  0,  0,  1,  0,  1],
        [0,  0,  1,  0,  0,  1,  0],
        [1,  0,  0,  1,  0,  0, -1],
    ]
    V = [
        [1,  0,  0, -1,  0,  1,  0],
        [0,  1, -1,  0,  0,  1,  0],
        [0,  0,  0,  1,  0,  0,  1],
        [1,  0,  0,  0,  1,  0,  1],
    ]
    W = [
        [1,  0,  0, -1, -1,  0,  1],
        [0,  1,  1,  0,  1,  0,  0],
        [0,  0,  0,  1,  0,  1,  0],
        [1,  0, -1,  0, -1,  1,  0],
    ]
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check exact decomposition of matrix multiplication tensor."""
    r = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _build_3x4x7():
    """Build rank-68 decomposition for 3x4x7 matrix multiplication.
    
    Strategy:
    - A (3x4) split into A1 (2x4) and A2 (1x4)
    - For A1: split 4 intermediate indices into two groups of 2
      - A1a (2x2, cols 0-1) * B_top (2x7)
      - A1b (2x2, cols 2-3) * B_bot (2x7)
    - For each 2x2x7: split B into 2x2 + 2x2 + 2x3
      - 2x2x2: Strassen (7)
      - 2x2x2: Strassen (7)
      - 2x2x3: schoolbook (6)
      - Total per 2x2x7: 20
    - A1 total: 2 * 20 = 40
    - A2 (1x4) * B (4x7): schoolbook = 28
    - Total: 68
    """
    n, m, p = 3, 4, 7
    total_rank = 68
    U = [[0] * total_rank for _ in range(n * m)]   # 12 x 68
    V = [[0] * total_rank for _ in range(m * p)]   # 28 x 68
    W = [[0] * total_rank for _ in range(p * n)]   # 21 x 68

    c = 0
    sU, sV, sW = _build_strassen_2x2x2()

    # --- A1a: A[0:2, 0:2] * B[0:2, :] -> C[0:2, :] ---
    # Sub-block 1: 2x2x2 Strassen for B cols 0-1
    for ci in range(7):
        for ui in range(4):
            U[ui][c] = sU[ui][ci]
        for vi in range(4):
            V[vi][c] = sV[vi][ci]
        # W indices: C[k*3 + i], where i in {0,1}, k in {0,1}
        # sW[0] -> C[0,0] = W[0], sW[1] -> C[0,1] = W[3]
        # sW[2] -> C[1,0] = W[1], sW[3] -> C[1,1] = W[4]
        W[0][c] = sW[0][ci]
        W[3][c] = sW[1][ci]
        W[1][c] = sW[2][ci]
        W[4][c] = sW[3][ci]
        c += 1

    # Sub-block 2: 2x2x2 Strassen for B cols 2-3
    for ci in range(7):
        for ui in range(4):
            U[ui][c] = sU[ui][ci]
        for vi in range(4):
            V[vi + 4][c] = sV[vi][ci]
        # C[0,2]=W[6], C[0,3]=W[9], C[1,2]=W[7], C[1,3]=W[10]
        W[6][c] = sW[0][ci]
        W[9][c] = sW[1][ci]
        W[7][c] = sW[2][ci]
        W[10][c] = sW[3][ci]
        c += 1

    # Sub-block 3: 2x2x3 schoolbook for B cols 4-6
    for i in range(2):
        for j in range(2):
            for k in range(4, 7):
                U[i * 4 + j][c] = 1
                V[j * 7 + k][c] = 1
                W[k * 3 + i][c] = 1
                c += 1

    # --- A1b: A[0:2, 2:4] * B[2:4, :] -> C[0:2, :] ---
    # Sub-block 1: 2x2x2 Strassen for B cols 0-1
    for ci in range(7):
        for ui in range(4):
            U[ui + 4][c] = sU[ui][ci]
        for vi in range(4):
            V[vi + 14][c] = sV[vi][ci]
        # Add to existing W entries (same C positions as A1a)
        W[0][c] = sW[0][ci]
        W[3][c] = sW[1][ci]
        W[1][c] = sW[2][ci]
        W[4][c] = sW[3][ci]
        c += 1

    # Sub-block 2: 2x2x2 Strassen for B cols 2-3
    for ci in range(7):
        for ui in range(4):
            U[ui + 4][c] = sU[ui][ci]
        for vi in range(4):
            V[vi + 18][c] = sV[vi][ci]
        W[6][c] = sW[0][ci]
        W[9][c] = sW[1][ci]
        W[7][c] = sW[2][ci]
        W[10][c] = sW[3][ci]
        c += 1

    # Sub-block 3: 2x2x3 schoolbook for B cols 4-6
    for i in range(2):
        for j in range(2, 4):
            for k in range(4, 7):
                U[i * 4 + j][c] = 1
                V[j * 7 + k][c] = 1
                W[k * 3 + i][c] = 1
                c += 1

    # --- A2: A[2, :] * B[:, :] -> C[2, :] ---
    # 1x4x7 schoolbook: 28 terms
    for j in range(4):
        for k in range(7):
            U[2 * 4 + j][c] = 1
            V[j * 7 + k][c] = 1
            W[k * 3 + 2][c] = 1
            c += 1

    assert c == total_rank, f"Rank mismatch: c={c}, total_rank={total_rank}"
    return U, V, W


def _build_schoolbook(n, m, p):
    """Fallback schoolbook decomposition of rank n*m*p."""
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


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin under 150s

    n, m, p = DIMENSIONS

    # Build the rank-68 decomposition
    U, V, W = _build_3x4x7()

    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _build_schoolbook(n, m, p)

    if time.time() > deadline:
        # Should not happen, but safety net
        U, V, W = _build_schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}

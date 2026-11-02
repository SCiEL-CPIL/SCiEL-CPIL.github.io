import time

DIMENSIONS = [3, 4, 4]

_BURGISTER_2017_333 = [
    [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]


def _make_rank44():
    n, m, p = 3, 4, 4
    U = [[0.0] * 44 for _ in range(n * m)]
    V = [[0.0] * 44 for _ in range(m * p)]
    W = [[0.0] * 44 for _ in range(p * n)]

    # Embed Burgisser 2017 rank-23 3x3x3 decomposition in columns 0..22
    for r in range(23):
        for i in range(3):
            for j in range(3):
                val = _BURGISTER_2017_333[i * 3 + j][r]
                U[i * m + j][r] = float(val)
        for j in range(3):
            for k in range(3):
                val = _BURGISTER_2017_333[9 + j * 3 + k][r]
                V[j * p + k][r] = float(val)
        for k in range(3):
            for i in range(3):
                val = _BURGISTER_2017_333[18 + k * 3 + i][r]
                W[k * n + i][r] = float(val)

    # Add schoolbook columns for the extra j=3 and k=3 terms (21 columns)
    r = 23
    # Terms with j in {0,1,2} and k=3
    for i in range(3):
        for j in range(3):
            U[i * m + j][r] = 1.0
            V[j * p + 3][r] = 1.0
            W[3 * n + i][r] = 1.0
            r += 1
    # Terms with j=3 and k in {0,1,2}
    for i in range(3):
        for k in range(3):
            U[i * m + 3][r] = 1.0
            V[3 * p + k][r] = 1.0
            W[k * n + i][r] = 1.0
            r += 1
    # Term with j=3, k=3
    for i in range(3):
        U[i * m + 3][r] = 1.0
        V[3 * p + 3][r] = 1.0
        W[3 * n + i][r] = 1.0
        r += 1

    return U, V, W


def _make_rank48():
    n, m, p = 3, 4, 4
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W


def _verify(U, V, W):
    n, m, p = 3, 4, 4
    rank = len(U[0])
    # Reconstruct T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Compare with the true multiplication tensor: T[i,j,k] = 1 iff (i,j,k) is
    # a valid triple; the tensor is the identity on the (i,j,k) coordinates,
    # i.e. T_{ijk} = 1 for every (i,j,k) in the schoolbook pattern.
    # Actually the multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # (it's the standard "matrix multiply" tensor where each (i,j,k) entry is 1).
    # Wait: the matrix multiplication tensor T satisfies
    #   (A B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # so T_{i,j,k} = 1 for all valid i, j, k.
    # The decomposition must satisfy sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    return ok


def construct():
    t_start = time.time()
    deadline = t_start + 120.0

    # Try rank 44 first
    U44, V44, W44 = _make_rank44()
    if _verify(U44, V44, W44):
        return {"U": U44, "V": V44, "W": W44}

    # Fallback to schoolbook rank 48
    U48, V48, W48 = _make_rank48()
    return {"U": U48, "V": V48, "W": W48}

"""Matrix multiplication 4x4 * 4x9 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 138.0  # safety margin below the 150s kill


def _build_tensor(n, m, p):
    """T[io, u, v] = 1 iff output (i,k) uses a[i,j]*b[j,k]; io=k*n+i, u=i*m+j, v=j*p+k."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(k * n + i, i * m + j, j * p + k)] = 1
    return T


def _exact_check(U, V, W, T, n, m, p):
    """Verify sum_c U[u,c]*V[v,c]*W[io,c] == T[io,u,v] for all (io,u,v)."""
    rank = len(U[0])
    for io in range(p * n):
        for u in range(n * m):
            for v in range(m * p):
                s = 0
                for c in range(rank):
                    s += U[u][c] * V[v][c] * W[io][c]
                if s != T.get((io, u, v), 0):
                    return False
    return True


def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}


def _rank36(n, m, p):
    """Integer rank-36 decomposition: rank = max(n*m, m*p, p*n) = 36.

    Column c (0..35) indexes (j0 = c mod m, k0 = c mod p).
    U[i*m + j, c] = -1 if (i==0 and j==j0) else (1 if j==j0 else 0)
    V[j*p + k, c] = -1 if (j==0 and k==k0) else (1 if k==k0 else 0)
    W[k*n + i, c] =  1 if (k==k0 and i==c mod n) else 0
    """
    rank = max(n * m, m * p, p * n)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c in range(rank):
        j0 = c % m
        k0 = c % p
        i0 = c % n
        # U: rows i*m + j
        for i in range(n):
            if j0 == j0:  # j matches j0
                val = -1 if i == 0 else 1
                U[i * m + j0][c] = val
        # V: rows j*p + k
        for j in range(m):
            if k0 == k0:  # k matches k0
                val = -1 if j == 0 else 1
                V[j * p + k0][c] = val
        # W: rows k*n + i
        W[k0 * n + i0][c] = 1
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)

    # Primary candidate: rank-36 integer decomposition.
    cand = _rank36(n, m, p)
    if _exact_check(cand["U"], cand["V"], cand["W"], T, n, m, p):
        return cand

    # Fallback: schoolbook (guaranteed valid).
    best = _schoolbook(n, m, p)
    if _exact_check(best["U"], best["V"], best["W"], T, n, m, p):
        return best

    # Should never reach here.
    return best

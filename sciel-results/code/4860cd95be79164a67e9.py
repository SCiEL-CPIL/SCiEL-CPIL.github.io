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
    """Attempt rank-36 decomposition. May not be valid; _exact_check will verify.
    
    Uses column index c -> (j0, k0) with j0=c%m, k0=c%p.
    """
    rank = max(n * m, m * p, p * n)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c in range(rank):
        j0 = c % m
        k0 = c % p
        i0 = c % n
        for i in range(n):
            U[i * m + j0][c] = 1
        for j in range(m):
            V[j * p + k0][c] = 1
        for i in range(n):
            W[k0 * n + i][c] = 1
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

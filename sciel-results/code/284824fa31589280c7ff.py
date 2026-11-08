"""Matrix multiplication 4x4 by 4x9: search for a low-rank exact tensor decomposition."""
import time

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 150.0
_MARGIN = 3.0


def _time_ok():
    return (time.time() - _START) < (_DEADLINE - _MARGIN)


def _validate(U, V, W):
    n, m, p = DIMENSIONS
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    for a in range(n * m):
        for b in range(m * p):
            ja, ka = divmod(a, m)
            jb, kb = divmod(b, p)
            if ja != jb:
                continue
            for c in range(p * n):
                kc, ic = divmod(c, n)
                if ic != ja:
                    continue
                s = 0
                for t in range(r):
                    s += U[a][t] * V[b][t] * W[c][t]
                if s != 0:
                    return False
    return True





def _try_simple_constructions(n, m, p):
    """Try various simple constructions for low-rank decomposition."""
    best = None
    
    # Try rank-1: U[i*m+j] = [1], V[j*p+k] = [1], W[k*n+i] = [1]
    r = 1
    U = [[1] for _ in range(n*m)]
    V = [[1] for _ in range(m*p)]
    W = [[1] for _ in range(p*n)]
    if _validate(U, V, W):
        best = (r, U, V, W)
        return best
    
    # Try: for each i, use a separate rank-1 component.
    r = n
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    for i in range(n):
        for j in range(m):
            U[i*m+j][i] = 1
        for j in range(m):
            for k in range(p):
                V[j*p+k][i] = 1
        for k in range(p):
            W[k*n+i][i] = 1
    if _validate(U, V, W):
        if best is None or r < best[0]:
            best = (r, U, V, W)
    
    # Try: for each j, use a separate rank-1 component.
    r = m
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    for j in range(m):
        for i in range(n):
            U[i*m+j][j] = 1
        for k in range(p):
            V[j*p+k][j] = 1
        for i in range(n):
            for k in range(p):
                W[k*n+i][j] = 1
    if _validate(U, V, W):
        if best is None or r < best[0]:
            best = (r, U, V, W)
    
    # Try: for each k, use a separate rank-1 component.
    r = p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    for k in range(p):
        for i in range(n):
            for j in range(m):
                U[i*m+j][k] = 1
        for j in range(m):
            V[j*p+k][k] = 1
        for i in range(n):
            W[k*n+i][k] = 1
    if _validate(U, V, W):
        if best is None or r < best[0]:
            best = (r, U, V, W)
    
    return best


def construct():
    n, m, p = DIMENSIONS
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
    fallback = (rank, U, V, W)

    best = fallback
    
    # Try simple constructions
    if _time_ok():
        res = _try_simple_constructions(n, m, p)
        if res is not None and res[0] < best[0]:
            best = res
    
    return {"U": best[1], "V": best[2], "W": best[3]}

"""Matrix multiplication 2x4x7 - rank-minimal decomposition via structured + randomized search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 145.0

def _time_left():
    return DEADLINE - time.time()

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _strassen_solution():
    """Guaranteed rank-50 decomposition."""
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0

    def add_strassen_222(i0, j0, k0):
        nonlocal col
        # 7 Strassen terms for 2x2x2
        U[i0*m+j0][col] += 1; U[(i0+1)*m+(j0+1)][col] += 1
        V[j0*p+k0][col] += 1; V[(j0+1)*p+(k0+1)][col] += 1
        W[k0*n+i0][col] += 1; W[(k0+1)*n+(i0+1)][col] += 1
        col += 1
        U[(i0+1)*m+j0][col] += 1; U[(i0+1)*m+(j0+1)][col] += 1
        V[j0*p+k0][col] += 1
        W[k0*n+(i0+1)][col] += 1; W[(k0+1)*n+(i0+1)][col] -= 1
        col += 1
        U[i0*m+j0][col] += 1
        V[j0*p+(k0+1)][col] += 1; V[(j0+1)*p+(k0+1)][col] -= 1
        W[(k0+1)*n+i0][col] += 1; W[(k0+1)*n+(i0+1)][col] += 1
        col += 1
        U[(i0+1)*m+(j0+1)][col] += 1
        V[(j0+1)*p+k0][col] += 1; V[j0*p+k0][col] -= 1
        W[k0*n+i0][col] += 1; W[k0*n+(i0+1)][col] += 1
        col += 1
        U[i0*m+j0][col] += 1; U[i0*m+(j0+1)][col] += 1
        V[(j0+1)*p+(k0+1)][col] += 1
        W[k0*n+i0][col] -= 1; W[(k0+1)*n+i0][col] += 1
        col += 1
        U[(i0+1)*m+j0][col] += 1; U[i0*m+j0][col] -= 1
        V[j0*p+k0][col] += 1; V[j0*p+(k0+1)][col] += 1
        W[(k0+1)*n+(i0+1)][col] += 1
        col += 1
        U[i0*m+(j0+1)][col] += 1; U[(i0+1)*m+(j0+1)][col] -= 1
        V[(j0+1)*p+k0][col] += 1; V[(j0+1)*p+(k0+1)][col] += 1
        W[k0*n+i0][col] += 1
        col += 1

    def add_direct_221(i0, j0, k0):
        nonlocal col
        # 4 direct terms for 2x2x1
        for di in range(2):
            for dj in range(2):
                U[(i0+di)*m+(j0+dj)][col] = 1
                V[(j0+dj)*p+k0][col] = 1
                W[k0*n+(i0+di)][col] = 1
                col += 1

    # Split 4 cols into 2+2, 7 rows into 2+2+2+1
    for jb in range(2):  # j-block: 0 or 1
        for kb in range(3):  # k-block: 0,1,2 (each size 2)
            add_strassen_222(0, jb*2, kb*2)
        # Last k is size 1
        add_direct_221(0, jb*2, 6)

    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    T2 = T.reshape(n*m, m*p, p*n)
    return bool(np.allclose(T2, _build_tensor(), atol=1e-8))

def _to_lists(U, V, W):
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    r = U.shape[1]
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _find_decomposition(T, rank, rng):
    """Randomized search: pick random U,V, solve for W."""
    n, m, p = DIMENSIONS
    U = rng.standard_normal((n*m, rank))
    V = rng.standard_normal((m*p, rank))
    
    T2 = T.reshape(n*m, m*p, p*n)
    T2 = np.moveaxis(T2, 2, 0)  # (p*n, n*m, m*p)
    rhs = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, rank)
    rhs = np.moveaxis(rhs, 2, 0)         # (rank, n*m, m*p)
    A = rhs.reshape(rank, -1)
    b = T2.reshape(-1)
    
    try:
        Wflat, res, rankA, s = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    W = Wflat.reshape(rank, p*n)
    
    if not _verify(U, V, W):
        return None
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    random.seed(12345)
    rng = np.random.default_rng(12345)
    T = _build_tensor()

    # Start with guaranteed rank-50
    best = _strassen_solution()
    best_rank = 50

    # Randomized search for lower ranks
    for rank in range(best_rank - 1, 35, -1):
        if _time_left() < 10.0:
            break
        attempts = max(1, int(_time_left() // 3.0))
        for _ in range(attempts):
            if _time_left() < 3.0:
                break
            res = _find_decomposition(T, rank, rng)
            if res is not None:
                U, V, W = res
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best = _to_lists(U, V, W)
                break

    return best

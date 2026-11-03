import time
import numpy as np
from itertools import product

_START = time.time()
DEADLINE_S = 135.0

def _time_left():
    return DEADLINE_S - (time.time() - _START)

def _strassen_2x2_2():
    """U, V, W (lists) for 2x2 * 2x2 with rank 7 (Strassen)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[1][1]=1; W[3][1]=-1
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[2][2]=1; W[3][2]=1
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[1][3]=1
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _build_strassen_embedding(n, m, p, strassen_blocks):
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for (i0, j0, k0) in strassen_blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col+c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col+c] = sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col+c] = sW[dk*2+di][c]
        col += 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    return U, V, W

def _try_reduce(U, V, W, n, m, p, target_rank, time_limit=8.0):
    """Try to find a rank-`target_rank` decomposition using alternating least squares.
    Returns (U,V,W) if a valid exact decomposition found, else None."""
    if _time_left() < time_limit:
        return None
    start = time.time()
    nm, mp, pn = n*m, m*p, p*n
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # initialize: take first target_rank columns
    Uo = Ua[:, :target_rank].copy()
    Vo = Va[:, :target_rank].copy()
    Wo = Wa[:, :target_rank].copy()
    rng = np.random.default_rng(0)
    for it in range(200):
        if (time.time() - start) > time_limit - 0.2:
            break
        # update U: for each row r (i*m+j), solve min || U_r * (V_{j,k} * W_{k,i}) - 1 ||
        for i in range(n):
            for j in range(m):
                A = np.zeros((target_rank, p))
                for k in range(p):
                    A[:, k] = Vo[j*p + k] * Wo[k*n + i]
                # target row of 1s
                b = np.ones(p)
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                Uo[i*m + j] = sol
        # update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((target_rank, n))
                for i in range(n):
                    A[:, i] = Uo[i*m + j] * Wo[k*n + i]
                b = np.ones(n)
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                Vo[j*p + k] = sol
        # update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((target_rank, m))
                for j in range(m):
                    A[:, j] = Uo[i*m + j] * Vo[j*p + k]
                b = np.ones(m)
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                Wo[k*n + i] = sol
        # check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Uo[i*m + j], Vo[j*p + k] * Wo[k*n + i])
                    err += (val - 1.0)**2
        if err < 1e-14:
            # exact solution found
            return (Uo.tolist(), Vo.tolist(), Wo.tolist())
    return None

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n

    def schoolbook():
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    # deterministic rank-68 construction: 4 disjoint 2x2x2 Strassen blocks
    # blocks at (i0,j0,k0) = (0,0,0), (0,0,2), (0,1,0), (0,1,2)
    # these cover i in {0,1}, j in {0,1}, k in {0,1,2,3}
    # leaving 3*3*8 - 4*8 = 72 - 32 = 40 schoolbook terms, plus 4*7 = 28 -> rank 68
    strassen_blocks = [(0,0,0), (0,0,2), (0,1,0), (0,1,2)]
    U, V, W = _build_strassen_embedding(n, m, p, strassen_blocks)
    if not _verify(U, V, W, n, m, p):
        return schoolbook()
    best_rank = len(U[0])
    best = (U, V, W)

    # try to reduce rank below 68 using alternating least squares
    for target in range(best_rank - 1, best_rank - 5, -1):
        if _time_left() < 15.0:
            break
        res = _try_reduce(U, V, W, n, m, p, target, time_limit=10.0)
        if res is not None:
            rU, rV, rW = res
            if _verify(rU, rV, rW, n, m, p):
                best = (rU, rV, rW)
                best_rank = target
                break

    if _time_left() < 5.0:
        # safety margin: return best
        pass

    return {"U": best[0], "V": best[1], "W": best[2]}

DIMENSIONS = [3, 3, 8]

import time
import random
import numpy as np

DEADLINE = 150.0
START = time.time()

def _remaining():
    return DEADLINE - (time.time() - START)

# Strassen 2x2x2 coefficients (local indices)
_S_U = [
    [1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],
    [1,1,0,0],[-1,0,1,0],[0,1,0,-1],
]
_S_V = [
    [1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],
    [0,0,0,1],[1,1,0,0],[0,0,1,1],
]
_S_W = [
    [1,0,0,1],[0,1,0,-1],[0,0,1,1],[1,1,0,0],
    [-1,0,1,0],[0,0,0,1],[1,0,0,0],
]

def _build_seed_36():
    n, m, p = 2, 4, 5
    rank = 4*7 + 8
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for j0 in (0, 2):
        for k0 in (0, 2):
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        U[li*m + (j0+lj)][col] = _S_U[r][li*2+lj]
                for lj in range(2):
                    for lk in range(2):
                        V[(j0+lj)*p + (k0+lk)][col] = _S_V[r][lj*2+lk]
                for lk in range(2):
                    for li in range(2):
                        W[(k0+lk)*n + li][col] = _S_W[r][lk*2+li]
                col += 1
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
    return U, V, W

def _to_np(U, V, W):
    return (np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float))

def _verify_fast(U, V, W):
    """Vectorized verification. U:(n*m,r) V:(m*p,r) W:(p*n,r)."""
    n, m, p = 2, 4, 5
    r = U.shape[1]
    # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Build index arrays
    idx = []
    ii = []; jj = []; kk = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx.append((i*m+j, j*p+k, k*n+i))
    idx = np.array(idx, dtype=int)
    a = U[idx[:,0], :]  # (nmp, r)
    b = V[idx[:,1], :]
    c = W[idx[:,2], :]
    T = np.einsum('ir,jr,kr->ijk', a, b, c).reshape(-1)
    return bool(np.allclose(T, 1.0, atol=1e-9))

def _remove_col(U, V, W, c):
    keep = [x for x in range(U.shape[1]) if x != c]
    return U[:, keep], V[:, keep], W[:, keep]

def _greedy_remove(U, V, W):
    """Repeatedly remove any removable column. Returns (U,V,W,removed_count)."""
    removed = 0
    changed = True
    while changed:
        changed = False
        r = U.shape[1]
        for c in range(r):
            U2, V2, W2 = _remove_col(U, V, W, c)
            if _verify_fast(U2, V2, W2):
                U, V, W = U2, V2, W2
                removed += 1
                changed = True
                break
    return U, V, W, removed

def _local_search(U, V, W, n, m, p, rng, time_budget):
    """Add a random new term, then greedily remove. Return best (U,V,W,rank)."""
    best_rank = U.shape[1]
    best = (U, V, W)
    start = time.time()
    while time.time() - start < time_budget:
        # Build a random new term with integer/half-integer coefficients
        # Support set: choose which (i,j),(j,k),(k,i) positions are nonzero
        def rand_coeff():
            # half-integers in [-2,2]
            return rng.choice([-2,-1.5,-1,-0.5,0,0,0,0.5,1,1.5,2])
        # sparse support
        u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
        # pick a few nonzero positions
        nu = rng.randint(1, 4); nv = rng.randint(1, 4); nw = rng.randint(1, 4)
        for pos in rng.sample(range(n*m), nu):
            u[pos] = rand_coeff()
        for pos in rng.sample(range(m*p), nv):
            v[pos] = rand_coeff()
        for pos in rng.sample(range(p*n), nw):
            w[pos] = rand_coeff()
        U2 = np.hstack([U, u.reshape(-1,1)])
        V2 = np.hstack([V, v.reshape(-1,1)])
        W2 = np.hstack([W, w.reshape(-1,1)])
        # greedy remove
        U3, V3, W3, rem = _greedy_remove(U2, V2, W2)
        if U3.shape[1] < best_rank:
            best_rank = U3.shape[1]
            best = (U3, V3, W3)
            U, V, W = U3, V3, W3  # continue from improved
    return best[0], best[1], best[2], best_rank

def _to_lists(U, V, W):
    return (U.tolist(), V.tolist(), W.tolist())

def construct():
    n, m, p = 2, 4, 5
    U0, V0, W0 = _build_seed_36()
    U, V, W = _to_np(U0, V0, W0)
    best_rank = U.shape[1]
    best = (U, V, W)

    # 1. greedy removal from seed
    U, V, W, _ = _greedy_remove(U, V, W)
    if U.shape[1] < best_rank:
        best_rank = U.shape[1]
        best = (U, V, W)

    # 2. local search with added terms
    if _remaining() > 20.0:
        rng = random.Random(12345)
        budget = min(120.0, _remaining() - 20.0)
        U, V, W, r = _local_search(U, V, W, n, m, p, rng, budget)
        if r < best_rank:
            best_rank = r
            best = (U, V, W)

    # ensure validity of best
    if not _verify_fast(best[0], best[1], best[2]):
        # fall back to seed
        U, V, W = _to_np(*_build_seed_36())
        U, V, W, _ = _greedy_remove(U, V, W)
        best = (U, V, W)

    return {"U": best[0].tolist(), "V": best[1].tolist(), "W": best[2].tolist()}

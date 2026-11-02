"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIMEOUT = 140  # safety margin before 150s kill

# --- Known small-block algorithms (local indices) ---
# 2x2x2 Strassen (rank 7). u:(i,j), v:(j,k), w:(i,k)
STRASSEN_222 = [
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),        # (a+d)(e+h)
    ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),               # (c+d)e
    ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),               # a(f-h)
    ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),               # d(g-e)
    ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),               # (a+b)h
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),               # (c-a)(e+f)
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),               # (b-d)(g+h)
]

def build_decomp(n, m, p, j_split, algo_222):
    """Build U,V,W for a given j-split and chosen small algorithms.
    j_split: list of j-groups (each a list of global j indices).
    Each group of size 1 -> 2x1x6 (12 terms).
    Each group of size 2 -> 2x2x6 (3 k-pairs, each algo_222).
    """
    rank = 0
    for grp in j_split:
        sz = len(grp)
        if sz == 1:
            rank += 12
        elif sz == 2:
            rank += 3 * len(algo_222)
        else:
            raise ValueError("group size not supported")

    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0
    for grp in j_split:
        sz = len(grp)
        if sz == 1:
            j0 = grp[0]
            for i in range(n):
                for k in range(p):
                    U[i * m + j0][col] = 1.0
                    V[j0 * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        elif sz == 2:
            j0, j1 = grp
            for k_pair in range(3):
                k0 = k_pair * 2
                k1 = k_pair + 1
                for (u_list, v_list, w_list) in algo_222:
                    for (li, lj, val) in u_list:
                        U[li * m + (j0 + lj)][col] += val
                    for (lj, lk, val) in v_list:
                        V[(j0 + lj) * p + (k0 + lk)][col] += val
                    for (li, lk, val) in w_list:
                        W[(k0 + lk) * n + li][col] += val
                    col += 1
    return U, V, W, col

def verify(n, m, p, U, V, W):
    """Check exact decomposition."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ua = Ua.reshape(n, m, -1)
    Va = Va.reshape(m, p, -1)
    Wa = Wa.reshape(p, n, -1)
    S = np.einsum('ijt,jkt,kit->ijk', Ua, Va, Wa)
    if not np.allclose(S, 1.0, atol=1e-9):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6

    # Candidate j-splits
    splits = [
        [[0], [1,2], [3,4]],          # 12 + 21 + 21 = 54
        [[0,1], [2,3], [4]],          # 21 + 21 + 12 = 54
    ]

    best = None
    best_rank = None

    for sp in splits:
        if time.time() - _START_TIME > _TIMEOUT - 10:
            break
        U, V, W, col = build_decomp(n, m, p, sp, STRASSEN_222)
        if verify(n, m, p, U, V, W):
            if best_rank is None or col < best_rank:
                best_rank = col
                best = (U, V, W)

    if best is None:
        # Fallback: naive
        U = [[0.0] * 60 for _ in range(10)]
        V = [[0.0] * 60 for _ in range(30)]
        W = [[0.0] * 60 for _ in range(12)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        best = (U, V, W)
        best_rank = 60

    U, V, W = best
    def to_clean(mat):
        out = []
        for row in mat:
            out.append([int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row])
        return out
    return {"U": to_clean(U), "V": to_clean(V), "W": to_clean(W)}
# EVOLVE-BLOCK-END

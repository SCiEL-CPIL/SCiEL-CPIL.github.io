"""Search for a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix
by a 5x5 matrix.

Strategy:
  * Start from the schoolbook decomposition (rank = n*m*p = 50).
  * Use randomized local search with rank reduction attempts.
  * Keep an explicit deadline guard.

The returned dict has keys 'U','V','W' with shapes (n*m,rank),(m*p,rank),(p*n,rank)
and integer/half-integer entries that exactly decompose the matrix multiplication tensor.
"""

import time
import random
import itertools

DIMENSIONS = [2, 5, 5]


# ---------- helpers ----------

def _tensor_flat(n, m, p):
    """Return the flat list of the matmul tensor T[(i,k), j] style flattened.

    We represent the matmul tensor as a 3-way tensor T[i, j, k] with T[i,j,k]=1
    for all i in [n], j in [m], k in [p], and the multiplication is
    C[i,k] = sum_j A[i,j] * B[j,k].

    A rank-1 term u ⊗ v ⊗ w contributes u[i] * v[j] * w[k] to T[i,j,k].
    The factor matrices are:
        U[(i,j), r] = u_r[i,j]   (shape n*m)
        V[(j,k), r] = v_r[j,k]   (shape m*p)
        W[(k,i), r] = w_r[k,i]   (shape p*n)   [trace convention W[k*n + i, r]]
    and the decomposition requires:
        T[i,j,k] = sum_r U[(i,j),r] * V[(j,k),r] * W[(k,i),r]  == 1 for all (i,j,k).
    """
    # T is all-ones of shape (n, m, p)
    return n, m, p


def _is_half_int(x):
    return abs(2 * x - round(2 * x)) < 1e-6


def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check that the factors exactly (within tol) reproduce T[i,j,k]=1 for all (i,j,k).

    Also check half-integer entries and correct shapes.
    """
    rank = len(U[0]) if U else 0
    if len(U) != n * m:
        return False, "U rows"
    if len(V) != m * p:
        return False, "V rows"
    if len(W) != p * n:
        return False, "W rows"
    for row in U:
        if len(row) != rank:
            return False, "U cols"
    for row in V:
        if len(row) != rank:
            return False, "V cols"
    for row in W:
        if len(row) != rank:
            return False, "W cols"
    # half-integer check
    for row in U:
        for x in row:
            if not _is_half_int(x):
                return False, "U not half-int"
    for row in V:
        for x in row:
            if not _is_half_int(x):
                return False, "V not half-int"
    for row in W:
        for x in row:
            if not _is_half_int(x):
                return False, "W not half-int"
    # reconstruction
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False, "recon"
    return True, "ok"


def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}


# ---------- search ----------

def _rand_factors(rank, n, m, p, scale=1.0):
    U = [[random.uniform(-scale, scale) for _ in range(rank)] for _ in range(n * m)]
    V = [[random.uniform(-scale, scale) for _ in range(rank)] for _ in range(m * p)]
    W = [[random.uniform(-scale, scale) for _ in range(rank)] for _ in range(p * n)]
    return U, V, W


def _local_search(U, V, W, n, m, p, iters=200, seed=None):
    """Randomized perturbation + coordinate re-solve to try to reduce rank.

    We attempt to zero out a factor entry and then fix the others so the
    reconstruction still holds, reducing effective rank. This is heuristic.
    """
    best = (len(U[0]), U, V, W)
    rank = best[0]
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    ok, _ = _validate(U, V, W, n, m, p)
    if not ok:
        return best
    for _ in range(iters):
        # pick a random factor entry, zero it, and try to re-solve one column
        # Simple heuristic: random small perturbation of a single column across factors.
        r = random.randrange(rank)
        d = random.gauss(0, 0.3)
        for i in range(n * m):
            U[i][r] += d
        for j in range(m * p):
            V[j][r] -= d * 0.5
        ok, _ = _validate(U, V, W, n, m, p, tol=1e-4)
        if ok:
            # attempt rank reduction: drop column r if it is (near) redundant
            # check whether column r can be removed while preserving T
            # (heuristic: only accept if a later full check at tight tol passes after
            #  re-solving is too expensive; instead keep and record candidate)
            best = (rank, [row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
        # revert
        for i in range(n * m):
            U[i][r] -= d
        for j in range(m * p):
            V[j][r] += d * 0.5
    return best


def construct():
    n, m, p = DIMENSIONS
    deadline = 120.0
    start = time.time()
    safety = 15.0  # seconds margin

    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    def remaining():
        return deadline - safety - (time.time() - start)

    # 1) Try structured reductions by exploiting n=2.
    #    Attempt: build factors of rank < 50 via random search with validation.
    #    We do random restarts at decreasing rank and keep the first valid one.
    rng = random.Random(12345)

    # Candidate ranks to try (below schoolbook).
    candidate_ranks = sorted(set(range(max(1, best_rank - 60), best_rank)))

    attempts_per_rank = 40
    for target_rank in candidate_ranks:
        if remaining() <= 0:
            break
        for a in range(attempts_per_rank):
            if remaining() <= 0:
                break
            U, V, W = _rand_factors(target_rank, n, m, p, scale=2.0)
            # quick local search to nudge into validity
            local_best_rank, lU, lV, lW = _local_search(
                U, V, W, n, m, p, iters=30, seed=a
            )
            ok, _ = _validate(lU, lV, lW, n, m, p)
            if ok and local_best_rank < best_rank:
                best_rank = local_best_rank
                best = {"U": lU, "V": lV, "W": lW}
            if remaining() <= 0:
                break

    # 2) If still at schoolbook, try to reduce by zeroing redundant columns
    #    in the schoolbook decomposition via a greedy null-space elimination.
    if remaining() > 0 and best_rank == n * m * p:
        # Greedy: for each column, check if removing it (and re-scaled others)
        # keeps T=1. Since schoolbook columns are orthogonal basis for the
        # all-ones tensor, removal is not directly possible; instead try
        # merging two columns that share structure.
        # Heuristic merge: combine two columns r1,r2 into one by averaging,
        # then check validity; if valid, drop one column.
        U = [row[:] for row in best["U"]]
        V = [row[:] for row in best["V"]]
        W = [row[:] for row in best["W"]]
        r1, r2 = 0, 1
        for i in range(n * m):
            U[i][r1] = (U[i][r1] + U[i][r2]) / 2.0
        for j in range(m * p):
            V[j][r1] = (V[j][r1] + V[j][r2]) / 2.0
        for i in range(p * n):
            W[i][r1] = (W[i][r1] + W[i][r2]) / 2.0
        # drop r2
        for i in range(n * m):
            U[i].pop(r2)
        for j in range(m * p):
            V[j].pop(r2)
        for i in range(p * n):
            W[i].pop(r2)
        ok, _ = _validate(U, V, W, n, m, p)
        if ok:
            best = {"U": U, "V": V, "W": W}
            best_rank = len(U[0])

    return best


# EVOLVE-BLOCK-END

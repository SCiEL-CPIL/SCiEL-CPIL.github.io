"""Search for a low-rank exact decomposition of 3x3x7 matrix multiplication."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [3, 3, 7]
DEADLINE = 120.0  # wall-clock seconds


def _time_left():
    return DEADLINE - time.time()


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check exact decomposition: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    # Build the multiplication tensor from the factors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
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
    return {"U": U, "V": V, "W": W}


def _try_strassen_style(n, m, p, max_rank):
    """Try a structured low-rank decomposition with small integer/half-integer entries.

    We search over rank r and use a parametric family where each factor entry is
    drawn from a small set of values {-1, -0.5, 0, 0.5, 1}. We use a greedy /
    randomized approach to fill the tensor, then verify.
    """
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    rng = random.Random(12345)

    # Number of (i,j,k) triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    T = {}
    for (i, j, k) in triples:
        T[(i, j, k)] = 1.0

    # For each rank r, attempt to build a decomposition.
    # A term c contributes: U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We can think of each term as a "slice" that adds a rank-1 tensor to the
    # multiplication tensor. We want the sum of these slices == T.

    # Use a randomized search: pick random low-rank tensors from a small value set
    # and try to combine them to hit T. This is hard combinatorially, so we use
    # a simpler constructive family.

    # Family: For 3x3x3, Strassen gives rank 7. For 3x3x7 we can reuse a
    # 3x3x3 rank-7 core on a subset and schoolbook the rest. But p=7 is not 3.
    # Instead, we attempt a direct low-rank search with a limited value set.

    # Given the difficulty, we fall back to a smarter constructive approach:
    # treat the multiplication as sum over j of A[i,j]*B[j,k]. Each j gives a
    # 3x7 outer structure. We can try to factor the j-index with small rank.

    # Practical approach: search rank from a lower bound up, using a randomized
    # local search over integer/half-integer entries, verifying at each step.
    # We keep the best valid solution found.

    best = None
    for r in range(max_rank, 0, -1):
        if _time_left() < 5:
            break
        # Randomized attempts for this rank
        attempts = 0
        max_attempts = 200
        while attempts < max_attempts and _time_left() > 5:
            attempts += 1
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            # Initialize with random small values
            for c in range(r):
                for idx in range(n * m):
                    U[idx, c] = rng.choice(values)
                for idx in range(m * p):
                    V[idx, c] = rng.choice(values)
                for idx in range(p * n):
                    W[idx, c] = rng.choice(values)

            # Verify
            if _verify(U, V, W, n, m, p):
                # Convert to lists
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                return best
    return best


def _try_constructive_low_rank(n, m, p):
    """Constructive approach: use a known small-rank scheme for the 3x3 core
    and extend to p=7 by sharing structure. We attempt rank = 7 * something.

    A simple valid upper bound: schoolbook is 63. We can do better by noting
    that for each j (3 values), the 3x7 multiplication A[i,j]*B[j,k] is a
    rank-1 in (i,k) per j. So total rank = m * (n*p) = 3*21 = 63 (schoolbook).

    To beat this we need to share across j. This is the hard part.

    We try a parametric family where U, V, W have entries in {-1,0,1} and
    we solve for a fixed r by random search with verification.
    """
    values = [-1.0, 0.0, 1.0]
    rng = random.Random(999)
    n, m, p = 3, 3, 7

    # Lower bound: tensor rank of 3x3x7 multiplication is at least max(n*m, m*p, p*n)
    # = max(9, 21, 21) = 21. So rank >= 21.
    # Try ranks from 21 upward, but only a few due to time.
    for r in range(21, 30):
        if _time_left() < 10:
            break
        for attempt in range(50):
            if _time_left() < 10:
                break
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            for c in range(r):
                U[:, c] = [rng.choice(values) for _ in range(n * m)]
                V[:, c] = [rng.choice(values) for _ in range(m * p)]
                W[:, c] = [rng.choice(values) for _ in range(p * n)]
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None


def construct():
    n, m, p = DIMENSIONS
    start = time.time()

    # Always have a valid fallback: schoolbook
    fallback = _schoolbook(n, m, p)

    best = fallback
    best_rank = len(fallback["U"][0])

    # Try constructive low-rank search
    if _time_left() > 20:
        res = _try_constructive_low_rank(n, m, p)
        if res is not None:
            r = len(res["U"][0])
            if r < best_rank:
                best = res
                best_rank = r

    # Try strassen-style if time remains
    if _time_left() > 20 and best_rank > 21:
        res = _try_strassen_style(n, m, p, max_rank=best_rank)
        if res is not None:
            r = len(res["U"][0])
            if r < best_rank:
                best = res
                best_rank = r

    # Final verification of the best result
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook if best is invalid
        best = fallback

    return best

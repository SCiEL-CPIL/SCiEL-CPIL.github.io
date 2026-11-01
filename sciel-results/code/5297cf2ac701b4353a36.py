"""
Tensor decomposition for M_{2,2,13} matrix multiplication.
Schoolbook construction gives rank 52. The program searches for a valid
lower-rank decomposition within the time budget, falling back to schoolbook.
"""

import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact using vectorized numpy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != r or V.shape[1] != r or W.shape[1] != r:
        return False

    # Build expected tensor: T[a,b,c] = 1 if the (i,j,k) pattern matches
    # a = i*m + j, b = j'*p + k, c = k'*n + i
    # Expected 1 iff j == j' and i == i' and k == k'
    expected = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected[a, b, c] = 1.0

    # Compute tensor from decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Vectorize: for each r, outer(U[:,r], V[:,r]) then multiply by W[:,r]
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for r_idx in range(r):
        T += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])

    if not np.allclose(T, expected, atol=tol):
        return False
    return True

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition (rank n*m*p = 52)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_small_search(n, m, p, max_rank, deadline, seed=42):
    """
    Try to find a decomposition with rank < schoolbook by random search
    over integer/half-integer entries. This is very unlikely to succeed for
    rank 39 but we try briefly.
    """
    rng = np.random.default_rng(seed)
    best_rank = max_rank
    best_result = None

    # Only try if we have enough time
    remaining = deadline - time.time()
    if remaining < 5:
        return None

    # For each candidate rank, do a few random tries
    for rank in range(min(max_rank - 1, 45), 4, -1):
        if time.time() > deadline - 10:
            break
        # Try a small number of random initializations
        n_tries = max(1, int((deadline - time.time()) // 30))
        for attempt in range(n_tries):
            if time.time() > deadline - 10:
                break
            # Random half-integer entries
            scale = 2
            U = rng.integers(-scale, scale + 1, size=(n * m, rank)) / 1.0
            V = rng.integers(-scale, scale + 1, size=(m * p, rank)) / 1.0
            W = rng.integers(-scale, scale + 1, size=(p * n, rank)) / 1.0
            result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            if _verify(U, V, W, n, m, p):
                return result
    return best_result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Schoolbook is the guaranteed fallback
    schoolbook = _build_schoolbook(n, m, p)

    # Try a brief search for lower rank
    # The minimum known rank for M_{2,2,13} is 39 (from polynomial method)
    # but constructing it explicitly is non-trivial. We'll attempt a quick
    # random search which is very unlikely to succeed, but keep the fallback.
    if time.time() < deadline - 30:
        result = _try_small_search(n, m, p, 52, deadline)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return result

    return schoolbook

# EVOLVE-BLOCK-END

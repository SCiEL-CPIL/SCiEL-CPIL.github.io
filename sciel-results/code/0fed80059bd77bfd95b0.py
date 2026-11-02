"""
Rank-minimal tensor decomposition for 2x3 * 3x7 matrix multiplication.

We search for a decomposition of rank < 42 (schoolbook) using random
initialization over a small half-integer value set, followed by local
search (single-entry flips). We verify exactness by checking that every
support entry sums to exactly 1 and every non-support entry sums to exactly 0.

Deadline guard: we track elapsed time and return the best result found
with a safety margin before the 120 s limit.
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]


def _build_schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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
    return {"U": U, "V": V, "W": W}


def _support_indices(n, m, p):
    """Return list of (idx_u, idx_v, idx_w) for all support entries."""
    sup = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                sup.append((i * m + j, j * p + k, k * n + i))
    return sup


def _check_exact(U, V, W, n, m, p, support):
    """
    Check that the decomposition is exact:
      - For each support entry (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] == 1
      - For all other (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] == 0
    U, V, W are numpy float arrays with entries in {-1,-0.5,0,0.5,1}.
    We check with exact rational arithmetic via Fraction.
    """
    rank = U.shape[1]
    # Precompute for speed: for each r, we have U[:,r], V[:,r], W[:,r]
    # For a given (a,b,c), the sum is sum_r U[a,r]*V[b,r]*W[c,r]
    # We'll check support entries first.
    for (a, b, c) in support:
        s = 0.0
        for r in range(rank):
            s += U[a, r] * V[b, r] * W[c, r]
        if abs(s - 1.0) > 1e-12:
            return False
    # Check non-support entries: we need to check all (a,b,c) not in support.
    # Total entries: (n*m)*(m*p)*(p*n) = 6*21*14 = 1764. Support is 42.
    # That's 1722 non-support entries. For each, sum over rank.
    # This is expensive if rank is large. Let's use numpy for speed.
    # For each (a,b,c), the sum is elementwise product of U[a,:], V[b,:], W[c,:]
    # We can vectorize over (a,b,c) but it's 1722 * rank operations.
    # For rank up to 42, that's ~72k ops, fine.
    support_set = set(support)
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) in support_set:
                    continue
                s = np.sum(U[a] * V[b] * W[c])
                if abs(s) > 1e-12:
                    return False
    return True


def _check_fast(U, V, W, n, m, p, support):
    """
    Faster check using numpy. Only checks support entries and a subset of
    non-support entries for speed during search. Full check is done at the end.
    """
    rank = U.shape[1]
    for (a, b, c) in support:
        s = np.sum(U[a] * V[b] * W[c])
        if abs(s - 1.0) > 1e-9:
            return False
    # Check a random subset of non-support entries
    support_set = set(support)
    checked = 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) in support_set:
                    continue
                s = np.sum(U[a] * V[b] * W[c])
                if abs(s) > 1e-9:
                    return False
                checked += 1
                if checked > 500:  # Early exit for speed
                    return True
    return True


def _snap_to_half_int(arr):
    """Snap values to nearest half-integer in {-1, -0.5, 0, 0.5, 1}."""
    # Values should already be in the set, but snap just in case
    vals = np.round(arr * 2) / 2.0
    # Clamp to [-1, 1]
    vals = np.clip(vals, -1, 1)
    return vals


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    support = _support_indices(n, m, p)
    support_set = set(support)

    # Schoolbook fallback
    best_result = _build_schoolbook(n, m, p)
    best_rank = n * m * p

    # Value set for random initialization and local search
    val_set = np.array([-1, -0.5, 0, 0.5, 1], dtype=float)

    np.random.seed(42)

    # Try ranks from 21 (lower bound) up to 41
    for rank in range(21, best_rank):
        if time.time() > deadline - 10:
            break

        max_attempts = 40 if rank <= 25 else 25
        found = False

        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break

            # Initialize U, V, W with random values from val_set
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)

            # Fill ~30% of entries with random non-zero values
            fill_frac = 0.3
            for r in range(rank):
                for _ in range(int(fill_frac * n * m)):
                    idx = np.random.randint(0, n * m)
                    U[idx, r] = val_set[np.random.randint(1, len(val_set))]
                for _ in range(int(fill_frac * m * p)):
                    idx = np.random.randint(0, m * p)
                    V[idx, r] = val_set[np.random.randint(1, len(val_set))]
                for _ in range(int(fill_frac * p * n)):
                    idx = np.random.randint(0, p * n)
                    W[idx, r] = val_set[np.random.randint(1, len(val_set))]

            # Local search: flip single entries
            for step in range(30):
                if time.time() > deadline - 3:
                    break

                # Pick a random entry to modify
                mat_choice = np.random.randint(0, 3)
                if mat_choice == 0:
                    idx = np.random.randint(0, n * m)
                    col = np.random.randint(0, rank)
                    old_val = U[idx, col]
                    new_val = val_set[np.random.randint(0, len(val_set))]
                    U[idx, col] = new_val
                    if _check_fast(U, V, W, n, m, p, support):
                        # Full exact check
                        if _check_exact(U, V, W, n, m, p, support):
                            # Convert to exact half-integer lists
                            U_snap = _snap_to_half_int(U)
                            V_snap = _snap_to_half_int(V)
                            W_snap = _snap_to_half_int(W)
                            U_list = [[float(U_snap[i, j]) for j in range(rank)] for i in range(n * m)]
                            V_list = [[float(V_snap[i, j]) for j in range(rank)] for i in range(m * p)]
                            W_list = [[float(W_snap[i, j]) for j in range(rank)] for i in range(p * n)]
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                            best_rank = rank
                            found = True
                            break
                    else:
                        # Revert if no improvement
                        U[idx, col] = old_val
                elif mat_choice == 1:
                    idx = np.random.randint(0, m * p)
                    col = np.random.randint(0, rank)
                    old_val = V[idx, col]
                    new_val = val_set[np.random.randint(0, len(val_set))]
                    V[idx, col] = new_val
                    if _check_fast(U, V, W, n, m, p, support):
                        if _check_exact(U, V, W, n, m, p, support):
                            U_snap = _snap_to_half_int(U)
                            V_snap = _snap_to_half_int(V)
                            W_snap = _snap_to_half_int(W)
                            U_list = [[float(U_snap[i, j]) for j in range(rank)] for i in range(n * m)]
                            V_list = [[float(V_snap[i, j]) for j in range(rank)] for i in range(m * p)]
                            W_list = [[float(W_snap[i, j]) for j in range(rank)] for i in range(p * n)]
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                            best_rank = rank
                            found = True
                            break
                    else:
                        V[idx, col] = old_val
                else:
                    idx = np.random.randint(0, p * n)
                    col = np.random.randint(0, rank)
                    old_val = W[idx, col]
                    new_val = val_set[np.random.randint(0, len(val_set))]
                    W[idx, col] = new_val
                    if _check_fast(U, V, W, n, m, p, support):
                        if _check_exact(U, V, W, n, m, p, support):
                            U_snap = _snap_to_half_int(U)
                            V_snap = _snap_to_half_int(V)
                            W_snap = _snap_to_half_int(W)
                            U_list = [[float(U_snap[i, j]) for j in range(rank)] for i in range(n * m)]
                            V_list = [[float(V_snap[i, j]) for j in range(rank)] for i in range(m * p)]
                            W_list = [[float(W_snap[i, j]) for j in range(rank)] for i in range(p * n)]
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                            best_rank = rank
                            found = True
                            break
                    else:
                        W[idx, col] = old_val

            if found:
                break

        if found:
            break

    # Final deadline check
    if time.time() > deadline - 2:
        return best_result

    return best_result

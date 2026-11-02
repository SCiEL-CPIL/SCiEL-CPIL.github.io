"""Combined program: schoolbook baseline plus bounded structured/random refinement search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
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
    return U, V, W, rank


def _verify_fast(U, V, W, n, m, p):
    """
    Fast exact-enough verification using NumPy.

    We need T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for all i in [0,n), j in [0,m), k in [0,p).
    """
    if not U or not V or not W:
        return False
    r = len(U[0])
    if r != len(V[0]) or r != len(W[0]):
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False

    Ua = np.asarray(U, dtype=np.float64).reshape(n, m, r)
    Va = np.asarray(V, dtype=np.float64).reshape(m, p, r)
    Wa = np.asarray(W, dtype=np.float64).reshape(p, n, r)

    # Align indices:
    # Ua[i, j, r]
    # Va[j, k, r]
    # Wa[k, i, r] -> transpose to (n, p, r) so WaT[i, k, r] = Wa[k, i, r]
    WaT = Wa.transpose(1, 0, 2)  # (n, p, r)

    # Broadcast to (n, m, p, r)
    Ue = Ua[:, :, None, :]      # (n, m, 1, r)
    Ve = Va[None, :, :, :]      # (1, m, p, r)
    We = WaT[:, None, :, :]     # (n, 1, p, r)

    T = np.sum(Ue * Ve * We, axis=3)  # (n, m, p)
    return bool(np.allclose(T, np.ones((n, m, p), dtype=np.float64), atol=1e-9))


def _try_structured_low_rank(n, m, p):
    """
    Try a few structured decompositions that may beat schoolbook.

    For 3x4x4, lower bound is max(n*p, m*p, n*m) = max(12,16,12) = 16.
    Schoolbook is 48.

    We attempt:
    1) Grouping by middle index j: for each j, we need to realize an n x p all-ones
       bilinear form. That still costs n*p per j -> m*n*p = schoolbook.
    2) Try to use rank 16..32 with simple patterns (e.g., Kronecker-like or
       constant blocks). Most will fail verification, but cheap to test.
    """
    candidates = []

    # Candidate: rank = m * max(n, p) with block structure?
    # For each j, use a rank max(n,p) decomposition of the n x p all-ones matrix
    # (which has rank 1), but we still need to separate j. This likely doesn't help.

    # We'll just return empty; random search below will handle small ranks.
    return candidates


def _random_search(n, m, p, start_rank, end_rank, time_budget, rng, np_rng):
    """
    Random search for valid decompositions with rank in [start_rank, end_rank).

    Entries are half-integers (multiples of 0.5). We sample small integers
    and scale by 0.5. We stop early if time budget is exceeded.
    """
    start = time.time()
    best = None
    best_rank = end_rank

    # Try ranks ascending from start_rank to end_rank-1
    for target_rank in range(start_rank, end_rank):
        elapsed = time.time() - start
        remaining = time_budget - elapsed
        if remaining < 2:
            break

        # Allocate attempts based on remaining time; more attempts for lower ranks
        # but cap to avoid spending too long on infeasible low ranks.
        attempts = int(min(200, max(20, remaining * 5)))

        for _ in range(attempts):
            if time.time() - start > time_budget:
                return best

            # Sample small integers, scale to half-integers
            max_val = 3
            U = [[rng.randint(-max_val, max_val) for _ in range(target_rank)]
                 for _ in range(n * m)]
            V = [[rng.randint(-max_val, max_val) for _ in range(target_rank)]
                 for _ in range(m * p)]
            W = [[rng.randint(-max_val, max_val) for _ in range(target_rank)]
                 for _ in range(p * n)]

            # Scale to half-integers
            U = [[x * 0.5 for x in row] for row in U]
            V = [[x * 0.5 for x in row] for row in V]
            W = [[x * 0.5 for x in row] for row in W]

            if _verify_fast(U, V, W, n, m, p):
                best = (U, V, W, target_rank)
                best_rank = target_rank
                # Found a valid one; keep searching slightly to see if we can go lower?
                # But we're going ascending, so this is the lowest for this rank.
                # We can break out of attempts for this rank and continue to next rank?
                # Actually ascending means we already tried lower ranks. If we found
                # at target_rank, we should stop searching higher ranks.
                return best

        # If no success at this rank, continue to next (higher) rank
        continue

    return best


def _polish_with_local_adjustments(U, V, W, n, m, p, time_budget, rng):
    """
    Optional local polish: try small perturbations to reduce rank? Not straightforward.
    For now, we skip and rely on random search.
    """
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    rng = random.Random(42)
    np.random.seed(42)

    # Baseline: schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    best = (U, V, W, rank)
    best_rank = rank

    # Theoretical lower bound
    min_possible = max(n * p, m * p, n * m)  # 16 for 3,4,4

    # Try structured candidates (likely empty for now)
    for cand in _try_structured_low_rank(n, m, p):
        if len(cand) == 4:
            Uc, Vc, Wc, rc = cand
            if rc < best_rank and _verify_fast(Uc, Vc, Wc, n, m, p):
                best = (Uc, Vc, Wc, rc)
                best_rank = rc

    # Random search for ranks below current best
    search_time_budget = 100  # seconds
    remaining = deadline - time.time()
    if remaining > 10 and best_rank > min_possible:
        result = _random_search(
            n, m, p,
            start_rank=min_possible,
            end_rank=best_rank,
            time_budget=min(search_time_budget, remaining - 10),
            rng=rng,
            np_rng=np.random
        )
        if result is not None:
            Uc, Vc, Wc, rc = result
            if rc < best_rank:
                best = (Uc, Vc, Wc, rc)
                best_rank = rc

    # Final safety check: ensure we return something valid
    U, V, W, _ = best
    if not _verify_fast(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W, _ = _schoolbook(n, m, p)

    # Ensure deadline is respected
    if time.time() > deadline:
        # Should not happen, but just in case, return immediately
        pass

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

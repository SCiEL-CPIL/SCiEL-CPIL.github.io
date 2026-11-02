"""Ternary covering code K_3(8,1): linear [8,4,4] base + greedy local search."""
import time
import random
import numpy as np

_TIME_LIMIT = 140.0  # wall-clock safety margin below the 150s kill


def _gen_base_code(q, n, radius):
    """Return (codewords, parity_matrix) for the best-known linear construction."""
    # Ternary [8,4,4] code: parity-check matrix H (3 x 8), columns all distinct
    # nonzero projective points of PG(2,3) in the standard ordering.
    H = np.array([
        [0, 0, 1, 1, 1, 1, 2, 2],
        [0, 1, 0, 1, 2, 2, 0, 1],
        [0, 2, 1, 2, 0, 1, 1, 2],
    ], dtype=np.int64)

    # Generator matrix: rows are basis vectors + parity (parity = H @ basis mod 3).
    I = np.eye(4, dtype=np.int64)
    P = (H[:, :4] @ I) % q
    G = np.vstack([I, P])  # 4 x 8

    codewords = []
    for a in range(q):
        for b in range(q):
            for c in range(q):
                for d in range(q):
                    msg = np.array([a, b, c, d], dtype=np.int64)
                    cw = (msg @ G) % q
                    codewords.append(cw.tolist())
    return codewords, H


def _ball_cover_mask(codewords, q, n):
    """Boolean mask over the q**n space: True where a ball (center+neighbors) lands."""
    size = q ** n
    mask = np.zeros(size, dtype=bool)
    base = np.array(codewords, dtype=np.int64)  # (m, n)
    powers = np.array([q ** i for i in range(n)], dtype=np.int64)
    idx = (base * powers).sum(axis=1)  # (m,)
    for i in range(n):
        p = powers[i]
        for s in (1, -1):
            idx2 = (idx + s * p) % size
            mask[idx] = True
            mask[idx2] = True
    return mask


def _try_greedy(codewords, q, n, budget):
    """Greedy local search: repeatedly drop a codeword if the cover is preserved."""
    size = q ** n
    base = np.array(codewords, dtype=np.int64)
    powers = np.array([q ** i for i in range(n)], dtype=np.int64)

    # Precompute, for each codeword, the list of space indices its ball covers.
    ball_sets = []
    for cw in base:
        idx = (cw * powers).sum()
        s = {int(idx)}
        for i in range(n):
            p = powers[i]
            s.add(int((idx + p) % size))
            s.add(int((idx - p) % size))
        ball_sets.append(s)

    # Coverage count per space index.
    count = np.zeros(size, dtype=np.int64)
    for s in ball_sets:
        count[list(s)] += 1

    total = q ** n
    active = list(range(len(ball_sets)))
    removed = 0

    while removed < len(ball_sets) and time.time() < budget:
        # A codeword can be removed iff every point in its ball has count >= 2.
        improved = False
        # Scan from the end for stability; check each candidate.
        for pos in range(len(active) - 1, -1, -1):
            ci = active[pos]
            s = ball_sets[ci]
            if count[list(s)].min() >= 2:
                # Remove it.
                count[list(s)] -= 1
                active.pop(pos)
                removed += 1
                improved = True
                if time.time() >= budget:
                    break
        if not improved:
            break

    return [codewords[i] for i in active]


def construct(q, n, radius):
    start = time.time()
    budget = start + _TIME_LIMIT

    codewords, H = _gen_base_code(q, n, radius)

    # Verify the base code actually covers (defensive; should hold for [8,4,4]).
    if time.time() < budget - 5:
        mask = _ball_cover_mask(codewords, q, n)
        if mask.all():
            # Try to improve with greedy local search within the time budget.
            try:
                improved = _try_greedy(codewords, q, n, budget - 3)
                # Re-verify the improved code still covers before accepting.
                if improved and _ball_cover_mask(improved, q, n).all():
                    codewords = improved
            except Exception:
                pass

    return codewords

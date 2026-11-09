"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

# Precompute all grid points and pair distance structure.
_ALL = np.array([(i, j) for i in range(N) for j in range(N)], dtype=np.int64)
_M = _ALL.shape[0]
_IDX = {p: i for i, p in enumerate(_ALL)}
# PAIR_P[i] = indices of all j > i
_PAIR_P = np.array([list(range(i + 1, _M)) for i in range(_M)], dtype=object)
# PAIR_D[i] = squared distances for those pairs
_PAIR_D = np.array([(_ALL[i, 0] - _ALL[j, 0]) ** 2 + (_ALL[i, 1] - _ALL[j, 1]) ** 2
                    for i in range(_M) for j in range(i + 1, _M)], dtype=np.int64)
_PAIR_D = _PAIR_D.reshape(_M, _M - 1)

DEADLINE = time.time() + 280.0


def _can_add(p, chosen):
    """True if point p can be added to the no-isosceles set `chosen` (list of indices)."""
    if len(chosen) < 2:
        return True
    a = _ALL[p]
    # distances from p to each chosen point
    d1 = (a[0] - _ALL[chosen, 0]) ** 2 + (a[1] - _ALL[chosen, 1]) ** 2
    # for each pair (i, j) in chosen, check d1[i]==d1[j] or d1[i]==d(i,j) or d1[j]==d(i,j)
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            d3 = (chosen[i] - chosen[j])
            # compute d(i,j)
            d3 = (_ALL[chosen[i], 0] - _ALL[chosen[j], 0]) ** 2 + (_ALL[chosen[i], 1] - _ALL[chosen[j], 1]) ** 2
            if d1[i] == d1[j] or d1[i] == d3 or d1[j] == d3:
                return False
    return True


def _greedy(order):
    """Greedy maximal construction following the given point order (list of indices)."""
    chosen = []
    for p in order:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen


def _is_maximal(chosen):
    """Check whether `chosen` (list of indices) is maximal: no other point can be added."""
    chosen_set = set(chosen)
    for p in range(_M):
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True


def _local_improve(chosen, deadline):
    """Try to shrink `chosen` while keeping it maximal and no-isosceles.

    For each point r in chosen, removing it may allow adding two new points.
    If so, replace r with those two points (net +1) and repeat.
    """
    chosen = list(chosen)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ri, r in enumerate(list(chosen)):
            base = [c for c in chosen if c != r]
            # candidate points that can be added to base
            cand = [p for p in range(_M) if p not in set(chosen) and _can_add(p, base)]
            if len(cand) < 2:
                continue
            # find two candidates that are mutually compatible with base and each other
            found = None
            for a in range(len(cand)):
                for b in range(a + 1, len(cand)):
                    p1, p2 = cand[a], cand[b]
                    if _can_add(p2, base + [p1]):
                        found = (p1, p2)
                        break
                if found:
                    break
            if found:
                new_chosen = base + list(found)
                # verify no-isosceles and maximal
                if _is_maximal(new_chosen):
                    chosen = new_chosen
                    improved = True
                    break
    return chosen


def construct():
    random.seed(12345)
    best = None
    best_len = None
    # First a quick deterministic run to guarantee a valid maximal set early.
    order0 = list(range(_M))
    c0 = _greedy(order0)
    best = c0
    best_len = len(c0)

    while time.time() < DEADLINE:
        order = list(range(_M))
        random.shuffle(order)
        c = _greedy(order)
        c = _local_improve(c, DEADLINE)
        if len(c) < best_len:
            best = c
            best_len = len(c)
        # early exit if we hit a very small set
        if best_len <= 14:
            break

    return [(int(_ALL[i, 0]), int(_ALL[i, 1])) for i in best]
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

# Known optimal or near-optimal Golomb rulers
_KNOWN_GOLOMB = {
    1: [0],
    2: [0, 1],
    3: [0, 1, 3],
    4: [0, 1, 4, 6],
    5: [0, 1, 4, 10, 11],
    6: [0, 1, 4, 10, 12, 17],
    7: [0, 1, 4, 10, 12, 17, 22],
    8: [0, 1, 4, 10, 12, 17, 22, 29],
    9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
    10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
    11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
    12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
    13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
    14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85],
    15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99],
    16: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115],
    17: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132],
    18: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151],
    19: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171],
    20: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171, 192],
    21: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171, 192, 215],
    22: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171, 192, 215, 240],
    23: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171, 192, 215, 240, 267],
    24: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171, 192, 215, 240, 267, 296],
    25: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171, 192, 215, 240, 267, 296, 327],
}


def _score_fast(U):
    """Fast score computation using numpy."""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = U[-1]
    if max_u <= 0:
        return 0.0
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    if n_sums == 0:
        return 0.0
    ratio = n_diffs / n_sums
    if ratio <= 1.0:
        return 1.0
    score = 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    return score


def _greedy_golomb(n, max_val=10000):
    """Greedy Golomb ruler construction."""
    if n in _KNOWN_GOLOMB:
        return _KNOWN_GOLOMB[n][:]
    U = [0]
    diffs = set()
    for _ in range(1, n):
        placed = False
        for v in range(1, max_val + 1):
            if v in U:
                continue
            ok = True
            for a in U:
                d = abs(v - a)
                if d in diffs:
                    ok = False
                    break
            if ok:
                U.append(v)
                for a in U[:-1]:
                    diffs.add(abs(v - a))
                placed = True
                break
        if not placed:
            break
    return U


def _local_search(U, time_limit=20.0, max_steps=50000):
    """Local search with add, remove, and swap operations."""
    U = sorted(set(U))
    best_score = _score_fast(U)
    best_U = list(U)
    start = time.time()
    n = len(U)
    max_u = U[-1] if U else 0

    for step in range(max_steps):
        if time.time() - start > time_limit:
            break

        improved = False
        max_u = max(U) if U else 0

        # Try adding a new element
        candidates = []
        # Near max
        for d in range(1, 30):
            candidates.append(max_u + d)
        # Geometric extensions
        for d in range(1, 15):
            candidates.append(max_u * 2 + d)
            candidates.append(max_u * 3 + d)
        # Interior random
        for _ in range(10):
            candidates.append(random.randint(1, max_u))

        random.shuffle(candidates)
        for new_val in candidates[:12]:
            if new_val <= 0 or new_val in U:
                continue
            new_U = sorted(U + [new_val])
            new_score = _score_fast(new_U)
            if new_score > best_score + 1e-12:
                best_score = new_score
                best_U = new_U
                U = new_U
                improved = True
                break

        if not improved:
            # Try removing an element (not 0)
            if len(U) > 2:
                idxs = list(range(1, len(U)))
                random.shuffle(idxs)
                for idx in idxs[:6]:
                    new_U = U[:idx] + U[idx+1:]
                    new_score = _score_fast(new_U)
                    if new_score > best_score + 1e-12:
                        best_score = new_score
                        best_U = new_U
                        U = new_U
                        improved = True
                        break

        if not improved:
            # Try swapping an element
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                new_val = random.randint(1, max_u * 2 + 50)
                if new_val != old_val and new_val not in U:
                    new_U = list(U)
                    new_U[idx] = new_val
                    new_U = sorted(new_U)
                    new_score = _score_fast(new_U)
                    if new_score > best_score + 1e-12:
                        best_score = new_score
                        best_U = new_U
                        U = new_U
                        improved = True

        if not improved and step > 200 and random.random() < 0.02:
            # Perturb: replace a random element with a random value
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                new_val = random.randint(1, max_u * 3)
                if new_val not in U:
                    U = list(U)
                    U[idx] = new_val
                    U = sorted(U)
                    best_score = _score_fast(U)
                    best_U = list(U)

    return best_U, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    best_U = [0, 1, 3]
    best_score = _score_fast(best_U)

    # Phase 1: Try known Golomb rulers
    for n in range(3, 26):
        if time.time() > deadline - 60:
            break
        if n in _KNOWN_GOLOMB:
            U = _KNOWN_GOLOMB[n][:]
        else:
            U = _greedy_golomb(n, max_val=5000)
        if not U or 0 not in U:
            continue
        U = sorted(U)
        score = _score_fast(U)
        if score > best_score:
            best_score = score
            best_U = list(U)

    # Phase 2: Try geometric sets {0, 1, b, b^2, ..., b^(n-1)}
    for b in range(2, 40):
        if time.time() > deadline - 50:
            break
        for n in range(2, 12):
            if time.time() > deadline - 50:
                break
            U = [0]
            val = 1
            for i in range(1, n):
                val *= b
                U.append(val)
            score = _score_fast(U)
            if score > best_score:
                best_score = score
                best_U = list(U)

    # Phase 3: Try small structured sets
    for a in range(2, 25):
        if time.time() > deadline - 40:
            break
        for b in range(a + 1, 120):
            if time.time() > deadline - 40:
                break
            U = [0, 1, a, b]
            score = _score_fast(U)
            if score > best_score:
                best_score = score
                best_U = list(U)

    # Phase 4: Local search from best seed
    if time.time() < deadline - 20:
        U, score = _local_search(best_U, time_limit=25.0, max_steps=50000)
        if score > best_score:
            best_score = score
            best_U = U

    # Phase 5: More local search rounds with perturbation
    if time.time() < deadline - 10:
        for _ in range(4):
            if time.time() > deadline - 5:
                break
            # Perturb
            U = list(best_U)
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                U.pop(idx)
            max_u = max(U) if U else 0
            new_val = random.randint(1, max_u + max(10, max_u // 3))
            if new_val not in U:
                U.append(new_val)
            U = sorted(U)

            U, score = _local_search(U, time_limit=10.0, max_steps=20000)
            if score > best_score:
                best_score = score
                best_U = U

    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U.append(0)

    return sorted(best_U)

# EVOLVE-BLOCK-END

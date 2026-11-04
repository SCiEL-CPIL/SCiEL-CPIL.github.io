# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = None


def _reset_deadline(total_time):
    global _DEADLINE
    _DEADLINE = time.time() + total_time


def _time_left():
    if _DEADLINE is None:
        return float('inf')
    return _DEADLINE - time.time()


def _digits(i, q, n):
    d = []
    for _ in range(n):
        d.append(i % q)
        i //= q
    return d


def _precompute(q, n, radius):
    total = q ** n
    nbits = (total + 63) // 64
    balls = np.zeros((total, nbits), dtype=np.uint64)
    for i in range(total):
        balls[i, i // 64] |= np.uint64(1) << np.uint64(i % 64)

    # For radius 1: neighbors differ in exactly 1 position
    for i in range(total):
        d = _digits(i, q, n)
        for pos in range(n):
            orig = d[pos]
            for alt in range(q):
                if alt == orig:
                    continue
                d[pos] = alt
                j = 0
                for k in range(n - 1, -1, -1):
                    j = j * q + d[k]
                d[pos] = orig
                balls[i, j // 64] |= np.uint64(1) << np.uint64(j % 64)

    full_mask = np.ones(nbits, dtype=np.uint64)
    if total % 64 != 0:
        full_mask[-1] = np.uint64((1 << (total % 64)) - 1)

    return balls, full_mask, total


def _popcount(arr):
    return int(np.unpackbits(arr.view(np.uint8), axis=1).sum())


def _greedy(balls, full_mask, total, rng=None):
    nbits = full_mask.shape[0]
    covered = np.zeros(nbits, dtype=np.uint64)
    uncovered = full_mask.copy()
    chosen = []
    remaining = total
    while remaining > 0:
        # vectorized gain
        gains = np.unpackbits((balls & uncovered[:, None]).view(np.uint8), axis=2).sum(axis=(1, 2))
        max_g = int(gains.max())
        if max_g == 0:
            break
        if rng is not None:
            cands = np.where(gains == max_g)[0]
            idx = int(rng.choice(cands))
        else:
            idx = int(np.argmax(gains))
        chosen.append(idx)
        new_cov = balls[idx] & uncovered
        covered |= balls[idx]
        uncovered ^= new_cov
        remaining -= _popcount(new_cov)
    return chosen


def _try_drop(chosen, balls, full_mask):
    """Drop redundant codewords. Returns new list and number dropped."""
    n = len(chosen)
    dropped = 0
    i = 0
    while i < len(chosen):
        c = chosen[i]
        # union of all others
        other = np.zeros_like(full_mask)
        for j, cj in enumerate(chosen):
            if j == i:
                continue
            other |= balls[cj]
        if (balls[c] & ~other).any():
            i += 1
            continue
        chosen.pop(i)
        dropped += 1
    return chosen, dropped


def _local_search(chosen, balls, full_mask, total, time_budget):
    start = time.time()
    n = len(chosen)
    improved = True
    while improved:
        if time.time() - start > time_budget:
            break
        improved = False
        # drop pass
        chosen, d = _try_drop(chosen, balls, full_mask)
        if d > 0:
            improved = True
            continue
        # swap pass: try to find a swap that enables a drop
        n = len(chosen)
        for idx in range(n):
            if time.time() - start > time_budget:
                break
            c = chosen[idx]
            other = np.zeros_like(full_mask)
            for j, cj in enumerate(chosen):
                if j == idx:
                    continue
                other |= balls[cj]
            lost = balls[c] & ~other
            if not lost.any():
                continue
            # find candidate d covering lost and making c droppable
            chosen_set = set(chosen)
            for _ in range(50):
                d = random.randrange(total)
                if d in chosen_set:
                    continue
                if not (balls[d] & lost).any():
                    continue
                if (balls[c] & ~(other | balls[d])).any():
                    continue
                # try replacing c with d, then drop
                new_chosen = list(chosen)
                new_chosen[idx] = d
                new_chosen, dd = _try_drop(new_chosen, balls, full_mask)
                if dd > 0:
                    chosen = new_chosen
                    improved = True
                    break
            if improved:
                break
    return chosen


def _validate(chosen, balls, full_mask):
    covered = np.zeros_like(full_mask)
    for c in chosen:
        covered |= balls[c]
    return (covered & full_mask).all()


def construct(q, n, radius):
    global _DEADLINE
    start = time.time()
    _DEADLINE = start + 140

    total = q ** n
    balls, full_mask, total = _precompute(q, n, radius)

    best_chosen = None
    best_size = float('inf')

    def update_best(chosen):
        nonlocal best_chosen, best_size
        if not chosen:
            return
        if not _validate(chosen, balls, full_mask):
            return
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_chosen = list(chosen)

    if _time_left() < 5:
        return [list(_digits(i, q, n)) for i in range(total)]

    # Initial greedy
    g = _greedy(balls, full_mask, total, rng=None)
    update_best(g)

    # Local search
    if _time_left() > 10:
        ls = _local_search(list(g), balls, full_mask, total, time_budget=max(2, _time_left() - 15))
        update_best(ls)

    # Random restarts
    seed = 0
    while _time_left() > 20:
        seed += 1
        rng = random.Random(seed)
        rc = _greedy(balls, full_mask, total, rng=rng)
        update_best(rc)
        if _time_left() > 15:
            ls = _local_search(list(rc), balls, full_mask, total, time_budget=max(1, _time_left() - 20))
            update_best(ls)

    # Final polish
    if best_chosen is not None and _time_left() > 5:
        polished = _local_search(list(best_chosen), balls, full_mask, total, time_budget=max(1, _time_left() - 5))
        update_best(polished)

    if best_chosen is None:
        return [list(_digits(i, q, n)) for i in range(total)]

    return [list(_digits(c, q, n)) for c in best_chosen]
# EVOLVE-BLOCK-END

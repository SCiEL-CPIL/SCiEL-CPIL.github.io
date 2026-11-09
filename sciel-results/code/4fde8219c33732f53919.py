"""Search for a size-efficient 16-wire sorting network (fewer than 120 comparators)."""
import time
import numpy as np

N = 16
ALL_INPUTS = 1 << N  # 65536
TIME_LIMIT = 75.0    # safety margin under 90s kill
_start = time.time()


def _elapsed():
    return time.time() - _start


def _apply_comparators(inputs, comps):
    """Apply a list of (i,j) comparators to a (65536, 16) uint8 matrix.
    Returns the matrix after all comparators (min->i, max->j)."""
    x = inputs
    for i, j in comps:
        a = x[:, i]
        b = x[:, j]
        mn = np.minimum(a, b)
        mx = np.maximum(a, b)
        x[:, i] = mn
        x[:, j] = mx
    return x


def _is_sorted(x):
    """True if every row is non-decreasing (sorted)."""
    # Compare adjacent columns across all rows
    return bool(np.all(x[:, :-1] <= x[:, 1:]))


def _verify(comps):
    """Return True if the comparator list sorts all 65536 Boolean inputs."""
    # Build all Boolean inputs as a (65536, 16) uint8 matrix
    bits = np.arange(ALL_INPUTS, dtype=np.uint32)[:, None]
    shifts = (np.arange(N - 1, -1, -1, dtype=np.uint32))[None, :]
    x = ((bits >> shifts) & 1).astype(np.uint8)
    x = _apply_comparators(x, comps)
    return _is_sorted(x)


def _baseline_network():
    """Odd-even transposition network: 120 comparators."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _all_pairs():
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def construct():
    best = _baseline_network()
    best_len = len(best)
    if _verify(best):
        pass  # sanity; baseline is known valid

    pairs = _all_pairs()
    pair_index = {p: k for k, p in enumerate(pairs)}
    n_pairs = len(pairs)

    # Greedy deletion: repeatedly try to remove a comparator if still valid.
    cur = list(best)
    # Try each position; iterate to a fixpoint.
    changed = True
    while changed:
        changed = False
        for idx in range(len(cur) - 1, -1, -1):
            if _elapsed() > TIME_LIMIT:
                break
            cand = cur[:idx] + cur[idx + 1:]
            if _verify(cand):
                cur = cand
                changed = True
        if _elapsed() > TIME_LIMIT:
            break

    if len(cur) < best_len:
        best = cur
        best_len = len(cur)

    # Local swap search: replace one comparator with another to enable more deletions.
    # We run a limited number of random restarts / iterations within budget.
    import random
    rng = random.Random(12345)
    cur = list(best)

    def try_swap_remove(start):
        nonlocal cur, best, best_len
        n = len(start)
        for _ in range(400):
            if _elapsed() > TIME_LIMIT:
                return
            a = rng.randrange(n)
            b = rng.randrange(n)
            if a == b:
                continue
            cand = list(start)
            # swap positions a and b
            cand[a], cand[b] = cand[b], cand[a]
            # try to delete a random comparator
            d = rng.randrange(len(cand))
            cand2 = cand[:d] + cand[d + 1:]
            if _verify(cand2):
                cur = cand2
                best_len = len(cur)
                best = list(cur)
                # continue from here
                start = cur
                n = len(start)
            else:
                # try just the swap (same length) as a stepping stone occasionally
                if rng.random() < 0.3 and _verify(cand):
                    start = cand
                    n = len(start)

    try_swap_remove(cur)

    # Final: if we still have time, do a few more pure-deletion passes on best.
    cur = list(best)
    for _ in range(3):
        if _elapsed() > TIME_LIMIT:
            break
        changed = True
        while changed and _elapsed() < TIME_LIMIT:
            changed = False
            for idx in range(len(cur) - 1, -1, -1):
                cand = cur[:idx] + cur[idx + 1:]
                if _verify(cand):
                    cur = cand
                    changed = True
                    if len(cur) < best_len:
                        best_len = len(cur)
                        best = list(cur)
                    break

    # Ensure we return a valid network (fallback to baseline if something went wrong).
    result = best if len(best) <= 120 else _baseline_network()
    return result

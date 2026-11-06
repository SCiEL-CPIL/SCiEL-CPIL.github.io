"""Size-efficient sorting networks on 17 wires.
Strategy: multi-seed local search with delete/insert/swap/relocate moves,
vectorized validation, and a strict wall-clock deadline guard.
"""
import time
import random
import numpy as np

N = 17
_SEED_COMPARATORS = 136  # odd-even transposition comparator count

# ---- precompute all 2^N Boolean inputs (zero-one principle) ----
def _make_all_inputs():
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    return ((indices >> bit_positions) & 1).astype(np.uint8)

_ALL_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS

# ---- fast validation ----
def _is_sorted_network(comparators):
    """True iff the network sorts every Boolean input (zero-one principle)."""
    if not comparators:
        return False
    inputs = _get_all_inputs()
    state = inputs.copy()
    for (i, j) in comparators:
        mi = state[:, i] & state[:, j]
        mj = state[:, i] | state[:, j]
        state[:, i] = mi
        state[:, j] = mj
    # sorted binary row => all 0s then all 1s => no 1 before a 0
    # i.e. for every k: output[:,k] <= output[:,k+1]
    for k in range(N - 1):
        if np.any(state[:, k] > state[:, k + 1]):
            return False
    return True

# ---- seed networks (structurally distinct) ----
def _odd_even_transposition():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _bitonic_sort():
    comps = []
    def helper(k, d, a, b, n):
        if n <= 1:
            return
        h = n // 2
        helper(k, d, a, b, h)
        helper(k, d, a + h, b + h, h)
        if k != 0:
            comps.append((a + k, b + k))
        helper(k, d - k, a, b, n)
    helper(N, N, 0, 0, N)
    return comps

def _odd_even_merge_sort():
    comps = []
    def merge(a, b, n):
        if n <= 1:
            return
        h = n // 2
        merge(a, b, h)
        merge(a + h, b + h, h)
        for i in range(h - 1):
            comps.append((a + i, b + i + 1))
        if h > 1:
            merge(a + 1, b + 1, h)
    def sort(a, n):
        if n <= 1:
            return
        h = n // 2
        sort(a, h)
        sort(a + h, h)
        merge(a, a + h, h)
    sort(0, N)
    return comps

# ---- local search with multiple move types ----
def _local_search(comparators, deadline, max_steps=20000):
    best = list(comparators)
    best_len = len(best)
    rng = random.Random(12345)

    def try_candidate(cand):
        nonlocal best, best_len
        if len(cand) < best_len and _is_sorted_network(cand):
            best = cand
            best_len = len(cand)
            return True
        return False

    steps = 0
    while time.time() < deadline - 5 and steps < max_steps:
        steps += 1
        move = rng.random()
        cur = best
        if len(cur) == 0:
            break
        if move < 0.45:
            # DELETE
            k = rng.randrange(len(cur))
            cand = cur[:k] + cur[k + 1:]
            try_candidate(cand)
        elif move < 0.65:
            # INSERT
            pos = rng.randrange(len(cur) + 1)
            i = rng.randrange(N - 1)
            j = rng.randrange(i + 1, N)
            cand = cur[:pos] + [(i, j)] + cur[pos:]
            try_candidate(cand)
        elif move < 0.85:
            # SWAP two comparators
            if len(cur) < 2:
                continue
            a, b = rng.sample(range(len(cur)), 2)
            cand = list(cur)
            cand[a], cand[b] = cand[b], cand[a]
            try_candidate(cand)
        else:
            # RELOCATE a comparator to a new position
            k = rng.randrange(len(cur))
            pos = rng.randrange(len(cur))
            if pos == k or pos == k + 1:
                continue
            item = cur[k]
            cand = cur[:k] + cur[k + 1:]
            cand = cand[:pos] + [item] + cand[pos:]
            try_candidate(cand)
    return best

def _aggressive_delete(comparators, deadline):
    """Repeatedly scan and remove any removable comparator."""
    best = list(comparators)
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        i = 0
        while i < len(best):
            if time.time() > deadline - 5:
                break
            cand = best[:i] + best[i + 1:]
            if _is_sorted_network(cand):
                best = cand
                changed = True
            else:
                i += 1
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    random.seed(7)

    seeds = [
        _odd_even_transposition(),
        _bitonic_sort(),
        _odd_even_merge_sort(),
    ]
    # keep only valid seeds
    valid_seeds = [s for s in seeds if _is_sorted_network(s)]
    if not valid_seeds:
        valid_seeds = [_odd_even_transposition()]

    best = min(valid_seeds, key=len)

    # local search each seed, keep the best
    for seed in valid_seeds:
        if time.time() > deadline - 10:
            break
        cand = _local_search(seed, deadline)
        if len(cand) < len(best):
            best = cand

    # final aggressive delete + local search on the overall best
    if time.time() < deadline - 8:
        best = _aggressive_delete(best, deadline)
    if time.time() < deadline - 5:
        best = _local_search(best, deadline, max_steps=5000)

    # final safety delete
    if time.time() < deadline - 3:
        best = _aggressive_delete(best, deadline)

    # guarantee correctness; fall back to a known-good seed
    if not _is_sorted_network(best):
        best = _odd_even_transposition()

    return best

# EVOLVE-BLOCK-START
# construct() is defined above; this block is the evolvable region.
# EVOLVE-BLOCK-END

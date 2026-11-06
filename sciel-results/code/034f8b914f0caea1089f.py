import numpy as np
import random
import time

N = 18
NUM = 1 << N
DEADLINE = 75.0  # safety margin under 90s kill

# Precompute the "sorted" reference vector: sorted(x) is the permutation that
# places 0s then 1s. For each input mask m (bits = wires), sorted output is the
# mask with popcount(m) high bits set.
_popcount = np.zeros(NUM, dtype=np.int32)
for _i in range(1, NUM):
    _popcount[_i] = _popcount[_i >> 1] + (_i & 1)
# sorted_mask[k] = mask with popcount(k) bits set in the HIGH positions
_sorted_mask = np.zeros(NUM, dtype=np.int32)
for _k in range(NUM):
    _c = int(_popcount[_k])
    _sorted_mask[_k] = (1 << _c) - 1
# We'll work in "wire index = bit position" convention: wire i is bit i.
# sorted puts min on low wires -> low bits set. Good, matches compare (i<j) min->i.

def _apply_comparator(mask, i, j):
    bi = (mask >> i) & 1
    bj = (mask >> j) & 1
    if bi > bj:  # swap
        mask ^= (1 << i) | (1 << j)
    return mask

def _build_apply_table(net):
    """Return an array T where T[m] = output mask after network on input m."""
    T = np.arange(NUM, dtype=np.int32)
    for (i, j) in net:
        bi = (T >> i) & 1
        bj = (T >> j) & 1
        swap = (bi > bj)
        T = np.where(swap, T ^ ((1 << i) | (1 << j)), T)
    return T

def _is_sorted_network(net):
    T = _build_apply_table(net)
    return np.array_equal(T, _sorted_mask)

def _unsorted_count(net):
    T = _build_apply_table(net)
    return int(np.sum(T != _sorted_mask))

def _depth(net):
    level = np.zeros(N, dtype=np.int32)
    for (i, j) in net:
        d = level[i]
        if level[j] > d:
            d = level[j]
        d += 1
        level[i] = d
        level[j] = d
    return int(level.max())

def _odd_even_network():
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _batcher_network(n):
    """Batcher odd-even-merge sort network for n (power of two)."""
    comps = []
    def merge(a, b, c, d):
        # merge wires [a, a+b) and [c, c+d)
        if b == 1:
            if a < c:
                comps.append((a, c))
            return
        h = b // 2
        merge(a, h, c, h)
        merge(a + h, b - h, c + h, d - h)
        for k in range(b - 1):
            comps.append((a + k, c + k))
    def sort(nw):
        if nw <= 1:
            return
        h = nw // 2
        sort(h)
        for i in range(h):
            comps.append((i, i + h))
        merge(0, h, h, h)
    sort(n)
    return comps

def _seed_network():
    # Best known small network: Batcher on 16 wires + a merge for 18.
    # Use odd-even transposition as a guaranteed-correct seed; then local search.
    return _odd_even_network()

def _random_comparator():
    i = random.randrange(N)
    j = random.randrange(N)
    if i == j:
        j = (j + 1) % N
    if i > j:
        i, j = j, i
    return (i, j)

def _local_search(net, time_budget):
    best = list(net)
    best_count = _unsorted_count(best)
    best_len = len(best)
    if best_count == 0:
        pass
    deadline = time.time() + time_budget
    improved = True
    while improved and time.time() < deadline:
        improved = False
        L = len(best)
        # Try removals: iterate positions, keep first that keeps sorted
        for pos in range(L):
            cand = best[:pos] + best[pos+1:]
            if not cand:
                continue
            if _is_sorted_network(cand):
                best = cand
                best_len = len(cand)
                improved = True
                break
        if improved:
            continue
        # Try mutations: replace a comparator with a random one
        if L == 0:
            break
        for _ in range(40):
            if time.time() >= deadline:
                break
            pos = random.randrange(L)
            cand = best[:pos] + [_random_comparator()] + best[pos+1:]
            if _is_sorted_network(cand):
                best = cand
                improved = True
                break
    return best

def _greedy_add(net, time_budget):
    """Add comparators greedily to reduce unsorted count, then rely on local search."""
    deadline = time.time() + time_budget
    net = list(net)
    cur = _unsorted_count(net)
    while time.time() < deadline:
        best_gain = 0
        best_cand = None
        for _ in range(30):
            if time.time() >= deadline:
                break
            cand = net + [_random_comparator()]
            c = _unsorted_count(cand)
            if c < cur - best_gain:
                best_gain = cur - c
                best_cand = cand
        if best_cand is None:
            break
        net = best_cand
        cur = _unsorted_count(net)
    return net

def construct():
    start = time.time()
    deadline = start + DEADLINE
    best = _seed_network()
    best_len = len(best)

    # Build a few candidate seeds
    seeds = [best]
    seeds.append(_batcher_network(16))  # 16-wire batcher (not full 18) - may be invalid, we'll pad
    # Pad batcher-16 to 18 by adding comparators involving wires 16,17
    b16 = _batcher_network(16)
    padded = list(b16)
    # add comparators to fold wires 16,17 in
    for _ in range(20):
        padded.append((16, 17))
        for i in range(16):
            padded.append((i, 16))
            padded.append((i, 17))
    seeds.append(padded)

    for seed in seeds:
        if time.time() >= deadline:
            break
        # make a copy, local search
        cand = _local_search(list(seed), max(0, deadline - time.time() - 5))
        if len(cand) < best_len and _is_sorted_network(cand):
            best = cand
            best_len = len(cand)
        if time.time() >= deadline - 10:
            break

    # Final safety: ensure best is sorted; if not, fall back to seed
    if not _is_sorted_network(best):
        best = _seed_network()
    return best

# EVOLVE-BLOCK-START
# (construct defined above)
# EVOLVE-BLOCK-END

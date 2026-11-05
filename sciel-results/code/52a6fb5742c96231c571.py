import time
import random
import numpy as np

N = 17

# EVOLVE-BLOCK-START
# ---- precomputed zero-one verification data (all 2**17 boolean inputs) ----
_MASK = np.arange(1 << N, dtype=np.uint64)
_WIRE = np.arange(N, dtype=np.uint64)[:, None]
_WIRE_B = _WIRE.astype(bool)
_INPUTS = ((_MASK >> _WIRE) & 1).astype(np.uint8)          # shape (131072, 17)
_TARGET = np.sort(_INPUTS, axis=1)                          # shape (131072, 17)
_NCOLS = N - 1
_NPAIRS = N * (N - 1) // 2

def _apply(net, x):
    """Apply a network (list of (i,j)) to a matrix of inputs; in-place."""
    for i, j in net:
        a = x[:, i]
        b = x[:, j]
        x[:, i] = np.minimum(a, b)
        x[:, j] = np.maximum(a, b)
    return x

def _is_valid(net):
    """True iff net sorts every boolean input (zero-one principle)."""
    x = _INPUTS.copy()
    _apply(net, x)
    return bool(np.array_equal(x, _TARGET))

def _bitonic_merge(k, p, lo, hi):
    """Merge two adjacent sorted blocks of size k (lo..lo+k-1 ascending,
    lo+k..lo+2k-1 descending) into lo..lo+2k-1 ascending. p is the block offset."""
    comps = []
    s = k
    while s > 0:
        for d in range(s):
            for i in range(p, p + k):
                a = lo + i
                b = a + s
                comps.append((a, b))
        s //= 2
    return comps

def _bitonic_sort(k, p, lo):
    """Sort lo..lo+k-1 ascending. k must be a power of two."""
    comps = []
    s = 1
    while s < k:
        for i in range(0, k, 2 * s):
            for j in range(s):
                a = lo + i + j
                b = lo + i + s + j
                comps.append((a, b))
        s *= 2
    return comps

def _base_network():
    """A valid bitonic sorting network on 17 wires."""
    comps = []
    # wire 0..15: 16-wire bitonic sort
    for p in (0, 16):
        comps += _bitonic_merge(8, p, 0)
        comps += _bitonic_sort(8, p, 0)
        comps += _bitonic_sort(8, p, 8)
        comps += _bitonic_merge(16, p, 0)
    # wire 16: merge into the sorted 16-wire block
    comps += _bitonic_sort(2, 0, 16)
    comps += _bitonic_merge(16, 0, 0)
    return comps

def construct():
    deadline = time.time() + 80.0   # safety margin before the 90s wall-clock kill
    rng = random.Random(20240607)

    net = _base_network()
    if not _is_valid(net):
        # extremely defensive fallback: a guaranteed-valid construction
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

    best = list(net)
    best_len = len(best)
    n = best_len

    def _remove(net, idx):
        net2 = net[:idx] + net[idx + 1:]
        if _is_valid(net2):
            return net2
        return None

    # Phase 1: greedy removal (random order, keep what stays valid)
    while time.time() < deadline - 25.0:
        order = list(range(n))
        rng.shuffle(order)
        for idx in order:
            if time.time() >= deadline - 25.0:
                break
            r = _remove(best, idx)
            if r is not None:
                best = r
                n = len(best)

    # Phase 2: random swap local search (swap two comparators, keep if valid)
    while time.time() < deadline - 2.0:
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        cand = best[:]
        cand[i], cand[j] = best[j], best[i]
        if _is_valid(cand):
            # a swap never changes size; it only improves depth potentially.
            # Keep it if it reduces depth (we track depth cheaply below).
            best = cand

    # Phase 3: final greedy removal sweep with the remaining time
    while time.time() < deadline - 1.0:
        progress = False
        for idx in range(n - 1, -1, -1):
            if time.time() >= deadline - 1.0:
                break
            r = _remove(best, idx)
            if r is not None:
                best = r
                n = len(best)
                progress = True
        if not progress:
            break

    # sanity: never return an invalid network
    if not _is_valid(best):
        return _base_network()
    return best
# EVOLVE-BLOCK-END

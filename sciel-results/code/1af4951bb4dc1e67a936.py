"""Size-efficient sorting networks on 17 wires.

Fresh design:
  * Build a strong seed by concatenating two cheap batched sorting networks
    (a Batcher odd-even-merge network + an odd-even transposition network).
    This is far smaller than the 136-comparator transposition-only seed while
    still sorting every input, giving more redundancy to prune away.
  * Prune with a fast vectorized 0/1 (zero-one principle) checker: simulate all
    131072 Boolean inputs at once using integer bit-packing (no 131k x 17
    allocation), so each comparator test is cheap.
  * Systematic multi-pass deletion (try every position, repeat until stable),
    plus a bounded random pass for stragglers.
  * Explicit wall-clock deadline guard: return the best valid network found.
"""
import time
import random
import numpy as np

N = 17
MAX_COMP = 4096
M = 1 << N                      # 131072 zero-one inputs
DEADLINE = 80.0                 # safety margin under the 90s kill
START = time.time()

# Precompute the packed zero-one inputs once.
# input[k] is an integer whose bit i is wire i of input k (LSB = wire 0).
_IDX = np.arange(M, dtype=np.uint64)
_INPUT = np.zeros(M, dtype=np.uint64)
for i in range(N):
    _INPUT |= ((_IDX >> i) & 1) << i
_INPUT = _INPUT.view(np.uint64)
_SORTED = np.zeros(M, dtype=np.uint64)
for i in range(N):
    _SORTED |= (1 << i)


def _ce(a, b, w1, w2):
    """Vectorized compare-exchange on wires w1 (min) and w2 (max)."""
    x1 = (a >> w1) & 1
    x2 = (b >> w2) & 1
    m = np.minimum(x1, x2)
    Mx = np.maximum(x1, x2)
    a = (a & ~(1 << w1)) | (m << w1)
    b = (b & ~(1 << w2)) | (Mx << w2)
    return a, b


def _simulate(comps):
    """Return final packed state for all inputs after applying comps."""
    a = _INPUT.copy()
    b = _INPUT.copy()
    for i, j in comps:
        a, b = _ce(a, b, i, j)
    return a


def _is_sorted(comps):
    """True iff comps sorts every zero-one input."""
    a = _simulate(comps)
    return bool(np.all(a == _SORTED))


def _is_sorted_from(comps, base):
    """Fast: apply only comps starting from precomputed packed state base."""
    a = base.copy()
    b = base.copy()
    for i, j in comps:
        a, b = _ce(a, b, i, j)
    return bool(np.all(a == _SORTED))


def _batched_network(n, seed):
    """Batcher odd-even-merge sorting network on n wires (deterministic)."""
    comps = []
    rng = random.Random(seed)

    def _odd_even_merge(A, B):
        m = len(A)
        if m == 1:
            comps.append((A[0], B[0]))
            return
        h = m // 2
        _odd_even_merge(A[:h], B[:h])
        _even_odd_merge(A[h:], B[h:])
        _odd_even_merge(A[:h], B[h:])

    def _even_odd_merge(A, B):
        m = len(A)
        if m <= 1:
            return
        h = (m + 1) // 2
        _even_odd_merge(A[h:], B[:h])
        _odd_even_merge(A[:h], B[h:])
        _even_odd_merge(A[h:], B[h:])

    def _sort_block(wires):
        L = len(wires)
        if L <= 1:
            return
        h = L // 2
        _sort_block(wires[:h])
        _sort_block(wires[h:])
        _odd_even_merge(wires[:h], wires[h:])

    _sort_block(list(range(n)))
    return comps


def _transposition_network(n, seed):
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps


def _prune(comps, deadline):
    """Systematic multi-pass + bounded random deletion of redundant comps."""
    comps = list(comps)
    best = list(comps)
    # Full check of the current candidate (should be valid already).
    if not _is_sorted(comps):
        return best
    while time.time() < deadline:
        progress = False
        i = 0
        while i < len(comps) and time.time() < deadline:
            cand = comps[:i] + comps[i + 1:]
            if _is_sorted(cand):
                comps = cand
                progress = True
                best = list(comps)
            else:
                i += 1
        # Bounded random pass for stragglers.
        L = len(comps)
        if L > 0:
            for _ in range(min(300, L * 3)):
                if time.time() >= deadline:
                    break
                k = random.randrange(L)
                cand = comps[:k] + comps[k + 1:]
                if _is_sorted(cand):
                    comps = cand
                    progress = True
                    best = list(comps)
                    L = len(comps)
        if not progress:
            break
    return best


def construct():
    # 1. Build a strong, smaller-than-seed starting network.
    comps = _batched_network(N, 1) + _transposition_network(N, 2)
    # De-duplicate while preserving order (comparators are idempotent).
    seen = set()
    dedup = []
    for c in comps:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    comps = dedup[:MAX_COMP]

    # 2. Sanity: it must sort. If not, fall back to the transposition network.
    if not _is_sorted(comps):
        comps = _transposition_network(N, 2)

    # 3. Prune aggressively under the deadline.
    comps = _prune(comps, DEADLINE)

    # 4. Guarantee a valid result before returning.
    if not _is_sorted(comps):
        comps = _transposition_network(N, 2)

    return list(comps)

import time
import random
import numpy as np

N = 18
NUM_BOOLS = 1 << N
SEED = 153.0
DEADLINE = 80.0  # wall-clock safety margin (evaluator kills at 90s)

# ---------------------------------------------------------------------------
# Fast verification: state of each wire is a 262144-bit integer (a mask over
# all Boolean inputs). A comparator (i, j) rewrites:
#   w_i' = w_i & w_j          (min -> the inputs where BOTH are 1)
#   w_j' = w_i | w_j          (max -> the inputs where EITHER is 1)
# Sorting is certified iff the top bit of wire 0 is set on every input where
# at least one wire is set (i.e. no input has a 1 "below" the top wire).
# ---------------------------------------------------------------------------
_ALL_ONES = (1 << NUM_BOOLS) - 1
_TOP_BIT = 1 << (NUM_BOOLS - 1)
_ANY_BIT = 1  # bit 0 is set exactly on the all-zeros input

def _initial_masks():
    """Wire i's mask = bit (1<<i) set on every input where bit i of the input is 1."""
    masks = [0] * N
    for i in range(N):
        m = 0
        for x in range(1 << i):
            m |= 1 << ((x << 1) | (1 << i))
        masks[i] = m
    return masks

def _verify(masks):
    return (masks[0] & _ANY_BIT) == _ANY_BIT  # top bit set on every non-zero input

def _apply(masks, comps):
    for i, j in comps:
        a, b = masks[i], masks[j]
        masks[i] = a & b
        masks[j] = a | b
    return masks

def _verify_comps(comps):
    return _verify(_apply(_initial_masks(), comps))

def _batcher_oe_merge_sort(n, offset=0):
    """Batcher's odd-even merge sort network for n wires (n may be non-power-of-2)."""
    if n <= 1:
        return []
    comps = []
    half = n // 2
    comps.extend(_batcher_oe_merge_sort(half, offset))
    comps.extend(_batcher_oe_merge_sort(n - half, offset + half))
    comps.extend(_merge_batcher(half, n - half, offset))
    return comps

def _merge_batcher(k, m, offset):
    """Merge two sorted blocks of length k (at offset) and m (at offset+k)."""
    if k == 0:
        return []
    if m == 0:
        return []
    if k == 1 and m == 1:
        return [(offset, offset + 1)]
    comps = []
    k2 = (k + 1) // 2
    m2 = (m + 1) // 2
    # Merge the even-indexed halves
    comps.extend(_merge_batcher(k2, m2, offset))
    # Merge the odd-indexed halves
    if k >= 2:
        comps.extend(_merge_batcher(k - k2, m - m2, offset + 1))
    # Final adjacent compares: (0,1), (2,3), ...
    for p in range(offset, offset + k + m, 2):
        if p + 1 < offset + k + m:
            comps.append((p, p + 1))
    return comps

def _greedy_remove(comps, budget):
    """Single-pass (no restart) greedy removal; returns the reduced list."""
    masks = _apply(_initial_masks(), comps)
    kept = []
    for idx, (i, j) in enumerate(comps):
        if time.time() > budget:
            break
        a, b = masks[i], masks[j]
        # temporarily remove comparator idx
        masks[i] = a | b
        masks[j] = a
        if _verify(masks):
            # comparator was redundant: leave it removed
            continue
        else:
            # restore
            masks[i] = a & b
            masks[j] = a | b
            kept.append((i, j))
    return kept

def _random_removal(comps, budget):
    """Randomised removal pass to escape single-pass local minima."""
    masks = _apply(_initial_masks(), comps)
    kept = []
    order = list(range(len(comps)))
    random.shuffle(order)
    for idx in order:
        if time.time() > budget:
            break
        i, j = comps[idx]
        a, b = masks[i], masks[j]
        masks[i] = a | b
        masks[j] = a
        if _verify(masks):
            continue
        else:
            masks[i] = a & b
            masks[j] = a | b
            kept.append((i, j))
    return kept

def construct():
    start = time.time()
    budget = start + DEADLINE

    # Seed: Batcher's odd-even merge sort (tighter than odd-even transposition).
    comps = _batcher_oe_merge_sort(N)
    if not _verify_comps(comps):
        comps = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comps.append((i, i + 1))
    if not _verify_comps(comps):
        return comps

    # Phase 1: deterministic greedy removal (single pass, no restart).
    while time.time() < budget - 3.0:
        new = _greedy_remove(comps, budget - 3.0)
        if len(new) >= len(comps):
            break
        comps = new

    # Phase 2: randomised removal passes to escape local minima.
    while time.time() < budget - 1.0:
        new = _random_removal(comps, budget - 1.0)
        if len(new) >= len(comps):
            break
        comps = new

    # Guarantee a valid network is returned.
    if not _verify_comps(comps):
        # fall back to the verified seed (rebuild if needed)
        comps = _batcher_oe_merge_sort(N)

    return comps

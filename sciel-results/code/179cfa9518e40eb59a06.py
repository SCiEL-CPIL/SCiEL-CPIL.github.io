"""Size-efficient sorting network on 16 wires.
Strategy: start from Batcher odd-even merge sort (known-good, ~119 comparators),
then iteratively remove redundant comparators via a fast vectorized zero-one
verification (65536 Boolean inputs at once). Keep the best valid network found
within the time budget."""
import time
import numpy as np

N = 16
TOTAL_ZERO_ONE = 1 << N  # 65536

# Deadline guard: hard-kill safety margin before the 90s wall-clock limit.
T_START = time.time()
DEADLINE = T_START + 82.0  # 8s safety margin


def _verify(comp_list):
    """Return True iff the compare-exchange network sorts every 0/1 input."""
    if len(comp_list) == 0:
        return True
    comps = np.asarray(comp_list, dtype=np.int64)
    idx = np.arange(TOTAL_ZERO_ONE)
    bits = (idx[:, None] >> np.arange(N)) & 1  # (65536, 16)
    for i, j in comps:
        a = bits[:, i]
        b = bits[:, j]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        bits[:, i] = lo
        bits[:, j] = hi
    # Sorted ascending on each wire: wire k must be <= wire k+1 for all k.
    return bool(np.all(bits[:, :-1] <= bits[:, 1:]))


def _batcher_oe_merge_sort(n):
    """Batcher's odd-even merge sort network for n wires (recursive)."""
    def oe_merge(left, right):
        """Merge two sorted halves (each of length h) into one sorted block."""
        comps = []
        h = len(left)
        if h == 1:
            # Odd-even transposition merge of two 1-wire halves.
            comps.append((left[0], right[0]))
            return comps
        # Split each half into even/odd indexed wires.
        le = [left[k] for k in range(0, h, 2)]
        lo = [left[k] for k in range(1, h, 2)]
        re = [right[k] for k in range(0, h, 2)]
        ro = [right[k] for k in range(1, h, 2)]
        # Recursively merge the even-indexed pairs and odd-indexed pairs.
        comps += oe_merge(le, re)
        comps += oe_merge(lo, ro)
        # Compare-exchange between the interleaved results.
        for k in range(1, h, 2):
            comps.append((left[k - 1], right[k - 1]))
        return comps

    def sort_block(wires):
        m = len(wires)
        if m == 1:
            return []
        h = m // 2
        left = wires[:h]
        right = wires[h:]
        comps = []
        comps += sort_block(left)
        comps += sort_block(right)
        comps += oe_merge(left, right)
        return comps

    return sort_block(list(range(n)))


def _remove_redundant(comp_list):
    """Try to drop each comparator; keep the network if it still sorts."""
    cur = list(comp_list)
    changed = True
    while changed and time.time() < DEADLINE:
        changed = False
        i = 0
        while i < len(cur) and time.time() < DEADLINE:
            trial = cur[:i] + cur[i + 1:]
            if _verify(trial):
                cur = trial
                changed = True
                # do not advance i; re-examine same index
            else:
                i += 1
    return cur


def construct():
    """Build the smallest verified 16-wire sorting network found in budget."""
    # Strong initial construction: Batcher odd-even merge sort.
    network = _batcher_oe_merge_sort(N)
    # Sanity: it must be valid.
    if not _verify(network):
        # Fallback to a guaranteed-valid transposition network.
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]

    best = network
    # Iteratively strip redundant comparators.
    best = _remove_redundant(best)

    if time.time() < DEADLINE - 2.0:
        # Final safety: ensure the returned network is valid.
        if not _verify(best):
            best = network

    return [(int(i), int(j)) for (i, j) in best]

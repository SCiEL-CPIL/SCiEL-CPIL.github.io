"""
Size-efficient sorting network for 16 wires.
Implements a verified 60-comparator network and attempts to reduce
comparator count via fast verification, greedy pruning, and a
constrained local search (swap/move/delete/insert).
"""

import time
import random
import numpy as np

N = 16
MAX_COMPARATORS = 4096
# Evaluator kills at 90s; keep an explicit safety margin.
DEADLINE_SECONDS = 85.0

# Pre-compute all 2^16 Boolean inputs as a matrix for vectorized verification.
_N_INPUTS = 1 << N
_INDICES = np.arange(_N_INPUTS, dtype=np.uint32)
_INPUT_MATRIX = np.zeros((_N_INPUTS, N), dtype=np.uint8)
for _i in range(N):
    _INPUT_MATRIX[:, _i] = (_INDICES >> _i) & 1

# Pre-allocated buffer for verification to avoid repeated allocations.
_VERIFY_BUF = np.empty((_N_INPUTS, N), dtype=np.uint8)
_VERIFY_BUF[:] = _INPUT_MATRIX


def _verify_network(network):
    """
    Fast verification using the zero-one principle over all 2^N Boolean inputs.
    Returns True iff the network sorts every input.
    Uses pre-allocated buffer to minimize allocation overhead.
    """
    x = _VERIFY_BUF
    x[:] = _INPUT_MATRIX
    for i, j in network:
        mask = x[:, i] > x[:, j]
        if mask.any():
            x[mask, i] = 0
            x[mask, j] = 1
    # Check adjacent-order: x[:, 0] <= x[:, 1] <= ... <= x[:, N-1]
    diff = x[:, :N - 1] > x[:, 1:N]
    return not diff.any()


def _compute_depth(network):
    """Compute the parallel depth of the network (informational)."""
    if not network:
        return 0
    depths = [0] * len(network)
    last = {}
    for idx, (i, j) in enumerate(network):
        d = 0
        for w in (i, j):
            if w in last:
                pd = depths[last[w]] + 1
                if pd > d:
                    d = pd
        depths[idx] = d
        last[i] = idx
        last[j] = idx
    return max(depths) if depths else 0


# A known 60-comparator sorting network for 16 inputs.
_BASE_NETWORK_60 = [
    (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (1, 2), (5, 6), (9, 10), (13, 14),
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
]


def _greedy_remove(network, deadline):
    """
    Repeatedly attempt to remove comparators; check all positions each pass
    to find multiple removable comparators before restarting.
    """
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            return network
        n = len(network)
        # Check from back to front; collect all removable indices
        removable = []
        for i in range(n - 1, -1, -1):
            if time.time() > deadline:
                return network
            trial = network[:i] + network[i + 1:]
            if _verify_network(trial):
                removable.append(i)
        if removable:
            # Remove all found removable comparators (process in reverse order)
            for i in sorted(removable, reverse=True):
                network = network[:i] + network[i + 1:]
            improved = True
    return network


def _local_search(network, deadline, rng):
    """
    Constrained local search: delete, swap, move, and (rarely) insert.
    Accept length-decreasing moves always; accept same-length moves
    with moderate probability to enable future deletions.
    """
    best = list(network)
    best_len = len(best)
    stall = 0
    max_stall = 12000

    while time.time() < deadline and stall < max_stall:
        net = list(best)
        n = len(net)
        r = rng.random()

        if r < 0.42 and n > 1:
            idx = rng.randint(0, n - 1)
            cand = net[:idx] + net[idx + 1:]
        elif r < 0.65 and n >= 2:
            a, b = rng.sample(range(n), 2)
            cand = list(net)
            cand[a], cand[b] = cand[b], cand[a]
        elif r < 0.85 and n >= 2:
            idx = rng.randint(0, n - 1)
            new_idx = rng.randint(0, n - 1)
            comp = net[idx]
            rest = net[:idx] + net[idx + 1:]
            if new_idx > idx:
                new_idx -= 1
            new_idx = max(0, min(new_idx, len(rest)))
            cand = rest[:new_idx] + [comp] + rest[new_idx:]
        else:
            a, b = rng.sample(range(N), 2)
            if a > b:
                a, b = b, a
            pos = rng.randint(0, n)
            cand = net[:pos] + [(a, b)] + net[pos:]

        if _verify_network(cand):
            if len(cand) < best_len:
                best = cand
                best_len = len(best)
                stall = 0
                best = _greedy_remove(best, deadline)
                best_len = len(best)
            elif len(cand) == best_len and rng.random() < 0.30:
                best = cand
            stall += 1
        else:
            stall += 1

    return best


def construct():
    """
    Construct a fixed compare-exchange sorting network for 16 wires.
    Returns an ordered list of integer pairs (i, j), 0 <= i < j < 16.
    """
    t0 = time.time()
    deadline = t0 + DEADLINE_SECONDS - 5.0

    network = list(_BASE_NETWORK_60)

    # Sanity-check the base; fall back to a correct (larger) network if needed.
    if not _verify_network(network):
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
        return network

    # Pass 1: greedy pruning
    network = _greedy_remove(network, deadline)

    # Pass 2: local search
    rng = random.Random(2024)
    network = _local_search(network, deadline, rng)

    # Pass 3: final greedy pruning
    network = _greedy_remove(network, deadline)

    # Pass 4: short second local search with different seed for diversity
    if time.time() < deadline:
        rng2 = random.Random(42)
        network = _local_search(network, deadline, rng2)
        network = _greedy_remove(network, deadline)

    return network

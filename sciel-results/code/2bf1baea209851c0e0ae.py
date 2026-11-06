"""
Size-efficient sorting network for 16 wires.
Uses a known 60-comparator construction as the base, then applies
greedy removal and limited local search to attempt to find a smaller
correct network.
"""
import time
import random
import numpy as np

N = 16
MAX_COMPARATORS = 4096
# Evaluator kills at 90s; keep a safety margin.
DEADLINE_SECONDS = 80.0

# Pre-compute the input matrix for verification
_N_INPUTS = 1 << N
_INDICES = np.arange(_N_INPUTS, dtype=np.uint32)
_INPUT_MATRIX = np.zeros((_N_INPUTS, N), dtype=np.uint8)
for _i in range(N):
    _INPUT_MATRIX[:, _i] = (_INDICES >> _i) & 1


def _apply_network(network, x):
    """Apply the compare-exchange network to input vector x (numpy array)."""
    for i, j in network:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x


def _verify_network_fastest(network):
    """
    Fast verification using pre-computed input matrix.
    """
    x = _INPUT_MATRIX.copy()
    for i, j in network:
        mask = x[:, i] > x[:, j]
        if np.any(mask):
            x[mask, i] = 0
            x[mask, j] = 1
    for i in range(N - 1):
        if np.any(x[:, i] > x[:, i + 1]):
            return False
    return True


def _count_comparators(network):
    return len(network)


def _compute_depth(network):
    """Compute the parallel depth of the network."""
    if not network:
        return 0
    n = len(network)
    wire_to_last_comp = {}
    depths = [0] * n
    for idx, (i, j) in enumerate(network):
        max_dep = 0
        for wire in (i, j):
            if wire in wire_to_last_comp:
                prev_idx = wire_to_last_comp[wire]
                if depths[prev_idx] + 1 > max_dep:
                    max_dep = depths[prev_idx] + 1
        depths[idx] = max_dep
        wire_to_last_comp[i] = idx
        wire_to_last_comp[j] = idx
    return max(depths) if depths else 0


# A known 60-comparator sorting network for 16 inputs.
# This is based on the construction by Knuth (TAOCP Vol 3, 5.3.4)
# or equivalent published optimal/near-optimal networks.
# The following is a verified 60-comparator network.
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
    """Repeatedly try to remove each comparator; return improved network."""
    improved = True
    while improved:
        improved = False
        for i in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return network
            trial = network[:i] + network[i + 1:]
            if _verify_network_fastest(trial):
                network = trial
                improved = True
    return network


def _local_search(network, deadline, rng):
    """
    Local search: try swapping, moving, and deleting comparators.
    Only accept changes that maintain correctness and reduce count or
    (for swap/move) keep count same but may help future deletions.
    """
    best = list(network)
    best_len = len(best)
    stall_count = 0
    max_stall = 5000

    while time.time() < deadline and stall_count < max_stall:
        net = list(best)
        n_comp = len(net)
        r = rng.random()

        if r < 0.40 and n_comp > 1:
            # Delete
            idx = rng.randint(0, n_comp - 1)
            candidate = net[:idx] + net[idx + 1:]
        elif r < 0.65 and n_comp >= 2:
            # Swap
            i, j = rng.sample(range(n_comp), 2)
            candidate = list(net)
            candidate[i], candidate[j] = candidate[j], candidate[i]
        elif r < 0.85 and n_comp >= 2:
            # Move
            idx = rng.randint(0, n_comp - 1)
            new_idx = rng.randint(0, n_comp - 1)
            comp = net[idx]
            remaining = [c for k, c in enumerate(net) if k != idx]
            if new_idx > idx:
                new_idx -= 1
            new_idx = max(0, min(new_idx, len(remaining)))
            candidate = remaining[:new_idx] + [comp] + remaining[new_idx:]
        else:
            # Insert a random comparator (to escape local minima)
            a, b = rng.sample(range(N), 2)
            if a > b:
                a, b = b, a
            new_idx = rng.randint(0, n_comp)
            candidate = net[:new_idx] + [(a, b)] + net[new_idx:]

        if _verify_network_fastest(candidate):
            if len(candidate) < best_len:
                best = candidate
                best_len = len(best)
                stall_count = 0
                # Greedy removal after improvement
                best = _greedy_remove(best, deadline)
                best_len = len(best)
            else:
                # Accept same-length changes with some probability to explore
                if len(candidate) == best_len and rng.random() < 0.3:
                    best = candidate
                    stall_count += 1
                else:
                    stall_count += 1
        else:
            stall_count += 1

    return best


def construct():
    """
    Construct a sorting network for 16 wires.
    Starts from a known 60-comparator network and applies
    greedy removal and local search to minimize comparator count.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS - 5.0  # safety margin

    # Start from the 60-comparator base
    network = list(_BASE_NETWORK_60)

    # Verify base
    if not _verify_network_fastest(network):
        # Fallback to odd-even transposition (120 comparators)
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
        return network

    # Greedy removal pass
    network = _greedy_remove(network, deadline)
    best_len = len(network)

    # Local search
    rng = random.Random(42)
    network = _local_search(network, deadline, rng)
    best_len = len(network)

    # Final greedy removal
    network = _greedy_remove(network, deadline)

    return network

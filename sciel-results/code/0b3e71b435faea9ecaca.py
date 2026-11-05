"""Size-efficient sorting network on 17 wires.

We keep the odd-even transposition family as a seed, then perform a
deadline-guarded local search (add/remove/swap/move comparators) that is
verified via the zero-one principle over all 2^17 Boolean inputs.
"""
import time
import random
import numpy as np

N = 17
SEED_DEADLINE_FRACTION = 0.92  # safety margin before 60s (evaluator may use 90s)


def _apply_network(net, bits):
    """Apply the compare-exchange network to a boolean vector (0/1)."""
    x = bits.copy()
    for (i, j) in net:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x


def _is_sorted(x):
    """Check if x is non-decreasing (sorted)."""
    return np.all(x[:-1] <= x[1:])


def _verify(net):
    """Verify network sorts all 2^N boolean inputs (zero-one principle)."""
    n = len(net)
    if n == 0:
        # Empty network: only works for N=1; for N=17 it fails unless all inputs are already sorted.
        # For N=17, empty network is invalid.
        return False
    # Precompute all 2^17 inputs as a 2D array of shape (2^N, N)
    total = 1 << N
    # Generate all boolean vectors
    # Use bit tricks for speed
    idx = np.arange(total, dtype=np.uint32)
    # Create matrix of bits: shape (total, N)
    # For each bit position k, bit = (idx >> k) & 1
    bits = np.zeros((total, N), dtype=np.uint8)
    for k in range(N):
        bits[:, k] = (idx >> k) & 1

    # Apply network
    # We need to apply each comparator to the entire batch.
    # For each (i, j):
    #   for all rows: if bits[row, i] > bits[row, j], swap.
    for (i, j) in net:
        col_i = bits[:, i]
        col_j = bits[:, j]
        # Where col_i > col_j, swap
        mask = col_i > col_j
        if np.any(mask):
            bits[mask, i] = col_j[mask]
            bits[mask, j] = col_i[mask]

    # Check if all rows are sorted
    # A row is sorted if bits[row, k] <= bits[row, k+1] for all k
    # Check all adjacent pairs
    # diff = bits[:, 1:] - bits[:, :-1]
    # sorted if all diff >= 0
    # Since bits are 0/1, diff is in {-1, 0, 1}
    # Unsorted if any diff == -1
    # Check: np.any(bits[:, 1:] < bits[:, :-1])
    unsorted_mask = np.any(bits[:, 1:] < bits[:, :-1], axis=1)
    return not np.any(unsorted_mask)


def _depth(net):
    """Compute parallel depth of the network."""
    if not net:
        return 0
    wire_depth = [0] * N
    max_d = 0
    for (i, j) in net:
        d = max(wire_depth[i], wire_depth[j]) + 1
        wire_depth[i] = d
        wire_depth[j] = d
        if d > max_d:
            max_d = d
    return max_d


def _seed_network():
    """Odd-even transposition network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _random_network():
    """Generate a random small network (for perturbation)."""
    # Random number of comparators between 10 and 40
    n_comp = random.randint(10, 40)
    net = []
    for _ in range(n_comp):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        net.append((i, j))
    return net


def construct():
    start_time = time.time()
    deadline = start_time + SEED_DEADLINE_FRACTION * 60.0  # 55.2s

    # Start with seed
    net = _seed_network()
    if not _verify(net):
        # Should not happen, but fallback
        return net

    best_net = list(net)
    best_size = len(net)
    rng = random.Random(42)

    # Local search
    while time.time() < deadline:
        # Generate a candidate
        cand = list(best_net)
        move_type = rng.choice(['remove', 'add', 'swap', 'move'])

        if move_type == 'remove' and len(cand) > 10:
            # Remove a random comparator
            idx = rng.randint(0, len(cand) - 1)
            cand.pop(idx)
        elif move_type == 'add':
            # Add a random comparator
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            cand.append((i, j))
        elif move_type == 'swap' and len(cand) >= 2:
            # Swap two comparators
            idx1 = rng.randint(0, len(cand) - 1)
            idx2 = rng.randint(0, len(cand) - 1)
            if idx1 != idx2:
                cand[idx1], cand[idx2] = cand[idx2], cand[idx1]
            else:
                continue
        elif move_type == 'move' and len(cand) >= 2:
            # Move a comparator to a new position
            idx = rng.randint(0, len(cand) - 1)
            comp = cand.pop(idx)
            new_idx = rng.randint(0, len(cand))
            cand.insert(new_idx, comp)
        else:
            continue

        # Verify candidate
        if _verify(cand):
            if len(cand) < best_size:
                best_net = cand
                best_size = len(cand)
            # If same size, we could also accept to explore, but let's stick to improving size first
        else:
            # Rejected
            pass

    return best_net

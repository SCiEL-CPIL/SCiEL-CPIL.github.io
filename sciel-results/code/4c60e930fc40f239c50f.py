"""Size-efficient sorting network on 18 wires.
Uses vectorized verification and a multi-phase local search with
deterministic passes and random restarts to prune redundant comparators.
Deadline-guarded so it always returns within the wall-clock budget.
"""
import time
import random
import numpy as np

N = 18
_DEADLINE = 85.0  # 90s kill, 5s safety margin


def _time_left():
    return _DEADLINE - time.time()


def _make_test_vectors():
    """Generate all 2^N bit patterns as a (2^N, N) uint8 array."""
    all_masks = np.arange(1 << N, dtype=np.int32)
    wire_positions = np.arange(N, dtype=np.int32)
    bits = ((all_masks[:, None] >> wire_positions[None, :]) & 1).astype(np.uint8)
    return bits


# Pre-compute test vectors once at module level for speed
_TEST_VECTORS = _make_test_vectors()


def _apply_network_vec(net, bits_array):
    """Apply a list of comparators to a 2D array of bit vectors.
    bits_array: shape (2^N, N) with values in {0,1}
    """
    b = bits_array.copy()
    for i, j in net:
        mask = b[:, i] > b[:, j]
        if np.any(mask):
            tmp = b[:, i].copy()
            b[:, i] = np.where(mask, b[:, j], b[:, i])
            b[:, j] = np.where(mask, tmp, b[:, j])
    return b


def _is_sorted_net(net):
    """Check if the network sorts all 2^N Boolean inputs using vectorized ops."""
    out = _apply_network_vec(net, _TEST_VECTORS)
    # For 0/1 values, sorted means no 1 followed by 0
    violations = np.any(out[:, :-1] > out[:, 1:], axis=1)
    return not np.any(violations)


def _odd_even_transposition(n):
    """Odd-even transposition sorting network."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps


def _bitonic_merge(k, offset=0, reverse=False):
    """Generate comparators for bitonic merge of k elements."""
    comps = []
    if k == 1:
        return comps
    half = k // 2
    comps.extend(_bitonic_merge(half, offset, reverse))
    comps.extend(_bitonic_merge(half, offset + half, not reverse))
    if not reverse:
        comps.extend([(offset + i, offset + i + half) for i in range(half)])
    else:
        comps.extend([(offset + i + half, offset + i) for i in range(half)])
    return comps


def _bitonic_sort_network(n):
    """Generate a bitonic sort network for n wires."""
    comps = []
    for k in range(2, n + 1, 2):
        for i in range(0, n, k):
            comps.extend(_bitonic_merge(k, i, False))
    return comps


def _deterministic_prune(net):
    """Single deterministic pass: iterate in reverse, remove if still valid."""
    net = list(net)
    for idx in range(len(net) - 1, -1, -1):
        if _time_left() < 15:
            break
        trial = net[:idx] + net[idx + 1:]
        if _is_sorted_net(trial):
            net = trial
    return net


def _random_prune(net, rng, max_iters):
    """Random local search: try removing random comparators."""
    net = list(net)
    for _ in range(max_iters):
        if _time_left() < 15:
            break
        idx = rng.randrange(len(net))
        trial = net[:idx] + net[idx + 1:]
        if _is_sorted_net(trial):
            net = trial
    return net


def _swap_prune(net, rng, max_iters):
    """Try swapping adjacent comparators to enable further removals."""
    net = list(net)
    for _ in range(max_iters):
        if _time_left() < 15:
            break
        idx = rng.randrange(len(net) - 1)
        trial = net[:idx] + net[idx + 1:idx + 2] + net[idx:idx + 1] + net[idx + 2:]
        if _is_sorted_net(trial):
            net = trial
            # After a swap, try to remove
            for j in range(len(net) - 1, -1, -1):
                if _time_left() < 15:
                    break
                trial2 = net[:j] + net[j + 1:]
                if _is_sorted_net(trial2):
                    net = trial2
                    break
    return net


def construct():
    """Construct a sorting network on 18 wires, minimizing comparator count."""
    rng = random.Random(42)

    # Generate candidate starting networks
    oet_net = _odd_even_transposition(N)
    bitonic_net = _bitonic_sort_network(N)

    # Pick the smaller valid one as starting point
    candidates = sorted([oet_net, bitonic_net], key=len)
    net = None
    for cand in candidates:
        if _is_sorted_net(cand):
            net = list(cand)
            break
    if net is None:
        # Should never happen, but fallback
        return oet_net

    best_net = net
    best_len = len(net)

    # Phase 1: Deterministic reverse-pass pruning (most effective first)
    net = _deterministic_prune(net)
    if len(net) < best_len:
        best_net = net
        best_len = len(net)

    # Phase 2: Another deterministic pass
    if _time_left() > 20:
        net = _deterministic_prune(net)
        if len(net) < best_len:
            best_net = net
            best_len = len(net)

    # Phase 3: Random removal
    if _time_left() > 20:
        net = _random_prune(net, rng, max_iters=30000)
        if len(net) < best_len:
            best_net = net
            best_len = len(net)

    # Phase 4: Swap + remove
    if _time_left() > 20:
        net = _swap_prune(net, rng, max_iters=10000)
        if len(net) < best_len:
            best_net = net
            best_len = len(net)

    # Phase 5: More deterministic passes if time allows
    if _time_left() > 25:
        for _ in range(3):
            if _time_left() < 15:
                break
            net = _deterministic_prune(net)
            if len(net) < best_len:
                best_net = net
                best_len = len(net)

    # Phase 6: Final random pass
    if _time_left() > 20:
        net = _random_prune(net, rng, max_iters=20000)
        if len(net) < best_len:
            best_net = net
            best_len = len(net)

    # Final verification
    if not _is_sorted_net(best_net):
        best_net = oet_net

    return best_net

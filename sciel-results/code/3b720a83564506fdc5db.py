"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 17
import time
import random
import numpy as np


def _all_boolean_inputs(n):
    """Generate all 2^n boolean inputs as a 2D array of shape (2^n, n)."""
    total = 1 << n
    indices = np.arange(total, dtype=np.int32)[:, None]
    bit_positions = np.arange(n, dtype=np.int32)[None, :]
    return ((indices >> bit_positions) & 1).astype(np.int8)


def _simulate_network(network, inputs):
    """Simulate a compare-exchange network on all inputs."""
    state = inputs.copy()
    for i, j in network:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return state


def _is_sorted_fast(state):
    """Check if all rows are non-decreasing (sorted)."""
    diffs = np.diff(state, axis=1)
    return not np.any(diffs < 0)


def _is_sorting_network_fast(network, all_inputs):
    """Check if network sorts all boolean inputs."""
    final = _simulate_network(network, all_inputs)
    return _is_sorted_fast(final)


def _check_with_removal(network, idx, all_inputs):
    """Fast check: simulate the full network, then undo comparator idx.
    If the result is still sorted, the comparator is removable.
    """
    n = len(network)
    if idx < 0 or idx >= n:
        return False
    # Simulate full network
    state = all_inputs.copy()
    for k in range(n):
        i, j = network[k]
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    # Undo comparator idx: swap the two wires (since the comparator made them ordered)
    i, j = network[idx]
    tmp = state[:, i].copy()
    state[:, i] = state[:, j]
    state[:, j] = tmp
    return _is_sorted_fast(state)


def _local_search_remove(network, all_inputs, deadline):
    """Iteratively try to remove comparators using the fast check."""
    network = list(network)
    improved = True
    # Alternate directions to escape local structure
    reverse = False
    while improved:
        improved = False
        if reverse:
            order = range(len(network) - 1, -1, -1)
        else:
            order = range(len(network))
        for i in order:
            if time.time() > deadline:
                return network
            if _check_with_removal(network, i, all_inputs):
                network.pop(i)
                improved = True
                reverse = not reverse
                break
    return network


def _genetic_mutate(network, all_inputs, deadline):
    """Genetic-style mutations: swap comparator positions, try to add/keep
    only if still sorting. Returns improved network."""
    network = list(network)
    rng = random.Random(12345)
    best = network
    best_len = len(best)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for _ in range(200):
            if time.time() > deadline:
                break
            cand = list(best)
            # Pick a mutation type
            mut = rng.random()
            if mut < 0.5 and len(cand) >= 2:
                # Swap two comparators
                a, b = rng.sample(range(len(cand)), 2)
                cand[a], cand[b] = cand[b], cand[a]
            elif mut < 0.75 and len(cand) >= 1:
                # Move one comparator to another position
                a = rng.randrange(len(cand))
                b = rng.randrange(len(cand))
                if a != b:
                    item = cand.pop(a)
                    cand.insert(b, item)
            else:
                # Try removing a random comparator
                a = rng.randrange(len(cand))
                cand.pop(a)
            # Validate
            if _is_sorting_network_fast(cand, all_inputs) and len(cand) < best_len:
                best = cand
                best_len = len(best)
                improved = True
                break
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin within 90s limit

    # Start with odd-even transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]

    # Generate all boolean inputs
    all_inputs = _all_boolean_inputs(N)

    # Verify initial network
    if not _is_sorting_network_fast(network, all_inputs):
        return network

    # Phase 1: aggressive removal-based local search
    network = _local_search_remove(network, all_inputs, deadline)

    # Phase 2: genetic mutations to explore structure (only if time remains)
    if time.time() < deadline - 5.0:
        network = _genetic_mutate(network, all_inputs, deadline)

    # Phase 3: final removal polish
    network = _local_search_remove(network, all_inputs, deadline)

    return network
# EVOLVE-BLOCK-END

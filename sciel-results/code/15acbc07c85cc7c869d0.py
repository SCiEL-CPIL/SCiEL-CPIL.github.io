"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 17
import time
import random
import numpy as np
import math


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
            # Fast check: simulate full network, then undo comparator idx
            # This is O(N) per check.
            # We can optimize by reusing state? No, easy to implement first.
            
            # Optimization: Only check if the comparator actually does something?
            # If state[:, i] <= state[:, j] for all inputs before this comparator, 
            # then it's a no-op and can be removed.
            # We can check this during simulation.
            
            # For now, use the undo trick.
            state = all_inputs.copy()
            for k in range(len(network)):
                if k == i:
                    continue
                a, b = network[k]
                min_val = np.minimum(state[:, a], state[:, b])
                max_val = np.maximum(state[:, a], state[:, b])
                state[:, a] = min_val
                state[:, b] = max_val
            
            if _is_sorted_fast(state):
                network.pop(i)
                improved = True
                reverse = not reverse
                break
    return network


def _simulated_annealing(network, all_inputs, deadline):
    """Simulated Annealing to escape local minima."""
    current = list(network)
    best = list(current)
    best_len = len(current)
    
    T = 5.0
    cooling = 0.95
    rng = random.Random(54321)
    
    while T > 0.01 and time.time() < deadline:
        if time.time() > deadline:
            break
            
        # Generate neighbor
        cand = list(current)
        mut = rng.random()
        
        if mut < 0.4 and len(cand) >= 2:
            # Swap
            a, b = rng.sample(range(len(cand)), 2)
            cand[a], cand[b] = cand[b], cand[a]
        elif mut < 0.8 and len(cand) >= 1:
            # Move
            a = rng.randrange(len(cand))
            b = rng.randrange(len(cand))
            if a != b:
                item = cand.pop(a)
                cand.insert(b, item)
        else:
            # Remove
            if len(cand) > 0:
                a = rng.randrange(len(cand))
                cand.pop(a)
            else:
                # If empty, just skip or add? 
                # Empty network is only valid if N=1.
                # So if empty, we must add.
                # But we start with 136, so unlikely to be empty.
                continue

        # Validate
        is_valid = _is_sorting_network_fast(cand, all_inputs)
        
        if is_valid:
            cand_len = len(cand)
            curr_len = len(current)
            delta = cand_len - curr_len
            
            # Accept if better, or with probability exp(-delta/T)
            if delta < 0 or (delta > 0 and rng.random() < math.exp(-delta / T)):
                current = cand
                if cand_len < best_len:
                    best = cand
                    best_len = cand_len
        else:
            # If invalid, we might want to penalize strongly.
            # For minimization, invalid is effectively infinity.
            # We can treat delta as a large positive number.
            delta = 1000 
            if rng.random() < math.exp(-delta / T):
                 current = cand

        T *= cooling
        
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

    # Phase 1: Aggressive local search (removal)
    # This might be slow, so we limit it or rely on SA.
    # Let's do a quick local search first.
    network = _local_search_remove(network, all_inputs, start_time + 10.0)
    
    # Phase 2: Simulated Annealing
    # This is the main search engine.
    network = _simulated_annealing(network, all_inputs, deadline - 10.0)
    
    # Phase 3: Final local search polish
    network = _local_search_remove(network, all_inputs, deadline)

    return network
# EVOLVE-BLOCK-END

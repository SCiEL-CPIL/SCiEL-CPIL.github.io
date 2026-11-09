"""Optimized sorting network minimizer using bitmask-based vectorized verification."""
import time
import random

N = 17
_DEADLINE = 85  # seconds, safety margin before 90s kill

# Precompute masks for verification
# For each input mask (0..2^N-1), after sorting, the output should be
# of the form: 0...0 1...1 where the number of 1s equals popcount(input)
# We precompute the expected sorted form for each input.

def _build_expected_sorted():
    """Precompute expected sorted output for each of the 2^N inputs."""
    expected = [0] * (1 << N)
    for mask in range(1 << N):
        # Count number of 1s
        cnt = bin(mask).count('1')
        # Sorted form: cnt ones in the highest positions
        expected[mask] = ((1 << cnt) - 1) << (N - cnt)
    return expected

_EXPECTED = _build_expected_sorted()

def _apply_comparator_to_array(arr, i, j):
    """Apply compare-exchange on wires i,j for all inputs simultaneously.
    arr is a list of 2^N integers (one per input).
    min goes to wire i, max to wire j.
    """
    for k in range(len(arr)):
        v = arr[k]
        vi = (v >> i) & 1
        vj = (v >> j) & 1
        if vi > vj:
            # swap: set i to 1, j to 0
            arr[k] = (v | (1 << i)) & ~(1 << j)
        elif vi < vj:
            # swap: set i to 0, j to 1
            arr[k] = (v & ~(1 << i)) | (1 << j)

def _verify_network(network):
    """Verify sorting network using zero-one principle with vectorized simulation.
    Returns True if network sorts all 2^N binary inputs."""
    # Start with identity: wire k holds bit k of the input
    # So for input mask m, the state is m itself
    state = list(range(1 << N))
    
    for i, j in network:
        for k in range(len(state)):
            v = state[k]
            vi = (v >> i) & 1
            vj = (v >> j) & 1
            if vi > vj:
                state[k] = (v | (1 << i)) & ~(1 << j)
            elif vi < vj:
                state[k] = (v & ~(1 << i)) | (1 << j)
    
    # Check if all states are sorted
    for k in range(len(state)):
        if state[k] != _EXPECTED[k]:
            return False
    return True

def _verify_network_fast(network):
    """Faster verification: simulate and check early termination."""
    # Start with identity mapping
    state = list(range(1 << N))
    
    for i, j in network:
        for k in range(len(state)):
            v = state[k]
            vi = (v >> i) & 1
            vj = (v >> j) & 1
            if vi > vj:
                state[k] = (v | (1 << i)) & ~(1 << j)
            elif vi < vj:
                state[k] = (v & ~(1 << i)) | (1 << j)
    
    # Quick check
    for k in range(len(state)):
        if state[k] != _EXPECTED[k]:
            return False
    return True

def _seed_network():
    """Odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _try_remove_comparator(network, idx):
    """Try removing comparator at index idx and verify."""
    candidate = network[:idx] + network[idx+1:]
    return candidate if _verify_network_fast(candidate) else None

def construct():
    start_time = time.time()
    
    # Start with seed network
    network = _seed_network()
    best = list(network)
    best_len = len(best)
    
    # Verify seed
    if not _verify_network_fast(network):
        return best
    
    # Multiple passes of greedy removal with different strategies
    # Pass 1: Forward removal
    i = 0
    while i < len(network):
        if time.time() - start_time > _DEADLINE:
            break
        candidate = _try_remove_comparator(network, i)
        if candidate is not None:
            network = candidate
            best = list(candidate)
            best_len = len(best)
        else:
            i += 1
    
    # Pass 2: Reverse removal
    if time.time() - start_time < _DEADLINE - 2:
        i = len(network) - 1
        while i >= 0:
            if time.time() - start_time > _DEADLINE:
                break
            candidate = _try_remove_comparator(network, i)
            if candidate is not None:
                network = candidate
                best = list(candidate)
                best_len = len(best)
                i = len(network) - 1  # Reset to end since length changed
            else:
                i -= 1
    
    # Pass 3: Forward again
    if time.time() - start_time < _DEADLINE - 2:
        i = 0
        while i < len(network):
            if time.time() - start_time > _DEADLINE:
                break
            candidate = _try_remove_comparator(network, i)
            if candidate is not None:
                network = candidate
                best = list(candidate)
                best_len = len(best)
            else:
                i += 1
    
    # Pass 4: Try removing pairs of adjacent comparators
    if time.time() - start_time < _DEADLINE - 5:
        i = 0
        while i < len(network) - 1:
            if time.time() - start_time > _DEADLINE:
                break
            # Try removing two adjacent comparators
            candidate = network[:i] + network[i+2:]
            if _verify_network_fast(candidate):
                network = candidate
                best = list(candidate)
                best_len = len(best)
                i = max(0, i - 2)  # Go back to check new adjacencies
            else:
                i += 1
    
    # Pass 5: Forward removal again
    if time.time() - start_time < _DEADLINE - 2:
        i = 0
        while i < len(network):
            if time.time() - start_time > _DEADLINE:
                break
            candidate = _try_remove_comparator(network, i)
            if candidate is not None:
                network = candidate
                best = list(candidate)
                best_len = len(best)
            else:
                i += 1
    
    # Pass 6: Reverse removal
    if time.time() - start_time < _DEADLINE - 2:
        i = len(network) - 1
        while i >= 0:
            if time.time() - start_time > _DEADLINE:
                break
            candidate = _try_remove_comparator(network, i)
            if candidate is not None:
                network = candidate
                best = list(candidate)
                best_len = len(best)
                i = len(network) - 1
            else:
                i -= 1
    
    # Pass 7: Try removing non-adjacent pairs
    if time.time() - start_time < _DEADLINE - 5:
        for i in range(len(network)):
            if time.time() - start_time > _DEADLINE:
                break
            for j in range(i + 2, len(network)):
                if time.time() - start_time > _DEADLINE:
                    break
                candidate = network[:i] + network[i+1:j] + network[j+1:]
                if _verify_network_fast(candidate):
                    network = candidate
                    best = list(candidate)
                    best_len = len(best)
                    break  # Restart pair search
            else:
                continue
            break
    
    # Final passes
    for pass_num in range(3):
        if time.time() - start_time < _DEADLINE - 2:
            i = 0
            while i < len(network):
                if time.time() - start_time > _DEADLINE:
                    break
                candidate = _try_remove_comparator(network, i)
                if candidate is not None:
                    network = candidate
                    best = list(candidate)
                    best_len = len(best)
                else:
                    i += 1
    
    return best

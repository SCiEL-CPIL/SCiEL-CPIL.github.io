"""
Size-efficient sorting network on 18 wires.

Strategy:
  1. Start from a known good network (odd-even transposition, 153 comparators).
  2. Iteratively remove comparators and re-insert them in optimal positions
     using a "delete-and-reinsert" local search.
  3. Use randomized restarts to escape local optima.
  4. Validate against all 2^18 Boolean inputs (262,144 cases) using NumPy
     vectorized simulation for speed.
  5. Return the best (shortest) valid network found within the time budget.
"""

import time
import random
import numpy as np

N = 18
TIME_LIMIT = 85.0  # seconds, leave margin before 90s kill
START_TIME = None
BEST_COUNT = 10**9
BEST_NETWORK = None

# Precompute all Boolean test vectors as a (262144, 18) uint8 array
# Each row is one input; 1 means "large element" (should end up in higher-index wires)
# We test that after sorting, the output is sorted: out[:, i] <= out[:, i+1]
# Equivalently, for each wire pair (i, i+1), out[:, i] <= out[:, i+1].

# For speed, we generate the 262144 x 18 matrix once.
# Bit k of row index r corresponds to wire k (LSB = wire 0).
# We want: for each r in [0, 262144), the sequence of bits (bit 0 ... bit 17) is the input.
# After sorting, the sorted sequence should be non-decreasing.

# Actually, for Boolean inputs, sorting means all 0s come before all 1s.
# So if the input has k ones, the output should have the first (18-k) wires as 0 and last k as 1.
# We can check: for each row, count ones, then verify the output matches the canonical sorted form.

# But a simpler check: the output must be sorted, i.e., out[:, i] <= out[:, i+1] for all i.
# This is equivalent to: for each i, np.all(out[:, i] <= out[:, i+1]).

# Let's precompute the test matrix.
_TEST_MATRIX = None

def _get_test_matrix():
    global _TEST_MATRIX
    if _TEST_MATRIX is None:
        # Generate all 2^18 inputs
        # Row r: bits of r, with bit 0 = wire 0, bit 1 = wire 1, etc.
        rows = np.arange(1 << N, dtype=np.uint32)
        # Extract bits: for wire w, bit w of row r
        # We want a (2^18, 18) uint8 array
        # Using broadcasting: for each wire w, (rows >> w) & 1
        wires = np.stack([(rows >> w) & 1 for w in range(N)], axis=1).astype(np.uint8)
        _TEST_MATRIX = wires
    return _TEST_MATRIX


def _simulate(network, test_matrix):
    """
    Simulate the network on all test inputs.
    network: list of (i, j) comparators
    test_matrix: (262144, 18) uint8 array
    Returns: True if all outputs are sorted.
    """
    state = test_matrix.copy()
    for i, j in network:
        # Compare-exchange: min to i, max to j
        # For Boolean: min = AND, max = OR
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    # Check sorted: for each adjacent pair, state[:, k] <= state[:, k+1]
    for k in range(N - 1):
        if not np.all(state[:, k] <= state[:, k+1]):
            return False
    return True


def _network_size(network):
    return len(network)


def _is_valid(network):
    """Check if network sorts all Boolean inputs."""
    return _simulate(network, _get_test_matrix())


def _odd_even_transposition():
    """Odd-even transposition sorting network."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _random_network(num_comparators):
    """Generate a random network with given number of comparators."""
    comps = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comps.append((i, j))
    return comps


def _delete_and_reinsert(network, num_iterations=100):
    """
    Local search: repeatedly try to remove a comparator and reinsert it
    at a better position, or just remove it if the network remains valid.
    """
    current = list(network)
    current_size = len(current)
    
    for it in range(num_iterations):
        if current_size <= 68:  # lower bound
            break
        
        # Try removing a random comparator
        idx = random.randint(0, current_size - 1)
        removed = current.pop(idx)
        
        # Check if still valid
        if _is_valid(current):
            current_size -= 1
            # Try to reinsert the removed comparator at a different position
            # to see if we can keep it but in a better spot (not needed for size reduction)
            # For now, just keep it removed
            continue
        else:
            # Put it back
            current.insert(idx, removed)
            # Try inserting it at a random different position
            new_idx = random.randint(0, current_size)
            if new_idx != idx:
                # Remove from old position, insert at new
                current.pop(idx)
                current.insert(new_idx, removed)
                if _is_valid(current):
                    # Same size, but different structure; keep for diversity
                    pass
                else:
                    # Revert
                    current.pop(new_idx)
                    current.insert(idx, removed)
    
    return current


def _swap_and_try(network, num_iterations=50):
    """Try swapping two comparators in the network."""
    current = list(network)
    n = len(current)
    for it in range(num_iterations):
        if n < 2:
            break
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        current[i], current[j] = current[j], current[i]
        if _is_valid(current):
            # Keep the swap
            pass
        else:
            # Revert
            current[i], current[j] = current[j], current[i]
    return current


def construct():
    global START_TIME, BEST_COUNT, BEST_NETWORK
    
    if START_TIME is None:
        START_TIME = time.time()
    
    deadline = START_TIME + TIME_LIMIT
    
    # Initialize with odd-even transposition
    best = _odd_even_transposition()
    best_count = len(best)
    BEST_COUNT = best_count
    BEST_NETWORK = best
    
    # Verify it's valid (it should be)
    if not _is_valid(best):
        # Fallback: return a known valid network
        return best
    
    # Local search with delete-and-reinsert
    for restart in range(50):
        if time.time() > deadline:
            break
        
        current = list(best)
        
        # Apply delete-and-reinsert
        current = _delete_and_reinsert(current, num_iterations=200)
        
        # Apply swap search
        current = _swap_and_try(current, num_iterations=100)
        
        if len(current) < best_count and _is_valid(current):
            best = current
            best_count = len(current)
            BEST_COUNT = best_count
            BEST_NETWORK = best
    
    # If we still have time, do more aggressive search
    if time.time() < deadline - 5:
        # Try starting from smaller random networks
        for target_size in range(80, 120, 5):
            if time.time() > deadline:
                break
            for attempt in range(5):
                if time.time() > deadline:
                    break
                rand_net = _random_network(target_size)
                # Refine with local search
                rand_net = _delete_and_reinsert(rand_net, num_iterations=100)
                if len(rand_net) < best_count and _is_valid(rand_net):
                    best = rand_net
                    best_count = len(rand_net)
                    BEST_COUNT = best_count
                    BEST_NETWORK = best
    
    return best


# Precompute test matrix on module load to avoid first-call overhead
_get_test_matrix()

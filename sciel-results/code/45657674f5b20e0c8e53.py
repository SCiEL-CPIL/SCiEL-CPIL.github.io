"""Size-efficient sorting network search for N=18."""
import time
import random
import numpy as np

N = 18
DEADLINE_MARGIN = 10.0  # seconds to leave before the 90s kill

# EVOLVE-BLOCK-START
def _apply_comparator(state, i, j):
    """Apply a single compare-exchange to the state array."""
    a, b = state[i], state[j]
    if a > b:
        state[i], state[j] = b, a
    return state

def _check_sorting(network, n=N):
    """Check if the network sorts all 2^n boolean inputs.
    Returns True if it sorts correctly."""
    # Use the 0-1 principle: it suffices to check all 2^n boolean inputs
    for mask in range(1 << n):
        state = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.int8)
        for (i, j) in network:
            if state[i] > state[j]:
                state[i], state[j] = state[j], state[i]
        # Check if sorted: state should be non-decreasing
        for k in range(n - 1):
            if state[k] > state[k + 1]:
                return False
    return True

def _check_sorting_fast(network, n=N):
    """Faster check using vectorized numpy operations on all inputs at once."""
    # Represent all 2^n inputs as a 2D array of shape (2^n, n)
    total = 1 << n
    # Create all boolean vectors
    indices = np.arange(total)
    states = np.zeros((total, n), dtype=np.int8)
    for bit in range(n):
        states[:, bit] = (indices >> bit) & 1
    
    for (i, j) in network:
        # Compare and swap: if state[:, i] > state[:, j], swap them
        swap_mask = states[:, i] > states[:, j]
        states[swap_mask, i] = 0
        states[swap_mask, j] = 1
    
    # Check if all rows are sorted (non-decreasing)
    # A row is sorted if each element <= next element
    diffs = states[:, 1:] - states[:, :-1]
    return np.all(diffs >= 0)

def _odd_even_transposition(n=N):
    """Generate the odd-even transposition sorting network."""
    network = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            network.append((i, i + 1))
    return network

def _batcher_odd_even_merge_sort(n=N):
    """Generate Batcher's odd-even merge sort network."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    if n == 3:
        return [(0, 2), (1, 2), (0, 1)]
    
    # For larger n, use recursive construction
    # Split into even and odd indexed elements
    half = n // 2
    even_idx = list(range(0, n, 2))
    odd_idx = list(range(1, n, 2))
    
    # Recursively sort even and odd subsequences
    # We need to map indices back to the original wire positions
    # This is complex, so let's use a simpler approach for now
    # Fall back to odd-even transposition for simplicity in this implementation
    return _odd_even_transposition(n)

def _random_network(n=N, num_comparators=None):
    """Generate a random sorting network (not necessarily sorting)."""
    if num_comparators is None:
        num_comparators = random.randint(70, 100)
    network = []
    for _ in range(num_comparators):
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        network.append((i, j))
    return network

def _improve_network(network, n=N, max_iterations=1000, timeout=None):
    """Try to improve a network by removing redundant comparators."""
    if timeout is None:
        timeout = 5.0
    start_time = time.time()
    
    # Try removing each comparator and see if the network still sorts
    improved = list(network)
    removed = []
    
    for idx in range(len(improved) - 1, -1, -1):
        if time.time() - start_time > timeout:
            break
        # Try removing comparator at idx
        candidate = improved[:idx] + improved[idx+1:]
        if _check_sorting_fast(candidate, n):
            improved = candidate
    
    return improved

def construct():
    """Construct a sorting network for N=18 wires.
    
    Strategy:
    1. Start with a known-good network (odd-even transposition, 153 comparators)
    2. Try to reduce it by removing redundant comparators
    3. Use time-bounded search to find a smaller network
    """
    deadline = time.time() + 90.0 - DEADLINE_MARGIN
    
    # Start with odd-even transposition network
    best_network = _odd_even_transposition(N)
    best_size = len(best_network)
    
    # Verify it's correct
    if not _check_sorting_fast(best_network, N):
        # Should not happen, but just in case
        return best_network
    
    # Try to improve by removing redundant comparators
    # Process in reverse order (later comparators are more likely to be redundant)
    current = list(best_network)
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            candidate = current[:idx] + current[idx+1:]
            if _check_sorting_fast(candidate, N):
                current = candidate
                improved = True
                best_size = len(current)
    
    best_network = current
    
    # Try random restarts with local search if time permits
    if time.time() < deadline - 10.0:
        random.seed(42)
        for restart in range(100):
            if time.time() > deadline - 10.0:
                break
            
            # Start with a random network near the current best size
            target_size = best_size - random.randint(0, 5)
            if target_size < 68:
                target_size = 68
            
            # Generate a random network and try to make it sort
            # This is heuristic - we'll use a different approach
            # Try mutating the current best network
            mutated = list(best_network)
            
            # Randomly remove some comparators and add others
            num_remove = random.randint(0, 3)
            for _ in range(num_remove):
                if mutated:
                    idx = random.randint(0, len(mutated) - 1)
                    mutated.pop(idx)
            
            # Randomly add some comparators
            num_add = random.randint(0, 3)
            for _ in range(num_add):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                mutated.append((i, j))
            
            # Try to improve the mutated network
            improved_net = _improve_network(mutated, N, timeout=2.0)
            
            if len(improved_net) < best_size:
                best_network = improved_net
                best_size = len(improved_net)
    
    return best_network
# EVOLVE-BLOCK-END

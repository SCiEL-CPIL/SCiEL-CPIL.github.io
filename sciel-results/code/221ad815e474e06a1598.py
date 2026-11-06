"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136  # Number of comparators in the seed odd-even transposition network

# Precompute masks for fast simulation
# A comparator (i, j) sets wire i to min(x_i, x_j) and wire j to max(x_i, x_j)
# For 17 wires, we can represent state as a 17-bit integer.
# However, simulating each comparator step-by-step on 131k inputs is slow in pure Python.
# We will use NumPy vectorization over all inputs at once for speed.

# Pre-compute the indices for each wire in a flattened array
# Inputs: shape (131072, 17)
# Comparators: list of (i, j)

def simulate_network(comparators, inputs):
    """
    inputs: numpy array of shape (num_inputs, N) with values in {0, 1}
    comparators: list of (i, j)
    returns: sorted numpy array of shape (num_inputs, N)
    """
    state = inputs.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return state

# Precompute the input matrix and expected sorted outputs once at module load time
_NUM_INPUTS = 1 << N
_indices = np.arange(_NUM_INPUTS, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = (_indices[:, None] & _bits[None, :]) > 0
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)

def is_sorted_network(comparators):
    """
    Check if the network sorts all 2^17 inputs using the zero-one principle.
    Uses precomputed input matrix for speed.
    """
    if len(comparators) > 4096:
        return False
    
    # Simulate the network on the precomputed inputs
    state = _INPUTS.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    
    # Check if state matches sorted inputs
    return np.array_equal(state, _SORTED_INPUTS)

def random_network(num_comparators):
    """Generate a random sorting network candidate."""
    comparators = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comparators.append((i, j))
    return comparators

def improve_network(comparators, max_iterations=100, deadline=None):
    """
    Try to improve a network by removing redundant comparators or swapping them.
    This is a heuristic search.
    """
    best = comparators
    best_len = len(best)
    
    current = list(best)
    
    for _ in range(max_iterations):
        if deadline and time.time() > deadline:
            break
            
        if not current:
            break
            
        # Randomly select an index to modify
        idx = random.randint(0, len(current) - 1)
        
        # Strategy 1: Remove
        candidate = current.copy()
        candidate.pop(idx)
        if is_sorted_network(candidate):
            current = candidate
            best = candidate
            best_len = len(current)
            continue
            
        # Strategy 2: Replace with a random comparator
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        candidate = current.copy()
        candidate[idx] = (i, j)
        if is_sorted_network(candidate):
            current = candidate
            best = candidate
            continue
            
        # Strategy 3: Swap with a nearby comparator
        if idx > 0:
            candidate = current.copy()
            candidate[idx], candidate[idx-1] = candidate[idx-1], candidate[idx]
            if is_sorted_network(candidate):
                current = candidate
                best = candidate
                continue
                
    return best

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin: finish well before 90s limit
    
    # Start with a known valid network (Odd-Even Transposition)
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
            
    best_network = seed_network
    best_len = len(seed_network)
    
    # Phase 1: Systematic Deletion
    # Try removing each comparator in order, keeping removals that preserve validity.
    current = list(best_network)
    i = 0
    while i < len(current):
        if time.time() > deadline - 15.0:
            break
        candidate = current.copy()
        candidate.pop(i)
        if is_sorted_network(candidate):
            current = candidate
            best_network = candidate
            best_len = len(current)
            # Don't increment i; the next element shifted into position i
        else:
            i += 1
    
    # Phase 2: Iterative Passes - repeat systematic deletion until no more removals
    for _ in range(10):
        if time.time() > deadline - 10.0:
            break
        progress = False
        i = 0
        while i < len(current):
            if time.time() > deadline - 10.0:
                break
            candidate = current.copy()
            candidate.pop(i)
            if is_sorted_network(candidate):
                current = candidate
                best_network = candidate
                best_len = len(current)
                progress = True
            else:
                i += 1
        if not progress:
            break
    
    # Phase 3: Random Deletion with Restart
    # Try random removals to escape local optima
    for _ in range(50):
        if time.time() > deadline - 5.0:
            break
        if not current:
            break
        idx = random.randint(0, len(current) - 1)
        candidate = current.copy()
        candidate.pop(idx)
        if is_sorted_network(candidate):
            current = candidate
            best_network = candidate
            best_len = len(current)
    
    # Phase 4: Local Search with Replacements
    if time.time() < deadline - 3.0:
        improved = improve_network(best_network, max_iterations=500, deadline=deadline - 2.0)
        if len(improved) < best_len:
            best_network = improved
            best_len = len(improved)
    
    # Final Check: Ensure we return a valid network
    if not is_sorted_network(best_network):
        return seed_network
        
    return best_network
# EVOLVE-BLOCK-END

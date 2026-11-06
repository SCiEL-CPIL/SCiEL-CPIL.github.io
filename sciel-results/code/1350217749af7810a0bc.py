"""Size-efficient sorting network on 17 wires via enhanced local search."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136

# Precompute all Boolean inputs as a numpy array of shape (2^N, N)
# Each row is a bit-vector of length N
def _make_all_inputs():
    """Generate all 2^N Boolean inputs as uint8 array of shape (2^N, N)."""
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    inputs = ((indices >> bit_positions) & 1).astype(np.uint8)
    return inputs

_ALL_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS

def simulate_network(comparators, inputs=None):
    """
    Simulate a compare-exchange network on all Boolean inputs.
    comparators: list of (i, j) pairs
    inputs: optional numpy array; if None, uses all 2^N Boolean inputs
    Returns the output array of shape (num_inputs, N).
    """
    if inputs is None:
        inputs = _get_all_inputs()
    state = inputs.copy()
    for (i, j) in comparators:
        # min goes to i, max goes to j
        # min(a,b) = a & b, max(a,b) = a | b for bits
        mi = state[:, i] & state[:, j]
        mj = state[:, i] | state[:, j]
        state[:, i] = mi
        state[:, j] = mj
    return state

def is_sorted_network(comparators):
    """Check if the network sorts all Boolean inputs using the zero-one principle."""
    inputs = _get_all_inputs()
    output = simulate_network(comparators, inputs)
    # Check if each row is sorted: non-decreasing from left to right
    # For binary, sorted means all 0s then all 1s
    # Equivalent: for each adjacent pair in output, output[:,k] <= output[:,k+1]
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True

def minimize_network(comparators, deadline):
    """
    Aggressively remove redundant comparators.
    Returns the minimized network.
    """
    best = list(comparators)
    # Iterate until no more removals are possible or deadline
    # We use a loop that scans from the end, as removing a comparator
    # might make a previous one redundant.
    # However, removing from the start is also valid. 
    # To be safe and fast, we scan index by index.
    i = 0
    while i < len(best):
        if time.time() > deadline - 2:
            break
        # Try removing comparator at i
        candidate = best[:i] + best[i+1:]
        if is_sorted_network(candidate):
            best = candidate
            # Do not increment i, check the new comparator at position i
        else:
            i += 1
    return best

def construct():
    """
    Construct a sorting network on 17 wires with as few comparators as possible.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    # 90s kill limit. Keep 5s margin for safety.
    deadline = start_time + 85.0
    
    # Start with the seed odd-even transposition network
    # This is a valid sorting network
    seed = [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
    
    best = list(seed)
    
    # Initial minimization
    best = minimize_network(best, deadline)
    
    best_len = len(best)
    
    # Search loop: Try to improve by "kicking" (adding random comparators) 
    # and then minimizing. This helps escape local minima.
    random.seed(42)
    
    while time.time() < deadline - 5:
        # Strategy: Add 1-3 random comparators, then minimize.
        # If the minimized result is shorter than current best, update best.
        kick_size = random.randint(1, 3)
        candidate = list(best)
        
        # Add random comparators
        for _ in range(kick_size):
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            # Insert at a random position to disrupt order
            pos = random.randint(0, len(candidate))
            candidate.insert(pos, (i, j))
            
        # Minimize the kicked network
        candidate = minimize_network(candidate, deadline)
        
        if len(candidate) < best_len:
            best = candidate
            best_len = len(candidate)
            
            # If we found a significant improvement, we might want to 
            # try a more aggressive kick or just continue.
            # Resetting the 'kick' strategy isn't strictly necessary, 
            # but we can try to kick harder if we are stuck.
            
        # If we are stuck for a while, we could try swapping adjacent comparators
        # but the kick+minimize is a strong heuristic.
        
    return best

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END

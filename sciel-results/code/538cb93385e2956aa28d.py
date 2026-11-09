import numpy as np
import time
import random

N = 18
MAX_COMPARATORS = 4096
DEADLINE = 85.0  # Safety margin before 90s limit

def _precompute_inputs():
    """Precompute all 2^18 binary inputs as a numpy array."""
    # 2^18 is 262144, which is manageable.
    # We represent inputs as uint8 or int for speed.
    # Shape: (2^N, N)
    inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(1 << N):
        # Unpack bits
        val = i
        for j in range(N):
            inputs[i, j] = val & 1
            val >>= 1
    return inputs

# Precompute inputs at module level to avoid doing it inside construct() repeatedly 
# if construct is called multiple times (though usually it's called once per eval).
# However, 262144 * 18 bytes is ~4.7MB, which is fine.
# Let's do it lazily inside construct to ensure we don't waste time if we fail fast? 
# No, the verification is the bottleneck. Precomputing once is good.
# But construct() should be self-contained. Let's precompute inside construct 
# only once if not already done.

_INPUTS_CACHE = None

def _get_inputs():
    global _INPUTS_CACHE
    if _INPUTS_CACHE is None:
        _INPUTS_CACHE = _precompute_inputs()
    return _INPUTS_CACHE

def _verify_network(network, inputs=None):
    """
    Verify if the network sorts all binary inputs.
    network: list of (i, j) tuples
    returns: True if valid, False otherwise
    """
    if inputs is None:
        inputs = _get_inputs()
    
    # Make a copy to avoid modifying the original inputs
    state = inputs.copy()
    
    for i, j in network:
        # Extract columns
        col_i = state[:, i]
        col_j = state[:, j]
        
        # Compute min and max
        # min = (col_i + col_j - abs(col_i - col_j)) // 2
        # Or simpler with numpy:
        # min_val = np.minimum(col_i, col_j)
        # max_val = np.maximum(col_i, col_j)
        
        # Since they are 0/1, min is AND, max is OR? No, min(0,1)=0, max(0,1)=1.
        # min(0,0)=0, max(0,0)=0. min(1,1)=1, max(1,1)=1.
        # Yes, for 0/1: min = AND, max = OR.
        # Wait, min(0,1)=0 (0&1=0), max(0,1)=1 (0|1=1). Correct.
        
        state[:, i] = col_i & col_j
        state[:, j] = col_i | col_j
        
    # Check if sorted: state[:, k] <= state[:, k+1] for all k
    # This is equivalent to: no inversion exists.
    # An inversion is state[:, k] = 1 and state[:, k+1] = 0.
    # We can check if any row has a '1' followed by a '0'.
    
    # Efficient check:
    # For each adjacent pair, check if left is 1 and right is 0.
    # inversions = state[:, :-1] & ~state[:, 1:]
    # if np.any(inversions): return False
    
    # Even faster: 
    # Check if state is monotonically non-decreasing.
    # np.diff(state, axis=1) >= 0
    # diff = state[:, 1:] - state[:, :-1]
    # if np.any(diff < 0): return False
    
    # Let's use the diff method, it's very fast in numpy.
    diff = state[:, 1:] - state[:, :-1]
    if np.any(diff < 0):
        return False
    return True

def _generate_seed_network():
    """Generate the Odd-Even Transposition network."""
    network = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            network.append((i, i + 1))
    return network

def construct():
    start_time = time.time()
    
    # Initialize with a valid network
    current_network = _generate_seed_network()
    best_network = current_network.copy()
    best_score = len(best_network)
    
    inputs = _get_inputs()
    
    # Verify seed just in case
    if not _verify_network(current_network, inputs):
        # This shouldn't happen, but if it does, we have a bug.
        # Fall back to a simple bubble sort network (which is OE Transposition anyway)
        pass
    
    random.seed(42) # For reproducibility in debugging, remove for production? 
                    # Actually, random search benefits from different seeds. 
                    # But determinism is good for debugging. Let's use random time seed.
    random.seed() 

    # Local Search Parameters
    # We try to remove comparators. If that fails, we swap to escape local minima.
    removal_attempts = 0
    swap_attempts = 0
    max_removal_attempts = 1000
    max_swap_attempts = 500
    
    while time.time() - start_time < DEADLINE:
        # Try to remove a random comparator
        if len(current_network) > 0:
            idx = random.randint(0, len(current_network) - 1)
            candidate = current_network.copy()
            candidate.pop(idx)
            
            if _verify_network(candidate, inputs):
                # Valid improvement!
                current_network = candidate
                if len(current_network) < best_score:
                    best_network = current_network.copy()
                    best_score = len(best_network)
                removal_attempts = 0
                swap_attempts = 0
            else:
                removal_attempts += 1
        
        # If we've tried many removals without success, try swaps to perturb
        if removal_attempts > max_removal_attempts:
            # Try swapping two comparators
            if len(current_network) > 1:
                idx1 = random.randint(0, len(current_network) - 1)
                idx2 = random.randint(0, len(current_network) - 1)
                if idx1 != idx2:
                    candidate = current_network.copy()
                    candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
                    
                    if _verify_network(candidate, inputs):
                        # Swap is valid, score is same, but structure changed
                        current_network = candidate
                        removal_attempts = 0
                    else:
                        swap_attempts += 1
            
            if swap_attempts > max_swap_attempts:
                # Stuck in a deep local minimum? 
                # Try a more aggressive move: replace a comparator with a random one?
                # Or just reset the removal counter to try different indices.
                # For now, let's just reset counters to keep trying.
                # A better strategy: try to remove a comparator that is "redundant" based on heuristics?
                # For now, simple random restart of the search phase.
                removal_attempts = 0
                swap_attempts = 0
                max_removal_attempts = int(max_removal_attempts * 1.5) # Increase patience
                max_swap_attempts = int(max_swap_attempts * 1.5)

    return best_network

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()
# EVOLVE-BLOCK-END

# Wait, the interface requires construct() to be the entry point.
# The block above defines construct(). I need to make sure the logic is inside or called by it.
# Let's refactor to put the logic directly in construct or a helper.

def _construct_impl():
    start_time = time.time()
    
    # Initialize with a valid network
    current_network = _generate_seed_network()
    best_network = current_network.copy()
    best_score = len(best_network)
    
    inputs = _get_inputs()
    
    # Verify seed
    if not _verify_network(current_network, inputs):
        return best_network # Fallback, though shouldn't happen

    # Local Search
    removal_attempts = 0
    swap_attempts = 0
    max_removal_attempts = 500
    max_swap_attempts = 200
    
    while time.time() - start_time < DEADLINE:
        # Try to remove a random comparator
        if len(current_network) > 0:
            idx = random.randint(0, len(current_network) - 1)
            candidate = current_network.copy()
            candidate.pop(idx)
            
            if _verify_network(candidate, inputs):
                # Valid improvement!
                current_network = candidate
                if len(current_network) < best_score:
                    best_network = current_network.copy()
                    best_score = len(best_network)
                removal_attempts = 0
                swap_attempts = 0
            else:
                removal_attempts += 1
        
        # If we've tried many removals without success, try swaps to perturb
        if removal_attempts > max_removal_attempts:
            if len(current_network) > 1:
                idx1 = random.randint(0, len(current_network) - 1)
                idx2 = random.randint(0, len(current_network) - 1)
                if idx1 != idx2:
                    candidate = current_network.copy()
                    candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
                    
                    if _verify_network(candidate, inputs):
                        current_network = candidate
                        removal_attempts = 0
                    else:
                        swap_attempts += 1
            
            if swap_attempts > max_swap_attempts:
                removal_attempts = 0
                swap_attempts = 0
                max_removal_attempts = int(max_removal_attempts * 1.5)
                max_swap_attempts = int(max_swap_attempts * 1.5)

    return best_network

# The actual function required by the task contract
def construct():
    return _construct_impl()

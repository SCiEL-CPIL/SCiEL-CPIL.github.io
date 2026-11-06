N = 18
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

# Precompute all 2^N boolean inputs as a NumPy array
# Shape: (2^N, N), dtype: uint8
_M = 1 << N
# Generate indices 0 to 2^N - 1
_indices = np.arange(_M, dtype=np.uint8)
# Extract bits
# We need bit w of each index.
# index >> w & 1
# We can do this by looping w, or using a clever trick.
# Looping w is fine for 18 bits.
_inputs = np.zeros((_M, N), dtype=np.uint8)
for w in range(N):
    _inputs[:, w] = (_indices >> w) & 1

def _verify_network_numpy(comparators):
    """
    Verify if the network sorts all 2^N boolean inputs using NumPy.
    """
    if not comparators:
        return N <= 1
    
    # Copy initial state
    # We need a copy for each verification call to avoid side effects
    state = _inputs.copy()
    
    for i, j in comparators:
        # Find inputs where wire i is 1 and wire j is 0
        # i.e., state[:, i] > state[:, j]
        mask = state[:, i] > state[:, j]
        
        if not np.any(mask):
            continue
            
        # Swap values where mask is True
        # tmp_i = state[:, i]
        # state[:, i] = where(mask, state[:, j], state[:, i])
        # state[:, j] = where(mask, tmp_i, state[:, j])
        
        tmp_i = state[:, i].copy()
        state[:, i] = np.where(mask, state[:, j], state[:, i])
        state[:, j] = np.where(mask, tmp_i, state[:, j])
        
    # Check if sorted
    # Sorted means for all adjacent wires w, w+1: state[:, w] <= state[:, w+1]
    # If any row has a 1 followed by a 0, it's not sorted.
    # diff = state[:, 1:] - state[:, :-1]
    # If any diff < 0, it's not sorted.
    # Or simply: np.all(state[:, :-1] <= state[:, 1:])
    
    # To be faster, we can check if there is any inversion.
    # Inversion exists if state[:, w] == 1 and state[:, w+1] == 0 for any w.
    # Let's just use the comparison.
    if not np.all(state[:, :-1] <= state[:, 1:]):
        return False
        
    return True

def _odd_even_transposition(n=N):
    """Generate odd-even transposition sorting network."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 10s safety margin
    
    # Start with odd-even transposition
    best = _odd_even_transposition(N)
    best_len = len(best)
    
    # Verify seed
    if not _verify_network_numpy(best):
        # Should never happen
        return []
        
    random.seed(42)
    current = list(best)
    
    max_iterations = 100000
    consecutive_failures = 0
    max_consecutive_failures = 1000
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        if len(current) <= 75: # Stop if we get close to known best
            break
        if consecutive_failures > max_consecutive_failures:
            # Restart with fresh random seed from best
            random.seed(random.randint(0, 2**32 - 1))
            consecutive_failures = 0
            current = list(best)
        
        # Try random modification
        mod_type = random.random()
        
        if mod_type < 0.80 and len(current) > 1:
            # Try removing a random comparator
            idx = random.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_numpy(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
                    # Greedy cascade: try to remove more
                    for _ in range(10):
                        if time.time() > deadline:
                            break
                        if len(current) <= 75:
                            break
                        idx2 = random.randint(0, len(current) - 1)
                        candidate2 = current[:idx2] + current[idx2+1:]
                        if _verify_network_numpy(candidate2):
                            current = candidate2
                            if len(current) < best_len:
                                best = list(current)
                                best_len = len(current)
                        else:
                            break
            else:
                consecutive_failures += 1
        
        else:
            # Try swapping two comparators
            if len(current) < 2:
                continue
            i, j = random.sample(range(len(current)), 2)
            candidate = list(current)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if _verify_network_numpy(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                consecutive_failures += 1
    
    return best

# EVOLVE-BLOCK-END

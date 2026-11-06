"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136

# Precompute input matrix
# Shape: (2^N, N)
_indices = np.arange(1 << N, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = (_indices[:, None] & _bits[None, :]) > 0
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)

def is_sorted_network(comparators):
    """
    Check if the network sorts all 2^17 inputs using the zero-one principle.
    """
    if len(comparators) > 4096:
        return False
    
    # If empty, check if inputs are already sorted (only true for N=1 or specific cases, but here N=17)
    # Actually, if comparators is empty, it only works if all inputs are sorted, which is false.
    
    state = _INPUTS.copy()
    for i, j in comparators:
        # min and max
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
        
    return np.array_equal(state, _SORTED_INPUTS)

def construct():
    start_time = time.time()
    deadline = start_time + 58.0 # Safety margin
    
    # 1. Generate Seed Network (Odd-Even Transposition)
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
            
    # Verify seed
    if not is_sorted_network(seed_network):
        # Fallback, though unlikely
        return seed_network

    current = list(seed_network)
    best = list(current)
    best_len = len(current)
    
    # 2. Aggressive Pruning Phase
    # Try to remove any comparator that is redundant.
    # We do this in multiple passes because removing one might make another redundant.
    changed = True
    while changed and time.time() < deadline - 10.0:
        changed = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 10.0:
                break
            candidate = current.copy()
            candidate.pop(idx)
            if is_sorted_network(candidate):
                current = candidate
                best = current.copy()
                best_len = len(current)
                changed = True
                # Reset idx to restart the pass? Or continue? 
                # Continuing might miss new redundancies created by this removal 
                # that are at lower indices? No, removing at idx affects higher indices?
                # Actually, removing a comparator can make *other* comparators redundant.
                # It's safer to restart the pass or just continue and do another full pass.
                # Let's just break the inner loop and let the while loop restart.
                break
        if changed:
            continue
        # If no change in a full pass, we are at a local minimum for removals.
        break

    # 3. Simulated Annealing / Local Search
    # We will try to improve further by swapping, moving, replacing.
    T = 5.0 # Initial temperature
    T_min = 0.1
    cooling_rate = 0.995
    
    current = best.copy()
    current_len = best_len
    
    # Pre-generate some random comparators for speed?
    # Or just generate on the fly.
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate a neighbor
        move_type = random.randint(0, 4)
        neighbor = current.copy()
        
        if move_type == 0: # Remove
            if not neighbor:
                continue
            idx = random.randint(0, len(neighbor) - 1)
            neighbor.pop(idx)
            
        elif move_type == 1: # Add
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            idx = random.randint(0, len(neighbor))
            neighbor.insert(idx, (i, j))
            
        elif move_type == 2: # Swap
            if len(neighbor) < 2:
                continue
            i1 = random.randint(0, len(neighbor) - 1)
            i2 = random.randint(0, len(neighbor) - 1)
            if i1 != i2:
                neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]
                
        elif move_type == 3: # Move
            if len(neighbor) < 2:
                continue
            i1 = random.randint(0, len(neighbor) - 1)
            i2 = random.randint(0, len(neighbor) - 1)
            if i1 != i2:
                item = neighbor.pop(i1)
                neighbor.insert(i2, item)
                
        else: # Replace
            if not neighbor:
                continue
            idx = random.randint(0, len(neighbor) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            neighbor[idx] = (i, j)

        # Check validity
        if is_sorted_network(neighbor):
            neighbor_len = len(neighbor)
            delta_e = neighbor_len - current_len
            
            if delta_e <= 0:
                # Improvement or equal
                current = neighbor
                current_len = neighbor_len
                if current_len < best_len:
                    best = current.copy()
                    best_len = current_len
            else:
                # Worse, accept with probability
                prob = np.exp(-delta_e / T)
                if random.random() < prob:
                    current = neighbor
                    current_len = neighbor_len
        else:
            # Invalid, reject (or accept with very low prob? No, invalid is bad)
            pass
            
        T *= cooling_rate
        
    # Final check
    if is_sorted_network(best):
        return best
    else:
        return seed_network

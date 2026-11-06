import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136

# Precompute inputs
def _make_all_inputs():
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    inputs = ((indices >> bit_positions) & 1).astype(np.uint8)
    return inputs

_ALL_INPUTS = _make_all_inputs()
# Random subset for fast checking
# Use a fixed seed for reproducibility of the subset itself
random.seed(12345)
_SUBSET_SIZE = 4096
_SUBSET_INDICES = np.array(random.sample(range(1 << N), _SUBSET_SIZE), dtype=np.int32)
_SUBSET_INPUTS = _ALL_INPUTS[_SUBSET_INDICES]

def simulate_network(comparators, inputs):
    """
    Simulate a compare-exchange network on given inputs.
    """
    state = inputs.copy()
    for (i, j) in comparators:
        mi = state[:, i] & state[:, j]
        mj = state[:, i] | state[:, j]
        state[:, i] = mi
        state[:, j] = mj
    return state

def is_sorted_output(outputs):
    """
    Check if the output rows are sorted (non-decreasing).
    """
    # For binary, sorted means no 1 before 0.
    # Check adjacent pairs: output[:, k] <= output[:, k+1]
    # This is equivalent to: (output[:, k] & ~output[:, k+1]) == 0 for all k
    # Or simply: np.any(output[:, :-1] > output[:, 1:])
    return not np.any(outputs[:, :-1] > outputs[:, 1:])

def fast_check(comparators, watchlist_indices):
    """
    Check against subset and watchlist.
    Returns True if passes, False otherwise.
    """
    # Check subset
    out_sub = simulate_network(comparators, _SUBSET_INPUTS)
    if not is_sorted_output(out_sub):
        return False
    
    # Check watchlist
    if watchlist_indices:
        out_watch = simulate_network(comparators, _ALL_INPUTS[watchlist_indices])
        if not is_sorted_output(out_watch):
            return False
            
    return True

def full_check(comparators, watchlist_indices):
    """
    Check against all inputs.
    If it fails, updates watchlist_indices with a failing input.
    Returns True if valid, False otherwise.
    """
    out_all = simulate_network(comparators, _ALL_INPUTS)
    if not is_sorted_output(out_all):
        # Find a failing input to add to watchlist
        # We want an index where the row is not sorted
        # failed_mask: True for rows that are NOT sorted
        failed_mask = np.any(out_all[:, :-1] > out_all[:, 1:], axis=1)
        if np.any(failed_mask):
            failing_indices = np.where(failed_mask)[0]
            # Add one random failing index to watchlist
            idx = random.choice(failing_indices)
            if idx not in watchlist_indices:
                watchlist_indices.append(idx)
        return False
    return True

def get_neighbor(comparators, move_type):
    """
    Generate a neighbor based on move_type.
    """
    n = len(comparators)
    if n == 0:
        # Must insert
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        return [(i, j)], 1 # +1 cost

    if move_type == 'remove' and n > 0:
        idx = random.randint(0, n - 1)
        new_comp = comparators[:idx] + comparators[idx+1:]
        return new_comp, -1 # -1 cost

    elif move_type == 'insert':
        pos = random.randint(0, n)
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        new_comp = comparators[:pos] + [(i, j)] + comparators[pos:]
        return new_comp, 1 # +1 cost

    elif move_type == 'swap' and n > 1:
        i1 = random.randint(0, n - 1)
        i2 = random.randint(0, n - 1)
        if i1 == i2:
            return comparators, 0
        new_comp = comparators[:]
        new_comp[i1], new_comp[i2] = new_comp[i2], new_comp[i1]
        return new_comp, 0

    elif move_type == 'move' and n > 1:
        i1 = random.randint(0, n - 1)
        i2 = random.randint(0, n - 1)
        if i1 == i2:
            return comparators, 0
        new_comp = comparators[:]
        # Remove from i1
        comp_to_move = new_comp.pop(i1)
        # Insert at i2 (adjust index if i1 < i2)
        if i1 < i2:
            i2 -= 1
        new_comp.insert(i2, comp_to_move)
        return new_comp, 0

    else:
        # Fallback
        return comparators, 0

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s for final check and return
    
    # Seed: Odd-Even Transposition
    seed = [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
    
    # Verify seed
    if not full_check(seed, []):
        raise ValueError("Seed is invalid")
        
    best_comparators = seed
    best_cost = len(seed)
    current_comparators = seed
    current_cost = len(seed)
    
    watchlist = []
    
    # Simulated Annealing parameters
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.01
    
    random.seed(42)
    
    while time.time() < deadline:
        # Choose a move
        # Weight moves: Remove is most important, Insert is needed to escape
        r = random.random()
        if r < 0.3:
            move_type = 'remove'
        elif r < 0.5:
            move_type = 'insert'
        elif r < 0.7:
            move_type = 'swap'
        else:
            move_type = 'move'
            
        neighbor, delta_cost = get_neighbor(current_comparators, move_type)
        
        # If we tried to remove from empty list, skip
        if len(neighbor) == 0 and move_type == 'remove':
            continue
            
        # Fast check
        if not fast_check(neighbor, watchlist):
            continue
            
        new_cost = current_cost + delta_cost
        
        # Acceptance criterion
        if new_cost < current_cost:
            # Always accept improvement
            current_comparators = neighbor
            current_cost = new_cost
            
            if new_cost < best_cost:
                # Full check before updating best
                if full_check(neighbor, watchlist):
                    best_comparators = neighbor
                    best_cost = new_cost
                    # Reset watchlist? No, keep it. It helps avoid known bad configs.
                else:
                    # If full check failed, revert current? 
                    # The fast check passed, so it's likely a hard-to-find failure.
                    # We accept it as 'current' to explore around it, but don't update 'best'.
                    # However, if it's invalid, we should probably revert to a valid state.
                    # Let's revert current to last valid best if this fails full check.
                    current_comparators = best_comparators
                    current_cost = best_cost
        else:
            # Worse or same cost
            # Accept with probability exp(-delta_cost / T)
            # delta_cost is positive if we increased size
            if delta_cost > 0:
                prob = np.exp(-delta_cost / T)
                if random.random() < prob:
                    # Accept worse solution to escape local minimum
                    # But we must ensure it's valid!
                    # Fast check passed. Full check is expensive.
                    # We will assume it's valid for the sake of exploration, 
                    # but if it turns out invalid later, we might be stuck.
                    # To be safe, only accept worse solutions if they pass a 
                    # slightly stricter fast check? Or just take the risk?
                    # Given the 90s limit, taking the risk is better than stalling.
                    # We will update current.
                    current_comparators = neighbor
                    current_cost = new_cost
            # If delta_cost == 0, we can accept to shuffle the network
            else:
                if random.random() < 0.1: # Low probability to accept neutral moves
                    current_comparators = neighbor
                    current_cost = new_cost
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T

    # Final verification of best
    if not full_check(best_comparators, watchlist):
        # Should not happen if best was always fully checked
        # But just in case, fallback to seed
        return seed
        
    return best_comparators

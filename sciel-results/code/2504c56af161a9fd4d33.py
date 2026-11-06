import time
import numpy as np
import random

N = 17
DEADLINE = 55.0  # Strict safety margin
START_TIME = time.time()
SUBSET_SIZE = 2048

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def _make_all_inputs():
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    return ((indices >> bit_positions) & 1).astype(np.uint8)

_ALL_INPUTS = None
_SUBSET_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS

def _get_subset_inputs():
    global _SUBSET_INPUTS
    if _SUBSET_INPUTS is None:
        all_inputs = _get_all_inputs()
        # Random sample
        indices = np.random.choice(len(all_inputs), SUBSET_SIZE, replace=False)
        _SUBSET_INPUTS = all_inputs[indices]
    return _SUBSET_INPUTS

def simulate(comps, data):
    """Simulate network on given data. Returns final uint8 array."""
    d = data.copy()
    for a, b in comps:
        lo = np.minimum(d[:, a], d[:, b])
        hi = np.maximum(d[:, a], d[:, b])
        d[:, a] = lo
        d[:, b] = hi
    return d

def count_unsorted(comps, data):
    """Count number of rows that are not sorted."""
    final = simulate(comps, data)
    # A row is sorted if final[:, i] <= final[:, i+1] for all i
    # For binary, this means no '10' pattern.
    # We can check if any adjacent pair is (1, 0).
    # diff = final[:, 1:] - final[:, :-1]
    # If diff < 0, it's unsorted.
    diffs = final[:, 1:] - final[:, :-1]
    return np.sum(diffs < 0)

def construct():
    global START_TIME
    START_TIME = time.time()
    
    all_inputs = _get_all_inputs()
    subset_inputs = _get_subset_inputs()
    
    # Phase 1: Greedy Seed
    # Start with empty, add comparators that reduce inversions on subset
    comps = []
    current_data_subset = subset_inputs.copy()
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Fast greedy on subset
    while _time_left() > 50.0: # Leave 50s for SA
        best_gain = 0
        best_pair = None
        # Sample a few pairs to speed up? No, N=17, pairs=136. 136 * 2048 * 2 ops is fast.
        for i, j in pairs:
            gain = np.sum(current_data_subset[:, i] > current_data_subset[:, j])
            if gain > best_gain:
                best_gain = gain
                best_pair = (i, j)
        
        if best_pair is None or best_gain == 0:
            break
            
        i, j = best_pair
        lo = np.minimum(current_data_subset[:, i], current_data_subset[:, j])
        hi = np.maximum(current_data_subset[:, i], current_data_subset[:, j])
        current_data_subset[:, i] = lo
        current_data_subset[:, j] = hi
        comps.append(best_pair)
        
        # Check if subset is sorted
        if np.all(current_data_subset[:, :-1] <= current_data_subset[:, 1:]):
            break
            
    # Verify seed on full set
    if count_unsorted(comps, all_inputs) > 0:
        # Fallback: Use Odd-Even Transposition if greedy failed to sort full set (unlikely on subset)
        # Generate Odd-Even Transposition
        comps = []
        for k in range(N):
            for i in range(0, N - 1, 2) if k % 2 == 0 else range(1, N - 1, 2):
                if i + 1 < N:
                    comps.append((i, i + 1))
        # This is 136 comparators.
        
    best_comps = comps[:]
    best_size = len(comps)
    
    # Phase 2: Simulated Annealing
    current_comps = comps[:]
    current_size = len(comps)
    current_sub_penalty = count_unsorted(current_comps, subset_inputs)
    
    T = 10.0
    T_min = 0.1
    # We want to run as many iterations as possible
    # Estimate time per iteration: ~1ms (subset eval) + occasional full eval.
    
    while _time_left() > 5.0:
        # Generate Move
        move_type = random.random()
        new_comps = current_comps[:]
        
        if move_type < 0.4 and len(new_comps) > 0:
            # Delete
            idx = random.randrange(len(new_comps))
            new_comps.pop(idx)
        elif move_type < 0.7:
            # Swap Wires
            if len(new_comps) > 0:
                idx = random.randrange(len(new_comps))
                i, j = new_comps[idx]
                new_i, new_j = random.sample(range(N), 2)
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                new_comps[idx] = (new_i, new_j)
            else:
                # Insert
                i, j = random.sample(range(N), 2)
                if i > j: i, j = j, i
                new_comps.append((i, j))
        elif move_type < 0.9:
            # Move
            if len(new_comps) > 1:
                idx = random.randrange(len(new_comps))
                c = new_comps.pop(idx)
                new_idx = random.randrange(len(new_comps) + 1)
                new_comps.insert(new_idx, c)
            else:
                # Insert
                i, j = random.sample(range(N), 2)
                if i > j: i, j = j, i
                new_comps.append((i, j))
        else:
            # Insert
            i, j = random.sample(range(N), 2)
            if i > j: i, j = j, i
            idx = random.randrange(len(new_comps) + 1)
            new_comps.insert(idx, (i, j))
            
        new_size = len(new_comps)
        new_sub_penalty = count_unsorted(new_comps, subset_inputs)
        
        # Calculate Energy
        # E = size + lambda * penalty
        # If penalty is 0 on subset, we should check full set to see if it's truly valid
        # To save time, we only check full set if subset penalty is 0 AND (size < best_size OR random())
        
        E_old = current_size + 10.0 * current_sub_penalty
        E_new = new_size + 10.0 * new_sub_penalty
        
        accept = False
        if new_sub_penalty == 0:
            # Check full validity
            full_penalty = count_unsorted(new_comps, all_inputs)
            if full_penalty == 0:
                # Valid network
                if new_size < best_size:
                    best_size = new_size
                    best_comps = new_comps[:]
                # Accept if better or equal size, or with prob
                if new_size <= current_size:
                    accept = True
                else:
                    # Accept larger valid network? Maybe, to explore.
                    # But we want to minimize size.
                    # If it's valid, we can keep it if T is high.
                    dE = E_new - E_old
                    if dE < 0 or random.random() < np.exp(-dE / T):
                        accept = True
            else:
                # Invalid, but subset was 0. High penalty.
                # Reject unless T is very high?
                # E_new effectively has high penalty.
                pass
        else:
            # Subset penalty > 0
            dE = E_new - E_old
            if dE < 0 or random.random() < np.exp(-dE / T):
                accept = True
                
        if accept:
            current_comps = new_comps
            current_size = new_size
            current_sub_penalty = new_sub_penalty
            
        # Cool down
        T *= 0.999
        if T < T_min:
            T = T_min
            
    # Phase 3: Redundancy Removal on best_comps
    # Try to remove each comparator
    improved = True
    while improved and _time_left() > 1.0:
        improved = False
        for idx in range(len(best_comps) - 1, -1, -1):
            if _time_left() < 0.5:
                break
            new_comps = best_comps[:idx] + best_comps[idx + 1:]
            if count_unsorted(new_comps, all_inputs) == 0:
                best_comps = new_comps
                improved = True
                break

    return best_comps

if __name__ == "__main__":
    result = construct()
    print(f"Comparators: {len(result)}")

import time
import random
import numpy as np

def _score_bitset(mask):
    """Count distinct d>0 such that a, a+d, a+2d all in set."""
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _arr_to_mask(arr):
    mask = 0
    for x in arr:
        mask |= (1 << (x - 1))
    return mask

def _mask_to_arr(mask):
    arr = []
    for x in range(1, 513):
        if mask & (1 << (x - 1)):
            arr.append(x)
    return arr

def _covered_diffs_mask(mask):
    """Return set of d values that are covered."""
    covered = set()
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            covered.add(d)
    return covered

def _precompute_ap_info():
    """Precompute for each position and each d, which other positions form a 3-AP."""
    # ap_pairs[d][i] = list of j such that i, i+d, i+2d or i-d, i, i+d or i-2d, i-d, i are in range
    # Actually, let's precompute: for each element position p (1..512) and each d (1..255),
    # what are the "partners" needed to complete a 3-AP with difference d?
    # If we have element at position p, then for difference d:
    #   - p could be the start: need p+d and p+2d
    #   - p could be the middle: need p-d and p+d
    #   - p could be the end: need p-2d and p-d
    # So for each (p, d), we get up to 3 pairs of required positions.
    
    # Let's create a structure: for each d, a list of all possible 3-APs (as tuples of 3 positions)
    # Then for scoring, we check if all 3 positions are in the set.
    
    # Actually, the bitset approach is already fast. Let's use a different optimization:
    # Precompute for each d, a bitmask of all positions that could be part of a 3-AP with difference d.
    # Then (mask & ap_mask[d] & (mask >> d) & (mask >> 2*d)) != 0 checks if any 3-AP with diff d exists.
    
    # Wait, that's not quite right. Let me think again.
    # For a fixed d, a 3-AP (a, a+d, a+2d) exists in the set iff there's some a such that
    # a, a+d, a+2d are all in the set. This is equivalent to:
    # (mask & (mask >> d) & (mask >> 2*d)) != 0
    # This is already O(1) per d with bit operations. The loop over 255 values of d is the bottleneck.
    
    # Alternative: use numpy to vectorize the check over all d at once?
    # Let's try a different approach: represent the set as a numpy boolean array and use
    # vectorized operations.
    
    pass

def _score_numpy(arr):
    """Score using numpy for speed."""
    # Convert to boolean array
    present = np.zeros(512, dtype=bool)
    for x in arr:
        present[x-1] = True
    
    count = 0
    # For each d, check if there's a 3-AP
    # We can vectorize this: for a given d, we need present[i] & present[i+d] & present[i+2*d] for some i
    # This is equivalent to: np.any(present[:512-2*d] & present[d:512-d] & present[2*d:])
    
    for d in range(1, 256):
        if np.any(present[:512-2*d] & present[d:512-d] & present[2*d:]):
            count += 1
    return count

def _fast_score(arr):
    """Fast scoring using bitset."""
    mask = _arr_to_mask(arr)
    return _score_bitset(mask)

def _sa_search(arr, deadline, seed=42):
    """Simulated annealing with swap-based moves."""
    rng = random.Random(seed)
    cur_arr = sorted(arr[:])
    cur_mask = _arr_to_mask(cur_arr)
    cur_score = _score_bitset(cur_mask)
    
    best_arr = cur_arr[:]
    best_score = cur_score
    
    # Temperature schedule
    t0 = 10.0
    t_final = 0.01
    # We'll estimate the number of iterations we can do
    # Each iteration: pick random i, random j, random new value, evaluate score
    # Let's time a few iterations to estimate
    
    start = time.time()
    # Estimate iterations per second
    test_start = time.time()
    test_iters = 0
    while time.time() - test_start < 0.1:
        test_iters += 1
    iters_per_sec = test_iters / 0.1
    
    total_budget = deadline - time.time()
    max_iters = int(total_budget * iters_per_sec * 0.8)
    
    # Simulated annealing
    for it in range(max_iters):
        if time.time() > deadline - 2:
            break
        
        # Temperature
        progress = it / max(max_iters, 1)
        T = t0 * (t_final / t0) ** progress
        
        # Pick a random position to change
        i = rng.randint(0, 31)
        old_val = cur_arr[i]
        
        # Pick a new value (random from 1..512, not already in set)
        # To make it more efficient, pick from positions that might help
        # For now, just random
        new_val = rng.randint(1, 512)
        
        if new_val == old_val:
            continue
        
        # Check if new_val is already in the set
        if new_val in cur_arr:
            continue
        
        # Try the move
        cur_arr[i] = new_val
        trial_arr = sorted(cur_arr)
        trial_mask = _arr_to_mask(trial_arr)
        trial_score = _score_bitset(trial_mask)
        
        delta = trial_score - cur_score
        
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            # Accept the move
            cur_arr = trial_arr
            cur_mask = trial_mask
            cur_score = trial_score
            
            if cur_score > best_score:
                best_score = cur_score
                best_arr = cur_arr[:]
        else:
            # Reject
            cur_arr[i] = old_val
    
    return best_arr, best_score

def _greedy_construct(deadline, seed=42):
    """Greedy construction: start with dense seeds, then add elements that maximize new covered differences."""
    rng = random.Random(seed)
    
    # Precompute "density" of each position: how many 3-APs it participates in
    # A position p participates in a 3-AP with difference d if:
    #   - p is start: p+d, p+2d in range
    #   - p is middle: p-d, p+d in range  
    #   - p is end: p-2d, p-d in range
    # Count total (d, role) pairs for each position
    
    density = np.zeros(512, dtype=int)
    for d in range(1, 256):
        for a in range(1, 513 - 2*d + 1):
            # 3-AP: a, a+d, a+2d
            density[a-1] += 1
            density[a+d-1] += 1
            density[a+2*d-1] += 1
    
    # Start with the 8 densest positions (spread out to avoid clustering)
    # Actually, let's start with a few random dense positions
    top_dense = np.argsort(density)[::-1]
    
    # Pick initial 8 positions from top dense, ensuring they're somewhat spread
    cur_set = set()
    for pos in top_dense[:20]:
        if len(cur_set) >= 8:
            break
        cur_set.add(pos + 1)  # 1-indexed
    
    if len(cur_set) < 8:
        for pos in top_dense[20:]:
            if len(cur_set) >= 8:
                break
            cur_set.add(pos + 1)
    
    cur_arr = sorted(cur_set)
    cur_mask = _arr_to_mask(cur_arr)
    cur_covered = _covered_diffs_mask(cur_mask)
    
    # Greedily add elements
    while len(cur_arr) < 32 and time.time() < deadline - 5:
        best_gain = -1
        best_elem = None
        best_new_covered = None
        
        # Check all possible elements
        for cand in range(1, 513):
            if cand in cur_set:
                continue
            
            trial_mask = cur_mask | (1 << (cand - 1))
            trial_covered = _covered_diffs_mask(trial_mask)
            gain = len(trial_covered - cur_covered)
            
            if gain > best_gain:
                best_gain = gain
                best_elem = cand
                best_new_covered = trial_covered
        
        if best_elem is None or best_gain == 0:
            # No improvement; pick random
            available = [x for x in range(1, 513) if x not in cur_set]
            if not available:
                break
            best_elem = rng.choice(available)
            cur_set.add(best_elem)
            cur_arr = sorted(cur_set)
            cur_mask = _arr_to_mask(cur_arr)
            cur_covered = _covered_diffs_mask(cur_mask)
        else:
            cur_set.add(best_elem)
            cur_arr = sorted(cur_set)
            cur_mask = _arr_to_mask(cur_arr)
            cur_covered = best_new_covered
    
    return cur_arr, cur_mask, cur_covered

def _local_search(arr, deadline, seed=42):
    """Local search: try swaps and replacements to improve score."""
    rng = random.Random(seed)
    cur_arr = sorted(arr[:])
    cur_mask = _arr_to_mask(cur_arr)
    cur_score = _score_bitset(cur_mask)
    
    best_arr = cur_arr[:]
    best_score = cur_score
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Try each position, replace with a random value
        for i in range(32):
            if time.time() > deadline - 5:
                break
            
            old_val = cur_arr[i]
            
            # Try several random replacements
            for _ in range(5):
                new_val = rng.randint(1, 512)
                if new_val == old_val or new_val in cur_arr:
                    continue
                
                cur_arr[i] = new_val
                trial_arr = sorted(cur_arr)
                trial_mask = _arr_to_mask(trial_arr)
                trial_score = _score_bitset(trial_mask)
                
                if trial_score > cur_score:
                    cur_mask = trial_mask
                    cur_score = trial_score
                    improved = True
                    
                    if cur_score > best_score:
                        best_score = cur_score
                        best_arr = cur_arr[:]
                    break
                else:
                    cur_arr[i] = old_val
            
            if improved:
                break
    
    return best_arr, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    best_arr = None
    best_score = 0
    
    # Phase 1: Multiple greedy constructions with different seeds
    num_greedy = 5
    greedy_deadline = start_time + (deadline - start_time) * 0.25
    
    for g in range(num_greedy):
        if time.time() >= greedy_deadline:
            break
        arr, mask, covered = _greedy_construct(greedy_deadline, seed=42 + g * 1000)
        score = _score_bitset(mask)
        if score > best_score:
            best_score = score
            best_arr = arr
    
    # Phase 2: Simulated annealing on best
    if time.time() < deadline - 30:
        sa_deadline = time.time() + (deadline - time.time()) * 0.4
        best_arr, best_score = _sa_search(best_arr, sa_deadline, seed=42)
    
    # Phase 3: More SA restarts with different seeds
    if time.time() < deadline - 20:
        remaining = deadline - time.time()
        num_sa = 3
        sa_budget = remaining / (num_sa + 1)
        
        for s in range(num_sa):
            if time.time() >= deadline - 5:
                break
            
            # Perturb the best solution
            perturbed = best_arr[:]
            for _ in range(4):
                i = random.randint(0, 31)
                new_val = random.randint(1, 512)
                if new_val not in perturbed:
                    perturbed[i] = new_val
            perturbed = sorted(set(perturbed))
            while len(perturbed) < 32:
                extra = random.randint(1, 512)
                if extra not in perturbed:
                    perturbed.append(extra)
            perturbed = sorted(perturbed[:32])
            
            sa_deadline = time.time() + sa_budget
            arr, score = _sa_search(perturbed, sa_deadline, seed=1000 + s * 777)
            
            if score > best_score:
                best_score = score
                best_arr = arr
    
    # Phase 4: Final local search
    if time.time() < deadline - 3:
        best_arr, best_score = _local_search(best_arr, deadline, seed=42)
    
    # Final validation
    result = sorted(set(best_arr))
    if len(result) != 32:
        result = list(range(1, 33))
    result = [x for x in result if 1 <= x <= 512]
    if len(result) < 32:
        extra = [x for x in range(1, 513) if x not in result]
        result.extend(extra[:32 - len(result)])
        result = sorted(result[:32])
    
    return result

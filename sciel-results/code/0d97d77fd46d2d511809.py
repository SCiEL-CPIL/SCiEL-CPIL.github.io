import time
import random
import sys

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin before 90s kill

def _exact_max_sidon_subset_size(arr, time_limit=0.05):
    """
    Exact B&B solver to find the maximum Sidon subset size.
    Optimized for N=12. Returns the size.
    Uses a boolean array for sums to speed up membership tests.
    Added sum-count upper bound for pruning.
    """
    n = len(arr)
    if n <= 1:
        return n
    
    # Sort to help with pruning
    arr_sorted = sorted(arr)
    best = 0
    start_time = time.time()
    
    # Sums range from 0 to 256 (max 128+128). Size 257.
    # We use a list of booleans for the current sums.
    # To avoid copying the list at every recursion step (which is expensive),
    # we can use a "undo" stack or just copy if n is small.
    # n=12, depth=12. Copying a list of 257 ints/bools is fast enough.
    
    # Precompute sum of squares bound? No, just simple sum count bound.
    # Max k such that k(k+1)/2 <= 257
    # k=22 -> 22*23/2 = 253. k=23 -> 276.
    # So max possible is 22.
    
    def dfs(idx, current_size, current_sums, current_elements):
        nonlocal best, start_time
        
        # Time check
        if time.time() - start_time > time_limit:
            return False
            
        # Pruning: if we can't beat best
        if current_size + (n - idx) <= best:
            return False
            
        # Update best
        if current_size > best:
            best = current_size
            if best == n:
                return True # Can't do better than n
            
        if idx == n:
            return False
        
        # Stronger Pruning: Sum Count Bound
        # Number of used sums
        used_sums = sum(1 for s in current_sums if s)
        available_sums = 257 - used_sums
        # Max additional elements 'k' that can be added such that k(k+1)/2 <= available_sums
        # Note: This assumes the new elements don't create sums that collide with EACH OTHER 
        # OR with EXISTING sums. The bound k(k+1)/2 <= available_sums is necessary but not sufficient.
        # It says: The new subset of size k must have k(k+1)/2 distinct sums.
        # All these sums must be in the available space.
        # So k(k+1)/2 <= available_sums.
        # Solve for k: k^2 + k - 2*available_sums <= 0
        # k <= (-1 + sqrt(1 + 8*available_sums)) / 2
        
        max_addable = int((-1 + (1 + 8 * available_sums) ** 0.5) / 2)
        
        if current_size + max_addable <= best:
            return False
            
        x = arr_sorted[idx]
        
        # Option 1: Include x
        # Check if x can be added to current_elements
        can_add = True
        new_sums = current_sums[:]  # Copy the boolean array
        
        # Check collisions with existing elements
        for y in current_elements:
            s = x + y
            if new_sums[s]:
                can_add = False
                break
            new_sums[s] = True
        
        if can_add:
            s_diag = 2 * x
            if new_sums[s_diag]:
                can_add = False
            else:
                new_sums[s_diag] = True
        
        if can_add:
            # Try including x first (often finds larger solutions faster, improving pruning)
            if dfs(idx + 1, current_size + 1, new_sums, current_elements + [x]):
                return True
        
        # Option 2: Exclude x
        if dfs(idx + 1, current_size, current_sums, current_elements):
            return True
            
        return False

    # Initial sums array: all False
    init_sums = [False] * 257
    dfs(0, 0, init_sums, [])
    return best

def construct():
    global _START_TIME, _DEADLINE
    
    # Ensure valid output: 12 distinct integers in [0, 128]
    
    candidates = []
    
    # 1. Arithmetic Progressions (known to have low Sidon numbers)
    # Step 1: 0..11
    candidates.append(list(range(12)))
    # Step 2: 0, 2, ..., 22
    candidates.append(list(range(0, 24, 2)))
    # Step 3: 0, 3, ..., 33
    candidates.append(list(range(0, 36, 3)))
    # Step 4: 0, 4, ..., 44
    candidates.append(list(range(0, 48, 4)))
    # Step 5: 0, 5, ..., 55
    candidates.append(list(range(0, 60, 5)))
    # Step 6: 0, 6, ..., 66
    candidates.append(list(range(0, 72, 6)))
    # Step 7: 0, 8, ..., 88
    candidates.append(list(range(0, 96, 8)))
    # Step 8: 0, 10, ..., 110
    candidates.append(list(range(0, 120, 10)))
    # Step 9: 0, 12, ..., 132 (out of range for 128? 11*12=132 > 128)
    # 0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132 -> 132 is out.
    # So max step for 12 elements in [0,128] is floor(128/11) = 11.
    candidates.append(list(range(0, 132, 11))) # 0, 11, ..., 121

    # 2. Dense Clusters
    # Two clusters
    candidates.append([0, 1, 2, 3, 4, 5, 60, 61, 62, 63, 64, 65])
    candidates.append([0, 1, 2, 3, 4, 5, 64, 65, 66, 67, 68, 69])
    # Three clusters
    candidates.append([0, 1, 2, 3, 42, 43, 44, 45, 84, 85, 86, 87])
    
    # 3. Perturbed APs
    # Try to break the AP structure slightly to see if it reduces max Sidon size further
    # or if it's already minimal.
    base_ap = list(range(12))
    rng = random.Random(42)
    for _ in range(50):
        c = base_ap[:]
        idx = rng.randint(0, 11)
        # Shift one element slightly
        shift = rng.choice([-2, -1, 1, 2])
        new_val = c[idx] + shift
        if 0 <= new_val <= 128 and new_val not in c:
            c[idx] = new_val
            candidates.append(c)
            
    # 4. Random dense sets in small range
    for _ in range(100):
        c = sorted(rng.sample(range(0, 20), 12))
        candidates.append(c)
        
    # 5. Random dense sets in medium range
    for _ in range(100):
        c = sorted(rng.sample(range(0, 40), 12))
        candidates.append(c)

    best_arr = None
    best_score = 12 # Max possible
    
    # Evaluate candidates
    for c in candidates:
        if len(set(c)) != 12:
            continue
        if any(x < 0 or x > 128 for x in c):
            continue
        # Use a tight time limit for initial screening
        score = _exact_max_sidon_subset_size(c, time_limit=0.02)
        if score < best_score:
            best_score = score
            best_arr = c[:]
            
    # If no candidate worked (shouldn't happen), fallback
    if best_arr is None:
        best_arr = list(range(12))
        best_score = _exact_max_sidon_subset_size(best_arr, time_limit=0.05)
        
    # Local Search / Simulated Annealing
    # Objective: Minimize best_score (max Sidon subset size)
    # Move: Change one element to create more sum collisions
    
    current_arr = best_arr[:]
    current_score = best_score
    best_arr_found = best_arr[:]
    best_score_found = best_score
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.995
    
    rng = random.Random(123)
    
    # Helper to check if a value is valid for the array
    def is_valid(val, arr, idx):
        if val < 0 or val > 128:
            return False
        if val == arr[idx]:
            return False
        for j, v in enumerate(arr):
            if j != idx and v == val:
                return False
        return True

    while time.time() < _DEADLINE:
        # Generate neighbor
        new_arr = current_arr[:]
        idx = rng.randint(0, 11)
        old_val = new_arr[idx]
        
        # Strategy: Try to move the element to a value that causes a sum collision
        # with existing pairs.
        # Or just random perturbation.
        # Let's try a random perturbation first, but biased towards the current value.
        # If current value is v, try v + delta where delta is small.
        if rng.random() < 0.5:
            # Small perturbation
            delta = rng.randint(-5, 5)
            new_val = old_val + delta
        else:
            # Random jump
            new_val = rng.randint(0, 128)
            
        # Ensure distinctness and range
        if not is_valid(new_val, new_arr, idx):
            continue
            
        new_arr[idx] = new_val
        new_arr.sort() # Keep sorted for solver efficiency
        
        # Evaluate
        new_score = _exact_max_sidon_subset_size(new_arr, time_limit=0.02)
        
        delta_score = new_score - current_score
        
        if delta_score < 0:
            # Improvement
            current_arr = new_arr
            current_score = new_score
            if current_score < best_score_found:
                best_score_found = current_score
                best_arr_found = new_arr[:]
        else:
            # Accept with probability
            if delta_score == 0:
                if rng.random() < 0.5:
                    current_arr = new_arr
            else:
                # Since score is integer, delta_score >= 1
                # exp(-1/T)
                prob = 2.718281828 ** (-delta_score / T)
                if rng.random() < prob:
                    current_arr = new_arr
                    current_score = new_score
                    
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Final validation
    assert len(best_arr_found) == 12
    assert len(set(best_arr_found)) == 12
    assert all(0 <= x <= 128 for x in best_arr_found)
    
    return best_arr_found

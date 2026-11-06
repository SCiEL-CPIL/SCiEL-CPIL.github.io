"""Simulated Annealing approach for 4-AP free sets."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 512

def _is_4ap_free(arr):
    """
    Check if a boolean array arr (index 0..N-1) is 4-AP free.
    Uses vectorized numpy operations for speed.
    Returns True if no 4-AP exists.
    """
    # arr[i] is True if (i+1) is in the set
    # We want to check if there exist a, d such that
    # arr[a-1] & arr[a+d-1] & arr[a+2d-1] & arr[a+3d-1] is True
    
    # Iterate over possible differences d
    # Max d is such that a + 3d <= N => 1 + 3d <= N => d <= (N-1)/3
    max_d = (N - 1) // 3
    
    for d in range(1, max_d + 1):
        # Shift the array to align terms
        # term 1: arr[0 : N-3d]
        # term 2: arr[d : N-2d]
        # term 3: arr[2d : N-d]
        # term 4: arr[3d : N]
        
        t1 = arr[:N-3*d]
        t2 = arr[d:N-2*d]
        t3 = arr[2*d:N-d]
        t4 = arr[3*d:N]
        
        # Check if any position has all four True
        # Using logical AND
        if np.any(t1 & t2 & t3 & t4):
            return False
            
    return True

def construct():
    deadline = time.time() + 55.0  # Safety margin
    
    # Initialize with a random set of medium density
    # Density around 0.4 is a good starting point for SA
    initial_density = 0.4
    current_arr = np.random.random(N) < initial_density
    current_size = np.sum(current_arr)
    
    # Ensure it's valid? If not, we just start with high energy (penalty).
    # But better to start with a valid set or handle penalty.
    # Let's use a penalty method: Energy = -size + lambda * (number of 4-APs)
    # But counting 4-APs is expensive.
    # Alternative: If invalid, reject move unless T is very high.
    
    # Let's try to find a valid starting point quickly with greedy
    # Greedy in random order
    order = list(range(N))
    random.shuffle(order)
    valid_arr = np.zeros(N, dtype=bool)
    for i in order:
        # Check if adding i creates 4-AP
        # We can reuse a fast check for single addition
        # Or just try and check full set (slow)
        # Let's use the single-add check logic from the seed but adapted to numpy?
        # No, let's just use the bitset logic from seed for initialization speed if needed,
        # or just start random and let SA fix it.
        
        # Actually, starting with a random set that might be invalid is fine for SA
        # if we define the objective as:
        # Score = size - 100 * (1 if invalid else 0)
        # But "invalid" is binary. We need a gradient.
        # Let's count the number of 4-APs.
        
        pass

    # Let's stick to a simpler heuristic:
    # 1. Greedy random init to get a valid set.
    # 2. SA to improve.
    
    # Fast greedy init using bitset (borrowed logic, optimized)
    bits = 0
    order = list(range(1, N + 1))
    random.shuffle(order)
    for x in order:
        # Check if x can be added
        ok = True
        # x is 4th
        for d in range(1, (x - 1) // 3 + 1):
            if (bits >> (x-d-1)) & 1 and (bits >> (x-2*d-1)) & 1 and (bits >> (x-3*d-1)) & 1:
                ok = False; break
        if not ok: continue
        # x is 3rd
        if x + 1 <= N:
            for d in range(1, (x - 1) // 2 + 1):
                if (bits >> (x+d-1)) & 1 and (bits >> (x-d-1)) & 1 and (bits >> (x-2*d-1)) & 1:
                    ok = False; break
        if not ok: continue
        # x is 2nd
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if (bits >> (x-d-1)) & 1 and (bits >> (x+d-1)) & 1 and (bits >> (x+2*d-1)) & 1:
                ok = False; break
        if not ok: continue
        # x is 1st
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (bits >> (x+d-1)) & 1 and (bits >> (x+2*d-1)) & 1 and (bits >> (x+3*d-1)) & 1:
                ok = False; break
        if not ok: continue
        
        bits |= (1 << (x-1))
        
    current_arr = np.zeros(N, dtype=bool)
    for i in range(N):
        if (bits >> i) & 1:
            current_arr[i] = True
            
    current_size = np.sum(current_arr)
    best_arr = current_arr.copy()
    best_size = current_size
    
    T = 10.0
    T_min = 0.1
    cooling = 0.9999
    
    while time.time() < deadline and T > T_min:
        # Random move: flip a bit
        idx = random.randint(0, N - 1)
        old_val = current_arr[idx]
        current_arr[idx] = not old_val
        
        new_size = current_size + (1 if not old_val else -1)
        
        # Check validity
        # Optimization: Only check if the move is likely to be valid?
        # No, just check.
        is_valid = _is_4ap_free(current_arr)
        
        if is_valid:
            # Accept if valid (always improves or maintains validity)
            # Actually, if it was valid and becomes valid, accept if size increased or same.
            # If it was invalid (shouldn't happen if we start valid and only accept valid), 
            # but let's assume we maintain validity invariant.
            if new_size >= current_size:
                current_size = new_size
                if new_size > best_size:
                    best_size = new_size
                    best_arr = current_arr.copy()
            else:
                # Valid but smaller? Unlikely to happen if we only flip one bit and maintain validity?
                # If we remove a bit, size decreases. We should accept with probability exp(-delta/T)
                # delta = -1 (size change). Energy = -size. Delta E = 1.
                # Accept with prob exp(-1/T)
                if random.random() < np.exp(-1 / T):
                    current_size = new_size
                else:
                    current_arr[idx] = old_val
        else:
            # Invalid. Reject.
            current_arr[idx] = old_val
            
        T *= cooling
        
    # Convert best_arr to list
    result = [i+1 for i in range(N) if best_arr[i]]
    return result

# EVOLVE-BLOCK-END

import numpy as np
import time
import random

def construct():
    n = 512
    start_time = time.time()
    max_time = 85.0  # Safety margin for 90s limit

    # 1. Pre-compute all 4-APs in [1, n]
    # Each AP is a tuple of 4 indices (a, a+d, a+2d, a+3d)
    all_aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Convert to numpy array for vectorized operations
    # Shape: (num_aps, 4)
    if all_aps:
        ap_array = np.array(all_aps, dtype=np.int32)
    else:
        ap_array = np.empty((0, 4), dtype=np.int32)
    
    # 2. Helper: Check if a bitset is 4-AP free
    # Returns True if no AP is fully contained in the set
    def is_valid(S):
        if len(ap_array) == 0:
            return True
        # Check all APs at once
        # S[ap_array] gives shape (num_aps, 4) of bools
        # If all 4 are True for any row, it's invalid
        return not np.all(S[ap_array], axis=1).any()
    
    # 3. Helper: Count number of "bad" APs (APs fully contained in S)
    # This is the objective to minimize (ideally 0)
    def count_bad_aps(S):
        if len(ap_array) == 0:
            return 0
        return int(np.sum(np.all(S[ap_array], axis=1)))

    # 4. Helper: Get all elements in the set
    def get_elements(S):
        return [i for i in range(1, n + 1) if S[i]]

    # 5. Initial Solution: Cantor Set (Base 3, no 2s)
    def cantor_set_bits():
        S = np.zeros(n + 1, dtype=bool)
        for i in range(1, n + 1):
            x = i
            valid = True
            while x > 0:
                if x % 3 == 2:
                    valid = False
                    break
                x //= 3
            if valid:
                S[i] = True
        return S

    S_cantor = cantor_set_bits()
    # Verify Cantor set is valid (it should be)
    if not is_valid(S_cantor):
        # Fallback to empty set if somehow invalid (shouldn't happen)
        S_cantor = np.zeros(n + 1, dtype=bool)

    # 6. Greedy Improvement on Initial Solution
    def greedy_improve(S):
        improved = True
        while improved:
            improved = False
            # Try to add any element
            for i in range(1, n + 1):
                if not S[i]:
                    S_new = S.copy()
                    S_new[i] = True
                    if is_valid(S_new):
                        S = S_new
                        improved = True
        return S

    S_current = greedy_improve(S_cantor)
    best_S = S_current.copy()
    best_count = int(np.sum(best_S))
    
    # 7. Simulated Annealing
    # Parameters
    T_start = 1.0
    T_end = 0.01
    alpha = 0.999 # Cooling rate
    
    # We need to estimate the "energy" difference.
    # Energy = -cardinality + lambda * (1 if invalid else 0)
    # But we want to stay valid. So we only accept valid moves, 
    # OR we use a penalty method.
    # Given the constraint is hard, let's stick to valid-only moves but allow 
    # "worsening" in terms of cardinality? No, that's just random walk.
    # 
    # Better approach for SA on constrained problems:
    # State space: All subsets.
    # Objective: Maximize Cardinality subject to Validity.
    # If we restrict to valid states, we can't use standard SA easily because 
    # "worsening" moves (removing elements) are the only way to escape local optima 
    # where we are stuck with a valid but small set.
    # 
    # Let's define Energy E(S) = -|S| if S is valid, else +infinity.
    # This is a discrete optimization.
    # Moves:
    # 1. Add a random element (if valid)
    # 2. Remove a random element (always valid)
    # 3. Swap: Remove one, Add one (if valid)
    #
    # To escape local optima where we can't add anything:
    # We must remove elements. Removing an element decreases |S|, so it's a "worsening" move.
    # We accept worsening moves with probability exp(-delta_E / T).
    # delta_E = E_new - E_old = (-|S_new|) - (-|S_old|) = |S_old| - |S_new|.
    # If we remove 1, delta_E = 1. P(accept) = exp(-1/T).
    
    T = T_start
    S = S_current.copy()
    current_count = int(np.sum(S))
    
    # Pre-compute indices for fast access
    all_indices = np.arange(1, n + 1)
    
    while time.time() - start_time < max_time:
        # Cooling
        T *= alpha
        
        # Generate a random move
        move_type = random.random()
        
        S_new = S.copy()
        
        if move_type < 0.4:
            # Try to add a random element
            i = random.choice(all_indices)
            if not S_new[i]:
                S_new[i] = True
                if not is_valid(S_new):
                    # Move invalid, reject immediately (or skip)
                    continue
                # Valid move
                delta_count = 1
                # Always accept if we improve (increase count)
                S = S_new
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_S = S.copy()
                    
        elif move_type < 0.8:
            # Try to remove a random element
            # Pick an element currently in S
            in_set = get_elements(S)
            if not in_set:
                continue
            i = random.choice(in_set)
            S_new[i] = False
            # Always valid to remove
            delta_count = -1
            # This is a worsening move. Accept with prob exp(-delta_E / T)
            # delta_E = -delta_count = 1
            if random.random() < np.exp(-1.0 / max(T, 1e-6)):
                S = S_new
                current_count -= 1
                # Update best? No, we are losing count.
                
        else:
            # Swap: Remove one, Add one
            in_set = get_elements(S)
            if not in_set:
                continue
            i_remove = random.choice(in_set)
            i_add = random.choice(all_indices)
            
            if i_add == i_remove:
                continue
            if S[i_add]:
                continue # Can't add what's already there
            
            S_new[i_remove] = False
            S_new[i_add] = True
            
            if is_valid(S_new):
                # Count is same. It's a neutral move.
                # Accept with some probability to explore?
                # Or just accept to randomize the landscape.
                # Let's accept with prob 0.5 to avoid cycling too much, 
                # or just always accept if valid to explore neighbors.
                S = S_new
                # current_count unchanged
                
    # 8. Final Greedy Polish
    best_S = greedy_improve(best_S)
    best_count = int(np.sum(best_S))
    
    # 9. Final check
    if not is_valid(best_S):
        # Should not happen, but fallback
        # Remove elements until valid
        for i in range(1, n + 1):
            if best_S[i] and not is_valid(best_S):
                best_S[i] = False
    
    return [i for i in range(1, n + 1) if best_S[i]]

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Size: {len(res)}, Time: {t1-t0:.2f}s")
    # Verify
    n = 512
    S = np.zeros(n+1, dtype=bool)
    for i in res:
        S[i] = True
    # Verify no 4-AP
    valid = True
    for d in range(1, (n-3)//3 + 1):
        for a in range(1, n-3*d+1):
            if S[a] and S[a+d] and S[a+2*d] and S[a+3*d]:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")

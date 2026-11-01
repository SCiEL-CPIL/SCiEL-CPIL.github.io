import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit
    
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute place values for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q
        
    # Precompute all words as indices
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q
            
    # Precompute ball_indices: for each word, list of indices within Hamming distance 1
    ball_indices = [[] for _ in range(total)]
    for i in range(total):
        w = words[i]
        ball_indices[i].append(i)
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i].append(new_idx)
                
    # Convert to numpy arrays for faster access in count updates
    ball_indices_np = [np.array(b, dtype=np.int32) for b in ball_indices]
    
    # Precompute membership matrix for fast swap validity checks
    # ball_membership[i, j] is True if j is in ball(i)
    # Size: 2187 x 2187 booleans ~ 4.8 MB. Acceptable.
    ball_membership = np.zeros((total, total), dtype=bool)
    for i in range(total):
        ball_membership[i, ball_indices_np[i]] = True
        
    # Greedy construction
    def greedy_build():
        covered = np.zeros(total, dtype=bool)
        code = []
        
        while not np.all(covered):
            best_c = -1
            max_new = -1
            # Vectorized greedy selection
            # new_cov[i] = sum(~covered[ball_indices_np[i]])
            # This is a bit expensive to do fully vectorized every step if not careful,
            # but total is small (2187).
            # Let's stick to loop for clarity and sufficient speed, or optimize if needed.
            # Actually, we can compute new_cov for all candidates at once?
            # new_cov = np.sum(~covered[ball_indices_np], axis=1) -- this requires a 3D array or similar.
            # Let's just loop. 2187 iterations * ~100 steps = 200k ops. Fast enough.
            
            for c in range(total):
                if c in code: # Optimization: skip already chosen? 
                    # Actually, we can't have duplicates in greedy.
                    # But checking 'c in code' is O(N). 
                    # Let's use a boolean array for chosen.
                    pass
            
            # Better Greedy:
            chosen_mask = np.zeros(total, dtype=bool)
            for c in code:
                chosen_mask[c] = True
                
            # Calculate new coverage for all unchosen candidates
            # We need sum of ~covered[ball] for each c not in code.
            # This is hard to vectorize directly without a huge temporary array.
            # Loop is fine.
            
            for c in range(total):
                if chosen_mask[c]:
                    continue
                ball = ball_indices_np[c]
                new_cov = np.sum(~covered[ball])
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break
                        
            if best_c == -1:
                break # Should not happen if covered is not all true
                
            code.append(best_c)
            chosen_mask[best_c] = True
            covered[ball_indices_np[best_c]] = True
        return code
    
    # Initial greedy
    current_code = set(greedy_build())
    best_code = list(current_code)
    best_size = len(best_code)
    
    # Initialize counts
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        cnt[ball_indices_np[c]] += 1
        
    def run_delete_phase(code_set, cnt_arr):
        """Removes redundant codewords. Returns True if any removed."""
        changed = False
        # Iterate over a copy to allow modification
        for c in list(code_set):
            ball = ball_indices_np[c]
            if np.min(cnt_arr[ball]) >= 2:
                code_set.remove(c)
                cnt_arr[ball] -= 1
                changed = True
        return changed
    
    def try_swap(c_remove, c_add, code_set, cnt_arr):
        """Try a swap. Returns True if successful."""
        if c_add in code_set:
            return False
        if c_remove == c_add:
            return False
            
        # Fast validity check using precomputed membership matrix
        # We need cnt[x] >= 2 for all x in ball(c_remove) \ ball(c_add)
        # mask_diff = ball_membership[c_remove] & ~ball_membership[c_add]
        # If any cnt[x] < 2 in mask_diff, invalid.
        
        # This creates a boolean array of size 2187.
        # We can avoid allocating a new array every time?
        # np.logical_and(ball_membership[c_remove], ~ball_membership[c_add])
        
        # Even better:
        # The points that lose coverage are in ball(c_remove) but NOT in ball(c_add).
        # Let's get the indices.
        # But checking min(cnt[indices]) is fast if indices are few.
        # However, finding the indices requires the mask.
        
        # Let's use the mask approach. It's vectorized and fast in C.
        mask_rem = ball_membership[c_remove]
        mask_add = ball_membership[c_add]
        
        # Points in rem but not in add
        # We only care about points where cnt < 2.
        # If any point in (rem \ add) has cnt < 2, swap is invalid.
        
        # Optimization: 
        # We can precompute for each c_remove, the list of points in its ball.
        # And for each c_add, a set or mask.
        # The mask approach is:
        # diff_mask = mask_rem & ~mask_add
        # if np.any(cnt_arr[diff_mask] < 2): return False
        
        # This allocates diff_mask. 
        # Alternative:
        # We know ball_size is 15.
        # We can iterate the 15 points in ball(c_remove) and check if they are in ball(c_add) and cnt >= 2.
        # 15 checks per swap.
        # vs 
        # Vectorized check on 2187 elements.
        # Vectorized is likely faster due to CPU cache and SIMD, despite the "wasted" checks on 0s.
        
        diff_mask = mask_rem & ~mask_add
        if np.any(cnt_arr[diff_mask] < 2):
            return False
            
        # Perform swap
        code_set.remove(c_remove)
        code_set.add(c_add)
        cnt_arr[ball_indices_np[c_remove]] -= 1
        cnt_arr[ball_indices_np[c_add]] += 1
        return True

    # Local Search Loop
    # We will run multiple restarts if time permits, or just a long single run.
    # Given the landscape, a single long run with random swaps is standard.
    
    while time.time() < deadline - 5.0:
        # 1. Delete Phase
        while run_delete_phase(current_code, cnt):
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                
        # 2. Swap Phase
        # Increase max_swaps significantly due to faster try_swap
        swaps_attempted = 0
        max_swaps = 2000 # Increased from 500
        while swaps_attempted < max_swaps and time.time() < deadline - 5.0:
            swaps_attempted += 1
            if len(current_code) == 0:
                break
                
            # Pick random codeword to remove
            c = random.choice(list(current_code))
            
            # Pick random candidate to add
            c_prime = random.randint(0, total - 1)
            
            # Try the swap
            if try_swap(c, c_prime, current_code, cnt):
                # After a valid swap, run delete phase
                while run_delete_phase(current_code, cnt):
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                        
                    # Early exit if we've hit a very good solution
                    if best_size <= 160:
                        break
                        
                if best_size <= 160:
                    break
                    
        if best_size <= 160:
            break

    # Final result conversion
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
        
    return result

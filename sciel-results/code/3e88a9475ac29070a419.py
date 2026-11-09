"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # seconds, safety margin
    
    def elapsed():
        return time.time() - start_time
        
    N_SMALL_END = 128
    N_LARGE_END = 256
    
    small_nums = list(range(2, N_SMALL_END + 1))
    large_nums = list(range(N_SMALL_END + 1, N_LARGE_END + 1))
    
    n_small = len(small_nums)
    n_large = len(large_nums)
    
    # Weights
    small_weights = [1.0 / x for x in small_nums]
    large_weights = [1.0 / x for x in large_nums]
    
    total_large_weight = sum(large_weights)
    
    # Precompute conflict masks
    # small_conflict_mask[i] : bitmask of small indices conflicting with i
    small_conflict_mask = [0] * n_small
    for i in range(n_small):
        for j in range(i + 1, n_small):
            a, b = small_nums[i], small_nums[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N_LARGE_END:
                small_conflict_mask[i] |= (1 << j)
                small_conflict_mask[j] |= (1 << i)
                
    # large_conflict_mask[i] : bitmask of large indices conflicting with small_nums[i]
    large_conflict_mask = [0] * n_small
    for i in range(n_small):
        a = small_nums[i]
        for j in range(n_large):
            b = large_nums[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N_LARGE_END:
                large_conflict_mask[i] |= (1 << j)
    
    def sum_large_weights_in_mask(mask):
        s = 0.0
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += large_weights[idx]
            m ^= lsb
        return s

    def get_score(small_mask, blocked_large_mask):
        # Sum of selected small weights
        s_small = 0.0
        m = small_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s_small += small_weights[idx]
            m ^= lsb
            
        # Sum of unblocked large weights
        # Total - Sum(blocked)
        s_blocked = sum_large_weights_in_mask(blocked_large_mask)
        s_large = total_large_weight - s_blocked
        
        return s_small + s_large

    def is_valid_small_mask(mask):
        # Check if mask is an independent set in small graph
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if (small_conflict_mask[idx] & mask) != 0:
                return False
            m ^= lsb
        return True

    def get_blocked_large_mask(small_mask):
        blocked = 0
        m = small_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            blocked |= large_conflict_mask[idx]
            m ^= lsb
        return blocked

    best_score = 0.0
    best_small_mask = 0
    best_blocked_large_mask = 0
    
    def update_best(sm, blm, score):
        nonlocal best_score, best_small_mask, best_blocked_large_mask
        if score > best_score:
            best_score = score
            best_small_mask = sm
            best_blocked_large_mask = blm

    # Local Search Function
    def local_search(initial_mask):
        # Validate initial mask
        if not is_valid_small_mask(initial_mask):
            # Try to fix by removing conflicting nodes? Or just start from scratch if invalid.
            # Greedy construction usually yields valid masks.
            pass
            
        current_mask = initial_mask
        current_blocked = get_blocked_large_mask(current_mask)
        current_score = get_score(current_mask, current_blocked)
        
        update_best(current_mask, current_blocked, current_score)
        
        improved = True
        iterations = 0
        max_iterations = 5000 # Safety cap per run
        
        while improved and elapsed() < DEADLINE and iterations < max_iterations:
            improved = False
            iterations += 1
            
            best_move_score = current_score
            best_move_type = None
            best_move_idx = None
            best_move_new_mask = current_mask
            best_move_new_blocked = current_blocked
            
            # Try Additions
            for i in range(n_small):
                if current_mask & (1 << i):
                    continue
                # Check small-small conflict
                if small_conflict_mask[i] & current_mask:
                    continue
                
                new_mask = current_mask | (1 << i)
                new_blocked = current_blocked | large_conflict_mask[i]
                
                # Calculate score delta
                # Delta = w_i - sum(large_weights in (large_conflict_mask[i] & ~current_blocked))
                # We can just call get_score for simplicity first, optimize if slow
                new_score = get_score(new_mask, new_blocked)
                
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move_type = 'add'
                    best_move_idx = i
                    best_move_new_mask = new_mask
                    best_move_new_blocked = new_blocked
            
            # Try Removals
            for i in range(n_small):
                if not (current_mask & (1 << i)):
                    continue
                
                new_mask = current_mask & ~(1 << i)
                new_blocked = get_blocked_large_mask(new_mask)
                new_score = get_score(new_mask, new_blocked)
                
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move_type = 'remove'
                    best_move_idx = i
                    best_move_new_mask = new_mask
                    best_move_new_blocked = new_blocked
            
            # Try Swaps (Remove i, Add j)
            # This is O(N^2) which might be slow. 
            # However, N=127. 127*127 = 16k checks.
            # Each check involves get_score which is O(N).
            # 16k * 128 = 2M ops per iteration.
            # Might be too slow for many iterations.
            # Let's restrict swaps to promising candidates or skip if too slow.
            # Given the "Add" and "Remove" moves cover the neighborhood, 
            # and we can do multiple Add/Remove steps, swaps might be redundant 
            # or can be simulated by two steps (Remove then Add).
            # However, a swap might be beneficial even if intermediate step is worse.
            # Let's try a limited set of swaps or just rely on Add/Remove.
            # Actually, "Remove then Add" is a valid path in local search.
            # So we don't strictly need direct swaps if we allow non-improving intermediate steps?
            # No, hill climbing only allows improving steps.
            # So we need direct swaps to escape local minima where removing hurts but swapping helps.
            
            # To keep it fast, we'll only try swaps for a subset of indices or limit it.
            # Let's try swaps for the first 20 most promising candidates?
            # Or just skip swaps and hope Add/Remove is enough?
            # Let's try swaps but with a time check.
            
            if elapsed() < DEADLINE - 1.0:
                for i in range(n_small):
                    if not (current_mask & (1 << i)):
                        continue
                    # Remove i
                    mask_without_i = current_mask & ~(1 << i)
                    
                    for j in range(n_small):
                        if mask_without_i & (1 << j):
                            continue
                        # Check if j conflicts with mask_without_i
                        if small_conflict_mask[j] & mask_without_i:
                            continue
                        
                        new_mask = mask_without_i | (1 << j)
                        new_blocked = get_blocked_large_mask(new_mask)
                        new_score = get_score(new_mask, new_blocked)
                        
                        if new_score > best_move_score:
                            best_move_score = new_score
                            best_move_type = 'swap'
                            best_move_idx = (i, j)
                            best_move_new_mask = new_mask
                            best_move_new_blocked = new_blocked

            if best_move_type is not None:
                current_mask = best_move_new_mask
                current_blocked = best_move_new_blocked
                current_score = best_move_score
                improved = True
                update_best(current_mask, current_blocked, current_score)
            else:
                improved = False

        return current_score, current_mask, current_blocked

    # Generate initial candidates
    initial_masks = []
    
    # 1. Empty set
    initial_masks.append(0)
    
    # 2. Greedy by weight (smallest numbers)
    mask = 0
    for i in range(n_small):
        if not (small_conflict_mask[i] & mask):
            mask |= (1 << i)
    initial_masks.append(mask)
    
    # 3. Greedy by weight / conflict count
    order = sorted(range(n_small), key=lambda i: -small_weights[i] / (1 + bin(small_conflict_mask[i]).count('1')))
    mask = 0
    for i in order:
        if not (small_conflict_mask[i] & mask):
            mask |= (1 << i)
    initial_masks.append(mask)
    
    # 4. Randomized Greedy (multiple times)
    for _ in range(20):
        if elapsed() > DEADLINE - 5: break
        order = list(range(n_small))
        random.shuffle(order)
        mask = 0
        for i in order:
            if not (small_conflict_mask[i] & mask):
                mask |= (1 << i)
        initial_masks.append(mask)
        
    # 5. Selecting numbers with large prime factors?
    # Heuristic: numbers that are prime or 2*prime might be good.
    
    # Run local search on each
    for m in initial_masks:
        if elapsed() > DEADLINE: break
        local_search(m)
        
    # Final Polish: Run local search again on the best found mask with more iterations?
    # Or just rely on the previous runs.
    
    # Construct final result
    result = []
    # Add selected small numbers
    m = best_small_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(small_nums[idx])
        m ^= lsb
        
    # Add unblocked large numbers
    blocked = best_blocked_large_mask
    for i in range(n_large):
        if not (blocked & (1 << i)):
            result.append(large_nums[i])
            
    result.sort()
    return result
# EVOLVE-BLOCK-END

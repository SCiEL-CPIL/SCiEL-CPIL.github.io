"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _check_validity(nums):
    """
    Checks if the set of subset sums of nums satisfies:
    1. No duplicate sums (all subset sums unique).
    2. No sum divides another distinct sum.
    """
    if not nums:
        return True
    
    # 1. Generate all subset sums and check for duplicates
    # sum_counts[s] = number of subsets that sum to s
    sum_counts = {0: 1}
    for n in nums:
        new_counts = {}
        for s, c in sum_counts.items():
            ns = s + n
            if ns in new_counts:
                new_counts[ns] += c
            else:
                new_counts[ns] = c
        for ns, c in new_counts.items():
            if ns in sum_counts:
                sum_counts[ns] += c
            else:
                sum_counts[ns] = c
        
        # Early exit if duplicates found
        for s, c in sum_counts.items():
            if s > 0 and c > 1:
                return False

    # 2. Check divisibility
    unique_sums = [s for s, c in sum_counts.items() if s > 0]
    if not unique_sums:
        return True

    sorted_sums = sorted(unique_sums)
    sum_set = set(sorted_sums)
    max_sum = sorted_sums[-1]

    for s in sorted_sums:
        multiple = 2 * s
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += s
            
    return True

def construct():
    start_time = time.time()
    deadline = 55.0  # Leave 5s margin for 60s limit
    
    # Candidate pool
    candidates = list(range(1, 257))
    
    best_set = []
    best_size = 0
    
    # Multiple restarts to explore different regions of the solution space
    while time.time() < deadline:
        current_set = []
        
        # Greedy Phase: Try to build a set by adding elements
        # Shuffle candidates to randomize order
        shuffled_candidates = candidates.copy()
        random.shuffle(shuffled_candidates)
        
        # Heuristic: Try larger numbers first? Or random?
        # Let's try a mix. 
        for c in shuffled_candidates:
            if time.time() > deadline:
                break
            new_set = current_set + [c]
            if _check_validity(new_set):
                current_set = new_set
        
        # Local Search Phase: Try to improve size via swaps or additions
        # We track the best set found in this restart
        restart_best = current_set
        restart_best_size = len(current_set)
        
        local_deadline = min(deadline, time.time() + 5.0)
        
        while time.time() < local_deadline:
            # Try to add a new element
            added = False
            for _ in range(20): # Try a few random additions
                if len(current_set) >= 30:
                    break
                c = random.choice(candidates)
                if c in current_set:
                    continue
                new_set = current_set + [c]
                if _check_validity(new_set):
                    current_set = new_set
                    added = True
                    if len(current_set) > restart_best_size:
                        restart_best = current_set
                        restart_best_size = len(current_set)
                    break
            
            if not added:
                # Try a swap
                if len(current_set) > 0:
                    idx = random.randint(0, len(current_set) - 1)
                    old_val = current_set[idx]
                    new_val = random.choice(candidates)
                    if new_val in current_set:
                        continue
                    
                    new_set = current_set.copy()
                    new_set[idx] = new_val
                    if _check_validity(new_set):
                        # Accept swap if it doesn't reduce size (it doesn't)
                        # We accept to explore
                        current_set = new_set
                        # Note: We don't update restart_best here unless we add later
                        # Swaps maintain size, so they only help if they enable future adds

        if len(restart_best) > best_size:
            best_set = restart_best
            best_size = len(restart_best)
            
        # If we found a very large set, we might stop early? 
        # Max possible is likely small, so keep going until deadline.
        
    return best_set
# EVOLVE-BLOCK-END

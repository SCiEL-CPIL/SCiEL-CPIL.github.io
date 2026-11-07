"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_valid_sums(sums):
    """Check if the set of sums contains no pair where one divides the other."""
    if not sums:
        return True
    sorted_sums = sorted(sums)
    max_s = sorted_sums[-1]
    for si in sorted_sums:
        if si * 2 > max_s:
            break
        for m in range(2, max_s // si + 1):
            if si * m in sums:
                return False
    return True

def _can_add_incremental(current_sums, cand):
    """
    Check if adding cand to the current set (represented by current_sums) keeps it valid.
    Returns (is_valid, new_sums)
    """
    new_sums = set(current_sums)
    for s in current_sums:
        new_sums.add(s + cand)
    new_sums.add(cand)
    
    if _is_valid_sums(new_sums):
        return True, new_sums
    else:
        return False, None

def _greedy_construct_random(candidates, deadline):
    """Greedy construction with random candidate order."""
    current = []
    current_sums = set()
    cand_list = list(candidates)
    random.shuffle(cand_list)
    
    for cand in cand_list:
        if time.time() > deadline:
            break
        is_valid, new_sums = _can_add_incremental(current_sums, cand)
        if is_valid:
            current.append(cand)
            current_sums = new_sums
            
    return current

def construct():
    start = time.time()
    # Safety margin: return before 55s to allow for overhead
    deadline = start + 50
    
    best = []
    best_sums = set()
    
    # Candidate pool: larger numbers are generally better for avoiding small divisors
    # We focus on the upper range to maximize size while keeping sums large
    candidates = list(range(256, 100, -1))
    
    # Run multiple random restarts
    # Each restart tries to build a set from scratch
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Vary the candidate pool slightly or just shuffle
        # Using a subset of large numbers helps
        pool = candidates[:]
        
        result = _greedy_construct_random(pool, deadline)
        
        if len(result) > len(best):
            best = result
            # Re-compute sums for best if needed, but we can just re-run greedy logic
            # or store sums. For now, we just keep the list.
            # To be efficient, we could return sums from greedy, but let's keep it simple.
            # We will re-validate best at the end if necessary, but greedy ensures validity.
            
            # Optional: Try to extend the best found so far with remaining time
            # This is a local search step
            if time.time() < deadline - 5:
                # Try to add more to the current best
                current_sums = set()
                for x in best:
                    # Rebuild sums for the current best
                    # This is O(N * S) which is fine for N~15
                    pass 
                
                # Efficiently rebuild sums for best
                temp_sums = set()
                for x in best:
                    new_temp = set(temp_sums)
                    for s in temp_sums:
                        new_temp.add(s + x)
                    new_temp.add(x)
                    temp_sums = new_temp
                
                # Try to add more candidates not in best
                for cand in candidates:
                    if time.time() > deadline - 2:
                        break
                    if cand in best:
                        continue
                    is_valid, new_sums = _can_add_incremental(temp_sums, cand)
                    if is_valid:
                        best.append(cand)
                        temp_sums = new_sums
                        
                # Update best if improved
                # (It is already updated in the loop above)

    # Ensure return type is list of ints
    return list(best)
# EVOLVE-BLOCK-END

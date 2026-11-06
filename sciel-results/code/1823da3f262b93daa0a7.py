"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # Greedy: build set one element at a time, picking smallest valid value
    # Maintain all current subset sums
    subset_sums = {0}
    result = []
    
    for i in range(12):
        # Try values starting from 1
        candidate = 1
        while True:
            # Check if adding candidate creates duplicate subset sums
            new_sums = set()
            valid = True
            for s in subset_sums:
                ns = s + candidate
                if ns in subset_sums or ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            if valid:
                subset_sums.update(new_sums)
                result.append(candidate)
                break
            candidate += 1
            # Time check
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # If we didn't get 12 elements, fall back to powers of 2
    if len(result) < 12:
        return [1 << i for i in range(12)]
    
    return result
# EVOLVE-BLOCK-END

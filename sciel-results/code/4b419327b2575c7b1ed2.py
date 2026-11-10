import time
import itertools

def construct():
    start_time = time.time()
    TIME_LIMIT = 90.0
    SAFETY_MARGIN = 5.0

    # Precompute modular inverses for denominators 1..10
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)

    # seen[r] will store the list of denominators (1-based) for residue r
    # Initialize with None
    seen = [None] * 101
    
    # We want to minimize the largest row length.
    # We iterate by subset size (length) from 1 to 10.
    # The first time we encounter a residue, it is with the minimum possible length.
    
    found_count = 0
    max_len_needed = 0
    
    for size in range(1, 11):
        # Generate all subsets of this size
        # Denominators are 1..10, indices 0..9
        for combo in itertools.combinations(range(10), size):
            # Calculate sum of inverses
            s = 0
            for idx in combo:
                s = (s + inv[idx + 1]) % 101
            
            if seen[s] is None:
                # Convert 0-based indices to 1-based denominators
                seen[s] = [idx + 1 for idx in combo]
                found_count += 1
        
        # Check if we have covered all residues
        if found_count == 101:
            max_len_needed = size
            break
            
        # Time check
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            # If we run out of time (should not happen), return best effort
            # Fill remaining with any valid subset (e.g., size 10)
            # But this branch is unlikely to be hit.
            break

    # If not all found (should not happen), fill with full set or something
    if found_count < 101:
        # Fallback: use all denominators for remaining
        full_set = list(range(1, 11))
        for r in range(101):
            if seen[r] is None:
                seen[r] = full_set

    return seen

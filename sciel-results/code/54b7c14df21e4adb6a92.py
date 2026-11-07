import time
import random

def _check_divisibility(sums_set):
    if len(sums_set) < 2:
        return True
    max_val = max(sums_set)
    for x in sums_set:
        if x * 2 > max_val:
            continue
        multiple = x * 2
        while multiple <= max_val:
            if multiple in sums_set:
                return False
            multiple += x
    return True

def _try_add(current_sums, x):
    # Check distinctness
    new_sums = set(current_sums)
    new_sums.add(x)
    for s in current_sums:
        new_sums.add(s + x)
    
    # If collision, len will be less than 2*len+1
    if len(new_sums) != 2 * len(current_sums) + 1:
        return None, None
        
    # Check divisibility
    if _check_divisibility(new_sums):
        return new_sums, x
    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    best_size = 0
    
    # Candidate pool
    pool = list(range(1, 257))
    
    # Strategy: Multiple randomized greedy passes
    # Bias towards larger numbers
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Create a biased permutation
        # We want to try large numbers first.
        # Method: Sort descending, then shuffle with a bias?
        # Or: Assign random weights, higher for larger numbers.
        
        # Simple approach:
        # 1. Descending order
        # 2. Random shuffle
        # 3. Mixed: Top 50% descending, bottom 50% random?
        
        if restart % 3 == 0:
            candidates = sorted(pool, reverse=True)
        elif restart % 3 == 1:
            candidates = pool[:]
            random.shuffle(candidates)
        else:
            # Bias: Larger numbers have higher chance to be earlier
            # Assign weight = x
            # Sample without replacement?
            # Simpler: Sort by random() / x
            candidates = pool[:]
            random.shuffle(candidates)
            # Sort by descending "value" but with noise
            # Let's just use a weighted shuffle simulation
            # Or simply: Take top half sorted desc, bottom half random, interleave?
            # Let's stick to simple variations for speed.
            candidates = sorted(pool, key=lambda x: random.random() / x, reverse=True)

        current_set = []
        current_sums = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
                
            new_sums, _ = _try_add(current_sums, x)
            if new_sums is not None:
                current_set.append(x)
                current_sums = new_sums
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                    
        # Optional: Local search on best_set?
        # If we have time, try to swap elements in best_set to allow adding more.
        # This is complex. Let's rely on more restarts.
        
    return best_set

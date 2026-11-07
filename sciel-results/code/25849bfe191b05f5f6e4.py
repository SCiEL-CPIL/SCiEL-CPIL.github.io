import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin (60s limit)
    
    def get_subset_sums(subset):
        """Get all non-empty subset sums efficiently."""
        n = len(subset)
        sums = [0]
        for x in subset:
            sums = sums + [s + x for s in sums]
        return sums[1:]  # exclude empty sum
    
    def is_valid(subset):
        """Check if subset has no two distinct subset sums where one divides the other."""
        n = len(subset)
        if n == 0:
            return True
        if n == 1:
            return True
            
        sums = get_subset_sums(subset)
        
        # Check for duplicates
        if len(set(sums)) != len(sums):
            return False
            
        sums_set = set(sums)
        max_sum = max(sums)
        
        # Check divisibility: for each sum S, no multiple k*S (k>=2) should be in sums_set
        for S in sums:
            if S <= 0:
                continue
            multiple = 2 * S
            while multiple <= max_sum:
                if multiple in sums_set:
                    return False
                multiple += S
                
        return True
    
    # Start with a known valid seed
    current_best = [2, 3]
    
    # Candidate pool: try to add elements
    # Use numbers that are likely to work - medium to large values
    candidates = list(range(1, 257))
    random.shuffle(candidates)
    
    # Greedy growth
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for c in candidates:
            if c in current_best:
                continue
            if time.time() > deadline:
                break
            test_set = current_best + [c]
            if is_valid(test_set):
                current_best = test_set
                improved = True
                break
    
    # Local search: try swapping elements
    best_size = len(current_best)
    search_set = list(current_best)
    
    while time.time() < deadline:
        # Try random swaps
        if len(search_set) < 2:
            break
        i = random.randint(0, len(search_set) - 1)
        c = random.randint(1, 256)
        if c in search_set:
            continue
        old = search_set[i]
        search_set[i] = c
        if is_valid(search_set):
            if len(set(search_set)) == len(search_set):  # ensure distinct
                pass
            else:
                search_set[i] = old
                continue
            # Check if it's still distinct
            if len(set(search_set)) != len(search_set):
                search_set[i] = old
                continue
            current_best = list(search_set)
            best_size = len(current_best)
        else:
            search_set[i] = old
    
    # Try to grow again after local search
    candidates2 = list(range(1, 257))
    random.shuffle(candidates2)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for c in candidates2:
            if c in current_best:
                continue
            if time.time() > deadline:
                break
            test_set = current_best + [c]
            if len(test_set) != len(set(test_set)):
                continue
            if is_valid(test_set):
                current_best = test_set
                improved = True
                break
    
    # Ensure return type is list of ints
    return [int(x) for x in current_best]

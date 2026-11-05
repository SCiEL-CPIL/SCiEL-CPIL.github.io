import time
import random

def _check_distinct_subset_sums(arr):
    """Check if all subset sums are distinct. Returns True if valid."""
    n = len(arr)
    sums = set()
    sums.add(0)
    for x in arr:
        current_sums = set()
        for s in sums:
            new_s = s + x
            if new_s in sums or new_s in current_sums:
                return False
            current_sums.add(new_s)
        sums.update(current_sums)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin
    
    n = 12
    # Start with binary solution
    arr = [1 << i for i in range(n)]
    best = list(arr)
    best_max = max(best)
    
    # Greedy Reduction Pass
    # Sort the array to process largest to smallest
    current = sorted(arr)
    
    for i in range(n - 1, -1, -1):
        if time.time() > deadline:
            break
            
        current_val = current[i]
        min_val = 1 if i == 0 else current[i-1] + 1
        
        # Try to reduce current[i] to the smallest possible valid value >= min_val
        # We scan from min_val upwards
        found = False
        # To avoid scanning a huge range if min_val is small and current_val is large,
        # we can limit the scan or use random sampling if range is large.
        # However, for the largest element, min_val is likely not too small.
        
        # If range is large, sample
        if current_val - min_val > 100:
            # Try random candidates first
            for _ in range(50):
                v = random.randint(min_val, current_val - 1)
                if v in current:
                    continue
                new_arr = current[:]
                new_arr[i] = v
                if _check_distinct_subset_sums(new_arr):
                    current = new_arr
                    found = True
                    break
            if found:
                continue
        
        # Linear scan from min_val
        for v in range(min_val, current_val):
            if v in current:
                continue
            new_arr = current[:]
            new_arr[i] = v
            if _check_distinct_subset_sums(new_arr):
                current = new_arr
                found = True
                break
        
    if max(current) < best_max:
        best = current
        best_max = max(current)
        
    # Local Search to fine-tune
    random.seed(42)
    current = best[:]
    
    for _ in range(200000):
        if time.time() > deadline:
            break
            
        i = random.randint(0, n - 1)
        old_val = current[i]
        
        # Try to reduce
        if old_val > 1:
            reduction = random.randint(1, min(old_val - 1, 50))
            new_val = old_val - reduction
            
            if new_val not in current:
                new_arr = current[:]
                new_arr[i] = new_val
                if _check_distinct_subset_sums(new_arr):
                    current = new_arr
                    if max(current) < best_max:
                        best = current[:]
                        best_max = max(current)
        
        # Try to increase (to enable future reductions)
        if old_val < best_max:
            increase = random.randint(1, 50)
            new_val = old_val + increase
            if new_val <= best_max and new_val not in current:
                new_arr = current[:]
                new_arr[i] = new_val
                if _check_distinct_subset_sums(new_arr):
                    current = new_arr
                    
    return best

import time
import random

def _all_subset_sums_distinct(nums):
    """Check if all subset sums of nums are distinct."""
    n = len(nums)
    sums = {0}
    for num in nums:
        new_sums = set()
        for s in sums:
            new_s = s + num
            if new_s in sums or new_s in new_sums:
                return False
            new_sums.add(new_s)
        sums = sums | new_sums
    return len(sums) == (1 << n)

def construct():
    deadline = time.time() + 80  # 10s safety margin
    best = [1 << i for i in range(12)]
    best_max = max(best)
    
    def check_valid(nums):
        if len(nums) != 12:
            return False
        if any(x <= 0 or x > 1048576 for x in nums):
            return False
        if len(set(nums)) != 12:
            return False
        return _all_subset_sums_distinct(nums)
    
    def try_decrease_max(current):
        current = sorted(current)
        idx = 11
        orig = current[idx]
        # Try decreasing the max element
        # First try fine-grained search near current max
        for new_val in range(orig - 1, max(0, orig - 300), -1):
            if new_val in current[:idx]:
                continue
            candidate = current[:]
            candidate[idx] = new_val
            if _all_subset_sums_distinct(candidate):
                return candidate
        # Then coarser search
        step = max(1, orig // 100)
        for new_val in range(orig - 300, 0, -step):
            if new_val in current[:idx]:
                continue
            candidate = current[:]
            candidate[idx] = new_val
            if _all_subset_sums_distinct(candidate):
                return candidate
        return None
    
    # Phase 1: Greedy from powers of two
    current = [1 << i for i in range(12)]
    for _ in range(100000):
        if time.time() > deadline - 15:
            break
        result = try_decrease_max(current)
        if result is None:
            break
        current = result
    
    if max(current) < best_max:
        best = current
        best_max = max(current)
    
    # Phase 2: Randomized local search
    for r in range(100):
        if time.time() > deadline - 5:
            break
        
        # Perturb the best solution
        start = best[:]
        num_perturb = random.randint(2, 8)
        for _ in range(num_perturb):
            idx = random.randint(0, 11)
            delta = random.randint(-max(1, start[idx]//5), max(1, start[idx]//5))
            new_val = start[idx] + delta
            if 0 < new_val <= 1048576:
                start[idx] = new_val
        
        if not check_valid(start):
            continue
        
        # Local search from this start
        local = sorted(start)
        for _ in range(50000):
            if time.time() > deadline - 3:
                break
            result = try_decrease_max(local)
            if result is None:
                # Try random element decrease
                candidate = local[:]
                idx = random.randint(0, 11)
                if candidate[idx] > 1:
                    new_val = max(1, candidate[idx] - random.randint(1, max(1, candidate[idx]//8)))
                    candidate[idx] = new_val
                    if new_val not in candidate[:idx] + candidate[idx+1:]:
                        if _all_subset_sums_distinct(candidate):
                            local = sorted(candidate)
                            continue
                break
            local = result
        
        if max(local) < best_max:
            best = local
            best_max = max(local)
            if best_max <= 200:
                break
    
    return best

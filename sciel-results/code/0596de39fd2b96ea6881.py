import time
from itertools import combinations

def _subset_sums_valid(sums_set):
    """Check if no two distinct values in sums_set divide one another."""
    sorted_sums = sorted(sums_set)
    n = len(sorted_sums)
    for i in range(n):
        si = sorted_sums[i]
        for j in range(i+1, n):
            sj = sorted_sums[j]
            if sj % si == 0:
                return False
    return True

def _get_all_subset_sums(nums):
    """Get all nonempty subset sums as a set."""
    n = len(nums)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        sums.add(s)
    return sums

def _is_valid_set(nums):
    """Check if the set is valid: all subset sums are divisibility-incomparable."""
    if len(nums) != len(set(nums)):
        return False
    for x in nums:
        if not (1 <= x <= 256):
            return False
    sums = _get_all_subset_sums(nums)
    return _subset_sums_valid(sums)

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    best_set = [31, 32, 34]
    best_size = len(best_set)
    
    # Try to build larger sets using greedy + local search
    # Start with large numbers to make subset sums large
    candidates = list(range(256, 0, -1))
    
    def try_extend(current_set, candidate_pool, max_size):
        """Try to extend current_set with elements from candidate_pool."""
        if time.time() > deadline:
            return current_set, candidate_pool
        
        if len(current_set) >= max_size:
            return current_set, candidate_pool
        
        current_sums = _get_all_subset_sums(current_set)
        
        for i, c in enumerate(candidate_pool):
            if time.time() > deadline:
                return current_set, candidate_pool
            
            new_set = current_set + [c]
            if len(new_set) != len(set(new_set)):
                continue
            
            # Efficiently check: new subset sums are old sums + c, plus c itself
            # Actually just compute all subset sums of new_set
            new_sums = _get_all_subset_sums(new_set)
            if _subset_sums_valid(new_sums):
                return new_set, candidate_pool[:i] + candidate_pool[i+1:]
        
        return current_set, candidate_pool
    
    # Greedy approach: start with large numbers
    for start_val in range(256, 100, -1):
        if time.time() > deadline:
            break
        
        current = [start_val]
        pool = [x for x in range(255, 0, -1) if x != start_val]
        
        for _ in range(30):
            if time.time() > deadline:
                break
            current, pool = try_extend(current, pool, 30)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Try different starting points and orders
    for start_val in range(200, 50, -5):
        if time.time() > deadline:
            break
        
        current = [start_val]
        # Try adding in different orders
        for next_val in range(256, start_val, -1):
            if time.time() > deadline:
                break
            if next_val == start_val:
                continue
            new_set = [start_val, next_val]
            if _is_valid_set(new_set):
                current = new_set
                break
        
        pool = [x for x in range(256, 0, -1) if x not in current]
        for _ in range(30 - len(current)):
            if time.time() > deadline:
                break
            current, pool = try_extend(current, pool, 30)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Local search: try to improve best_set
    if best_size < 30 and time.time() < deadline - 10:
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(len(best_set)):
                if time.time() > deadline - 5:
                    break
                for new_val in range(256, 0, -1):
                    if time.time() > deadline - 5:
                        break
                    if new_val in best_set:
                        continue
                    new_set = best_set[:i] + [new_val] + best_set[i+1:]
                    if _is_valid_set(new_set):
                        best_set = new_set
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try adding a new element
                for new_val in range(256, 0, -1):
                    if time.time() > deadline - 5:
                        break
                    if new_val in best_set:
                        continue
                    new_set = best_set + [new_val]
                    if _is_valid_set(new_set):
                        best_set = new_set
                        best_size = len(best_set)
                        break
    
    return list(best_set)

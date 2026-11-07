import random
import time

def _verify_subset_products(nums):
    """Verify all 2^k subset products are distinct."""
    k = len(nums)
    if k > 20:
        return False # Safety
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= nums[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def construct():
    """Return a list of up to 18 distinct integers in 2..40 with distinct subset products."""
    
    # Known optimal candidate (size 16)
    best_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    if not _verify_subset_products(best_set):
        # Fallback
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    best_size = len(best_set)
    best_result = best_set
    
    # Pool of all valid numbers
    all_nums = list(range(2, 41))
    
    # 1. Try to extend the known best set by 1
    current = set(best_result)
    for num in all_nums:
        if num not in current:
            candidate = list(current) + [num]
            if _verify_subset_products(candidate):
                return candidate # Found 17
    
    # 2. Local Search / Hill Climbing
    # Try to find a set of size 17 or 18
    
    start_time = time.time()
    timeout = 50.0 # Safety margin
    
    def random_set(size):
        return random.sample(all_nums, size)
    
    # Strategy: Start with random sets of size 15, try to grow
    for _ in range(1000):
        if time.time() - start_time > timeout:
            break
            
        # Random restart
        current = set(random_set(15))
        
        # Greedy growth
        changed = True
        while changed and time.time() - start_time < timeout:
            changed = False
            # Shuffle pool to avoid bias
            pool = [n for n in all_nums if n not in current]
            random.shuffle(pool)
            
            for num in pool:
                candidate = list(current) + [num]
                if _verify_subset_products(candidate):
                    current.add(num)
                    changed = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best_result = list(current)
                    break # Restart pool check as set changed
        
        # If we found 17 or 18, we can stop
        if best_size >= 17:
            break

    # 3. Try swaps on the best found set to see if we can improve
    # (Optional, if we didn't find 17 yet)
    if best_size < 17:
        current = set(best_result)
        for _ in range(5000):
            if time.time() - start_time > timeout:
                break
            # Pick random element to remove
            to_remove = random.choice(list(current))
            current.remove(to_remove)
            
            # Try to add a new element
            pool = [n for n in all_nums if n not in current]
            random.shuffle(pool)
            
            added = False
            for num in pool:
                candidate = list(current) + [num]
                if _verify_subset_products(candidate):
                    current.add(num)
                    added = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best_result = list(current)
                    break
            
            if not added:
                current.add(to_remove) # Restore

    # Ensure types
    result = []
    seen = set()
    for x in best_result:
        if isinstance(x, int) and not isinstance(x, bool) and 2 <= x <= 40 and x not in seen:
            seen.add(x)
            result.append(x)
            
    return result

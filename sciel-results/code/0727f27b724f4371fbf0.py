import time
import random
import sys

# EVOLVE-BLOCK-START

DEADLINE_SECONDS = 85.0
START_TIME = time.time()

def _is_valid(elements):
    """
    Check if the set of elements satisfies the condition:
    1. All non-empty subset sums are distinct.
    2. No two distinct subset sums divide each other.
    """
    n = len(elements)
    if n == 0:
        return True
    
    # Generate all subset sums
    # Using a set of sums
    sums = {0}
    for x in elements:
        # Add x to all existing sums
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        sums = sums.union(new_sums)
    
    # Remove 0
    sums.discard(0)
    
    # Check for distinct sums
    # If len(sums) != 2**n - 1, there are duplicates
    # But we can just check if any sum divides another
    # We need to check pairwise divisibility
    # To speed up, convert to list
    sum_list = list(sums)
    max_sum = max(sum_list)
    
    # Check divisibility
    # For each sum s, check if any other sum is a multiple of s
    # We can iterate s and check multiples
    for s in sum_list:
        # Check multiples 2s, 3s, ... up to max_sum
        # If s is small, there are many multiples.
        # If s is large, few.
        for m in range(2, max_sum // s + 1):
            if s * m in sums:
                return False
    return True

def _greedy_construct(target_size, pool):
    """
    Try to construct a valid set of target_size from pool.
    Uses random restarts internally.
    """
    best_set = []
    for _ in range(100): # 100 restarts
        if time.time() - START_TIME > DEADLINE_SECONDS - 5:
            break
        current_set = []
        available = list(pool)
        random.shuffle(available)
        
        # Try to add elements one by one
        for _ in range(target_size):
            added = False
            # Try to find an element that can be added
            # Shuffle available again for variety
            random.shuffle(available)
            for candidate in available:
                new_set = current_set + [candidate]
                if _is_valid(new_set):
                    current_set = new_set
                    available.remove(candidate)
                    added = True
                    break
            if not added:
                break
        
        if len(current_set) > len(best_set):
            best_set = current_set
            if len(best_set) == target_size:
                break
                
    return best_set

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # We want to maximize size.
    # Max possible is 30, but 2^30 is too big for fast check.
    # Let's try to find a set of size 20 first.
    # If found, try 21, 22...
    
    # Pool of candidates: 1 to 256
    # Avoid 1, as 1 divides everything.
    # So pool is 2..256
    pool = list(range(2, 257))
    
    best_size = 0
    best_set = []
    
    # Try sizes from high to low? Or low to high to build up?
    # Greedy from empty is better.
    # Let's try to get a set of size 20.
    
    # Strategy: 
    # 1. Try to build a set of size 20.
    # 2. If successful, try to extend to 21, 22...
    # 3. If not, try to build a set of size 19, etc.
    
    # Actually, just run _greedy_construct with target_size = 30.
    # It will stop early if it can't add more.
    # But _is_valid is slow for large sets.
    # So we should limit the check to sets of size <= 20 or 25.
    
    # Let's modify _is_valid to be faster or just use it for small sets.
    # For n=20, 2^20 = 1M sums. Check is fast.
    # For n=25, 2^25 = 33M sums. Check might take 5-10s.
    # For n=30, 2^30 = 1B sums. Too slow.
    
    # So we can only reliably check sets up to size 25.
    # Let's aim for size 20-25.
    
    # We will try to construct a set of size 25.
    # If we fail, we return the best we found.
    
    # To improve success, we can use a smarter pool.
    # Larger numbers might be better?
    # Let's just use random.
    
    # Run multiple attempts for different target sizes
    # Start with 20, then 25
    
    # Attempt 1: Target 20
    set_20 = _greedy_construct(20, pool)
    if len(set_20) > len(best_set):
        best_set = set_20
    
    # Attempt 2: Target 25 (might be slower)
    # Only if we have time
    if time.time() - START_TIME < DEADLINE_SECONDS - 20:
        set_25 = _greedy_construct(25, pool)
        if len(set_25) > len(best_set):
            best_set = set_25
            
    # Attempt 3: Try to extend best_set if it's small?
    # If best_set is size 15, try to add more.
    
    # Ensure we return a list of ints
    if not best_set:
        # Fallback to a small valid set
        # {2, 3} -> sums 2, 3, 5. 2|? no. 3|? no. 5|? no.
        # {2, 3} is valid.
        return [2, 3]
        
    return best_set

# EVOLVE-BLOCK-END

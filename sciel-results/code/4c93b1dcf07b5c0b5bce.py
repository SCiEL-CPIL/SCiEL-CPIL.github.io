import time
import random

def _is_valid_exact(s):
    """
    Check if the set s has cardinality-separated subset sums using bitsets.
    """
    n = len(s)
    if n < 2:
        return True
    
    # sums[k] is an integer where bit i is 1 if sum i is achievable by a subset of size k
    # Max sum is n * 256 = 15360. We can use Python ints as bitsets.
    
    # Initialize
    # sums[0] has bit 0 set (empty sum)
    sums = [0] * (n + 1)
    sums[0] = 1 # bit 0 is 1
    
    for x in s:
        # Update from high k to low k to avoid using x multiple times in same step
        # Actually, we iterate over existing subsets and add x to them.
        # Let's do it carefully.
        # new_sums[k] = sums[k] | (sums[k-1] << x)
        
        # We need to update sums in place or use a copy.
        # Since we are adding x to subsets of size k-1 to get size k,
        # and we iterate k from n down to 1, we can update in place if we are careful?
        # No, standard knapsack update:
        # for k in range(n, 0, -1):
        #     sums[k] |= sums[k-1] << x
        
        # However, we must ensure we don't use x more than once.
        # The standard 0/1 knapsack bitset update handles this by iterating downwards.
        
        # But wait, `sums` is a list of ints.
        # We need to mask the bits? No, Python ints are arbitrary precision.
        # But we should limit the bit length to avoid explosion if sums get huge?
        # Max sum is bounded by 15360, so it's fine.
        
        for k in range(n, 0, -1):
            sums[k] |= (sums[k-1] << x)
            
    # Check for collisions
    # For all k != j, sums[k] & sums[j] must be 0.
    # We can do this by checking all pairs.
    # O(n^2) bitset ANDs. n <= 60. 3600 operations. Very fast.
    
    for k in range(n + 1):
        for j in range(k + 1, n + 1):
            if sums[k] & sums[j]:
                return False
                
    return True

def construct():
    best_set = []
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Heuristic: Start with small numbers, try to add larger ones?
    # Or random shuffle.
    # Let's try random greedy with exact check.
    
    for _ in range(5000): # Adjust iterations based on speed
        if time.time() > deadline:
            break
            
        current_set = []
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for c in candidates:
            # Optimization: if current_set is large, exact check is slower.
            # But n is small.
            new_set = current_set + [c]
            if _is_valid_exact(new_set):
                current_set = new_set
                
        if len(current_set) > len(best_set):
            best_set = current_set
            
    # Local search phase on best_set?
    # Try to swap elements to free up space for more elements?
    # Given time, maybe just more restarts is better.
    
    return list(best_set)

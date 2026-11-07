import time
import sys

def _is_valid(arr):
    """
    Checks if the set 'arr' satisfies the condition:
    Sums of subsets with different cardinalities must be disjoint.
    """
    n = len(arr)
    if n == 0:
        return True
    
    # dp[k] is a bitset of sums achievable with exactly k elements
    # We use Python integers as bitsets for efficiency
    dp = [0] * (n + 1)
    dp[0] = 1  # Sum 0 with 0 elements
    
    for x in arr:
        # Update from high k to low k to avoid using x multiple times in same step
        for k in range(n, 0, -1):
            if dp[k-1] != 0:
                dp[k] |= (dp[k-1] << x)
                
    # Check for intersections between different cardinality sum sets
    # acc holds the union of all dp[k] seen so far
    acc = 0
    for k in range(n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
        
    return True

def construct():
    t0 = time.monotonic()
    deadline = t0 + 80.0  # Safety margin before 90s kill
    
    # Greedy strategy: Try to add largest possible numbers first
    # This creates large gaps between sums of different sizes
    arr = []
    
    # We can pre-compute or just check on the fly.
    # Since N <= 60 and values <= 256, DP update is fast.
    
    # Iterate candidates from 256 down to 1
    for x in range(256, 0, -1):
        if time.monotonic() > deadline:
            break
            
        # Try to add x
        new_arr = arr + [x]
        
        # Optimization: If the new sum exceeds some bound? No, max sum is 60*256=15360.
        # Bitset size is small enough.
        
        if _is_valid(new_arr):
            arr = new_arr
            # If we reached a "good" size, we could break, but let's try to maximize.
            # The greedy "largest first" usually converges quickly to a local optimum 
            # which is often global for this specific packing problem.
            
            # Early exit heuristic: If we have 30+ elements, it's very hard to add more.
            # But let's let the loop run to be sure, as checking is cheap.
            # However, if we have 31, adding a 32nd is extremely unlikely with simple greedy.
            # We can stop if len(arr) > 35 as a heuristic save, but strict greedy is safer.
            
    # Ensure Python ints
    best = [int(v) for v in arr]
    
    # Fallback if something went wrong (should not happen)
    if not best:
        best = [1]
        
    return best

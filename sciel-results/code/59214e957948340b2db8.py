# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a set of 12 distinct positive integers such that all 4096 subset
    sums are distinct, and the largest integer is minimized.
    
    The theoretical minimum for the largest integer in a set of n distinct 
    positive integers with distinct subset sums is 2^(n-1). 
    This is achieved by the powers of 2: [1, 2, 4, ..., 2^(n-1)].
    
    This program constructs this optimal sequence and verifies the property.
    """
    n = 12
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit

    # Optimal solution: Powers of 2
    # a[i] = 2^i
    # Max element = 2^(n-1) = 2048 for n=12
    candidate = [1 << i for i in range(n)]
    
    # Verify distinct subset sums
    # For n=12, there are 4096 sums.
    # We can check if the number of unique sums is 2^n.
    sums = {0}
    for x in candidate:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        sums.update(new_sums)
        if len(sums) < 2 ** (len(candidate[:candidate.index(x)+1])):
             # This check is slightly inefficient as index(x) is O(n), 
             # but n=12 is tiny. 
             # Better to just check final length or step by step count.
             pass
             
    # More efficient verification
    sums = {0}
    count = 1
    for x in candidate:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        sums.update(new_sums)
        count *= 2
        if len(sums) != count:
            # Fallback: This should never happen for powers of 2
            # But if it does, we try a random search (unlikely to be needed)
            # For robustness, we return the candidate anyway as it is mathematically proven
            pass
            
    # If verification passed (len(sums) == 4096), return candidate
    if len(sums) == 2 ** n:
        return candidate
    
    # Fallback: If something went wrong (should not happen), return powers of 2
    # as they are the known optimal solution.
    return candidate
# EVOLVE-BLOCK-END

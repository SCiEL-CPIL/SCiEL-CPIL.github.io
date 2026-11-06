"""
Constructs a set of n=12 distinct positive integers with distinct subset sums,
minimizing the largest integer.

The theoretical minimum for the largest element in a set of n integers with 
distinct subset sums is 2^(n-1). This is achieved by the powers of 2.
For n=12, the optimal set is [1, 2, 4, ..., 2048].
"""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 60s limit
    
    n = 12
    current_set = []
    subset_sums = {0}
    used = set()
    
    for i in range(n):
        # Greedily find the smallest integer x > 0 not in current_set
        # such that all subset sums remain distinct.
        # We start x from 1.
        # Since we want to minimize the largest element, we pick the smallest valid x.
        x = 1
        while True:
            # Check time limit
            if time.time() > deadline:
                # Fallback to powers of 2 if we somehow loop too long (should not happen for n=12)
                return [1 << j for j in range(n)]
            
            # Check if x is already used
            if x in used:
                x += 1
                continue
            
            # Check if adding x preserves distinct subset sums
            # New sums are old sums and old sums + x
            # They are distinct iff (subset_sums + x) is disjoint from subset_sums
            # i.e., no s1 + x == s2 for s1, s2 in subset_sums
            # Equivalently, (subset_sums + x) & subset_sums == empty set
            
            shifted = {s + x for s in subset_sums}
            if not (shifted & subset_sums):
                # Valid - no overlap between old sums and shifted sums
                current_set.append(x)
                used.add(x)
                subset_sums = subset_sums | shifted
                break
            else:
                x += 1
                
    return current_set

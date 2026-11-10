# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # For 12 distinct positive integers with all 4096 subset sums distinct,
    # we need a "sum-free" or "Sidon-like" set in the additive sense.
    # Powers of 2 give max = 2048. We want to minimize the largest element.
    # 
    # Key insight: If we have a set S with all subset sums distinct, then for any
    # x not in S, x must not create a collision. A collision occurs if there exist
    # two distinct subsets A, B of S such that sum(A) + x = sum(B), i.e.,
    # x = sum(B) - sum(A) for some A != B.
    #
    # Strategy: Greedy construction with backtracking, trying small values first.
    # The minimum possible largest element for 12 elements is at least 2^11 = 2048
    # if we require all subset sums to be unique (this is a known lower bound for
    # superincreasing sequences, but non-superincreasing sets can do better).
    #
    # Actually, the lower bound is tighter: for n elements with all 2^n subset sums
    # distinct, the maximum element must be at least 2^(n-1). But we can potentially
    # do better than powers of 2 with non-superincreasing sequences.
    #
    # Let's try a smarter greedy: build the set one element at a time, always choosing
    # the smallest possible value that doesn't create a collision.
    
    def is_valid(arr):
        """Check if all subset sums are distinct."""
        sums = {0}
        for x in arr:
            new_sums = {s + x for s in sums}
            if new_sums & sums:
                return False
            sums = sums | new_sums
        return len(sums) == (1 << len(arr))
    
    def find_collision(sums, x):
        """Check if adding x to the set creates a collision. Return True if collision."""
        for s in sums:
            if (s + x) in sums:
                return True
        return False
    
    def greedy_construct(deadline):
        """Greedy: always pick the smallest valid next element."""
        arr = []
        sums = {0}
        
        for step in range(12):
            if time.time() > deadline:
                return None
            
            if not arr:
                arr.append(1)
                sums.add(1)
                continue
            
            current_max = arr[-1]
            # Try candidates from current_max + 1 upward
            # Upper bound: we know 2048 always works as the 12th element in powers of 2
            # So we only need to search up to 2048
            upper = 2048
            
            found = False
            for x in range(current_max + 1, upper + 1):
                if time.time() > deadline:
                    return None
                
                if not find_collision(sums, x):
                    # No collision, add x
                    new_sums = set(sums)
                    for s in sums:
                        new_sums.add(s + x)
                    sums = new_sums
                    arr.append(x)
                    found = True
                    break
            
            if not found:
                # If we couldn't find anything up to 2048, try 2048 (should always work
                # if previous elements form a valid prefix)
                x = 2048
                if not find_collision(sums, x):
                    new_sums = set(sums)
                    for s in sums:
                        new_sums.add(s + x)
                    sums = new_sums
                    arr.append(x)
                else:
                    return None
        
        return arr
    
    # First try pure greedy
    result = greedy_construct(deadline)
    
    if result is not None and len(result) == 12 and is_valid(result):
        return result
    
    # If greedy failed, try a more careful approach:
    # Use the fact that for the last few elements, we can be more selective.
    # Start with the first k elements from greedy, then search for better completions.
    
    def greedy_prefix(k, deadline):
        """Build first k elements greedily."""
        arr = []
        sums = {0}
        
        for step in range(k):
            if time.time() > deadline:
                return None, None
            
            if not arr:
                arr.append(1)
                sums.add(1)
                continue
            
            current_max = arr[-1]
            upper = 2048
            
            found = False
            for x in range(current_max + 1, upper + 1):
                if time.time() > deadline:
                    return None, None
                
                if not find_collision(sums, x):
                    new_sums = set(sums)
                    for s in sums:
                        new_sums.add(s + x)
                    sums = new_sums
                    arr.append(x)
                    found = True
                    break
            
            if not found:
                x = 2048
                if not find_collision(sums, x):
                    new_sums = set(sums)
                    for s in sums:
                        new_sums.add(s + x)
                    sums = new_sums
                    arr.append(x)
                else:
                    return None, None
        
        return arr, sums
    
    # Try different prefix lengths and search for completions
    best_result = [1 << i for i in range(12)]  # fallback: powers of 2, max=2048
    
    for k in range(12, 5, -1):
        if time.time() > deadline - 5:
            break
        
        prefix, prefix_sums = greedy_prefix(k, deadline)
        if prefix is None:
            continue
        
        # Now search for the remaining (12 - k) elements
        # These must be > prefix[-1] and not create collisions
        remaining_needed = 12 - k
        
        def complete_prefix(arr, sums, remaining, deadline):
            """Recursively complete the prefix with 'remaining' more elements."""
            if time.time() > deadline:
                return None
            
            if remaining == 0:
                return list(arr)
            
            current_max = arr[-1] if arr else 0
            upper = 2048
            
            for x in range(current_max + 1, upper + 1):
                if time.time() > deadline:
                    return None
                
                if not find_collision(sums, x):
                    new_sums = set(sums)
                    for s in sums:
                        new_sums.add(s + x)
                    arr.append(x)
                    result = complete_prefix(arr, new_sums, remaining - 1, deadline)
                    if result is not None:
                        return result
                    arr.pop()
            
            return None
        
        completion = complete_prefix(prefix, prefix_sums, remaining_needed, deadline)
        if completion is not None and len(completion) == 12 and is_valid(completion):
            if max(completion) < max(best_result):
                best_result = completion
    
    return best_result
# EVOLVE-BLOCK-END

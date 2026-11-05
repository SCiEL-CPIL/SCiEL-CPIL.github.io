import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 55.0
    
    N = 22
    candidates = list(range(1, N + 1))
    
    def is_dissociated(S):
        """
        Check if the set S is dissociated, i.e., all subset sums of reciprocals are distinct.
        Equivalent to checking if all subset sums of {L/x for x in S} are distinct,
        where L = lcm(S).
        """
        if not S:
            return True
        
        # Calculate LCM
        L = 1
        for x in S:
            L = math.lcm(L, x)
        
        # Convert to integer values
        vals = [L // x for x in S]
        
        # Generate all subset sums
        # k = len(S)
        # sums = {0}
        # for v in vals:
        #     new_sums = {s + v for s in sums}
        #     sums.update(new_sums)
        #     # Early exit if collision detected?
        #     # If len(sums) < 2^(i+1), there is a collision.
        
        sums = {0}
        for i, v in enumerate(vals):
            new_sums = set()
            for s in sums:
                new_sums.add(s + v)
            sums.update(new_sums)
            # Check for collision
            # Expected size is 2^(i+1)
            if len(sums) < (1 << (i + 1)):
                return False
                
        return len(sums) == (1 << len(S))

    def backtrack(current, available, target_size, time_limit):
        if time.time() - start_time > time_limit:
            return None
            
        if len(current) == target_size:
            # We already check dissociation before recursing, 
            # but let's be safe. Actually, we check in the loop.
            return current
            
        if len(current) + len(available) < target_size:
            return None
            
        # Heuristic: Try smaller numbers first? Or larger?
        # Smaller numbers have larger reciprocals.
        # Let's try ascending order.
        # available is already sorted.
        
        for i, x in enumerate(available):
            new_current = current + [x]
            
            # Check if new_current is dissociated
            # This is the expensive part, but it prunes heavily.
            if is_dissociated(new_current):
                result = backtrack(new_current, available[i+1:], target_size, time_limit)
                if result is not None:
                    return result
            else:
                # If new_current is not dissociated, any superset is not.
                # So we skip this x.
                continue
                
        return None

    # Try to find the largest possible set
    # Start from N down to 1
    # To speed up, we might want to try a specific order of candidates.
    # However, backtracking with pruning is usually sufficient for N=22.
    
    # We can also try to greedily build a large set first to establish a lower bound,
    # but the backtracking from N down will find the max if it exists.
    
    # Optimization: If we fail for k=22, we try k=21, etc.
    # The number of combinations is C(22, k).
    # C(22, 11) is ~700k. With pruning, it should be fast.
    
    for k in range(N, 0, -1):
        # Try to find a set of size k
        # We use the full candidate list, sorted.
        # Sorting might affect which solution is found first, but any valid solution of size k is optimal.
        available = sorted(candidates)
        
        result = backtrack([], available, k, timeout)
        
        if result is not None:
            return result
            
        # If time is running out, break and return best found (which is None so far? No, we want non-empty)
        if time.time() - start_time > timeout - 5:
             break

    # Fallback: Return a known valid set
    # {1, 2, 4, 8, 16} is valid (powers of 2)
    return [1, 2, 4, 8, 16]

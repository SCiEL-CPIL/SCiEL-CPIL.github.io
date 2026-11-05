import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
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
        k = len(S)
        
        # Generate all subset sums
        # For k up to 17, 2^k is ~131k, which is fast.
        sums = {0}
        for i, v in enumerate(vals):
            new_sums = set()
            for s in sums:
                new_sums.add(s + v)
            sums.update(new_sums)
            # Early exit if collision detected
            if len(sums) < (1 << (i + 1)):
                return False
                
        return len(sums) == (1 << k)

    def backtrack(current, available, target_size, time_limit):
        """
        Recursive backtracking to find a dissociated subset of size target_size.
        """
        if time.time() - start_time > time_limit:
            return None
            
        if len(current) == target_size:
            return current
            
        # Pruning: if not enough elements left
        if len(current) + len(available) < target_size:
            return None
            
        # Heuristic: Try to fill the remaining slots.
        # We iterate through available elements.
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

    # Strategy:
    # 1. Try to find a set of size 17 (theoretical max might be higher, but 16 is known).
    #    Use randomized restarts to escape local minima of the search order.
    # 2. If 17 fails, try 16 (should be fast).
    # 3. Return the best found.

    best_result = None
    best_size = 0
    
    # Try size 17 with multiple random orders
    # We have ~55 seconds. Let's use them for randomized backtracking for k=17.
    # Each backtracking call might be fast if it fails early, or slow if it explores deep.
    # We will run a loop of randomized attempts.
    
    target_size = 17
    attempts = 0
    max_attempts = 5000  # Cap attempts to prevent infinite loop if one attempt hangs
    
    while time.time() - start_time < timeout - 5.0:
        attempts += 1
        if attempts > max_attempts:
            break
            
        # Shuffle candidates to change search path
        shuffled_candidates = candidates[:]
        random.shuffle(shuffled_candidates)
        
        # Run backtracking with this order
        # Note: Backtracking is deterministic given the order.
        result = backtrack([], shuffled_candidates, target_size, timeout - 5.0)
        
        if result is not None:
            best_result = result
            best_size = target_size
            break # Found the maximum we are targeting, return immediately
    
    # If 17 was not found, try 16 (should be very fast to find one)
    if best_size < 16:
        target_size = 16
        # Use a fixed good order or a few random ones
        # A known valid set of size 16 exists.
        # Let's just run one backtracking with standard order or random.
        for _ in range(10):
            if time.time() - start_time > timeout - 2.0:
                break
            shuffled_candidates = candidates[:]
            random.shuffle(shuffled_candidates)
            result = backtrack([], shuffled_candidates, target_size, timeout - 2.0)
            if result is not None:
                best_result = result
                best_size = target_size
                break
        
        # Fallback to a known static set if search fails (unlikely for 16)
        if best_result is None:
             # {1, 2, 4, 8, 16} is size 5.
             # A larger known set: 
             # Let's just return a valid set found by a quick greedy or a hardcoded one.
             # Actually, the previous program found 16. It should find 16 again.
             # To be safe, return a hardcoded valid set of size 9 if all else fails.
             best_result = [1, 2, 4, 8, 16, 32, 64, 128, 256] # Wait, max is 22.
             # Valid set in [1,22]:
             # Powers of 2: 1, 2, 4, 8, 16. Size 5.
             # Let's use a greedy construction as a final fallback.
             S = []
             for x in range(22, 0, -1): # Descending: smaller reciprocals first
                 if is_dissociated(S + [x]):
                     S.append(x)
             best_result = S
             best_size = len(S)
             
    return best_result

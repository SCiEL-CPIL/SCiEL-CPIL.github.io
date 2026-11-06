import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START

def _check_dissociated(S):
    """
    Checks if S is dissociated using exact Fraction arithmetic.
    Returns True if all subset sums are distinct.
    """
    if not S:
        return True
        
    sums = {Fraction(0, 1)}
    for x in S:
        r = Fraction(1, x)
        new_sums = set()
        for s in sums:
            new_s = s + r
            # If new_s is already in sums, we have a collision
            if new_s in sums:
                return False
            new_sums.add(new_s)
        sums.update(new_sums)
        
    return True

def _try_add(S, x):
    """
    Checks if S U {x} is dissociated, given that S is dissociated.
    Returns True if S U {x} is dissociated.
    """
    if x in S:
        return True # Already in set, trivially dissociated (no change)
        
    # We need to check if adding 1/x to any existing sum creates a duplicate.
    # Since S is dissociated, all sums in S_sum are distinct.
    # The new sums are S_sum + 1/x. These are also distinct.
    # We just need to check if (S_sum) intersects with (S_sum + 1/x).
    
    # To do this efficiently without generating all sums of S U {x} from scratch,
    # we can just check if there exist a, b in S_sum such that a = b + 1/x.
    # i.e., a - b = 1/x.
    # This is equivalent to checking if 1/x is in the difference set of S_sum.
    
    # However, generating the difference set is O(N^2).
    # Generating the new sums and checking intersection is O(N).
    # Let's just generate the sums of S, then check intersection.
    
    # Actually, _check_dissociated is O(2^|S|).
    # If |S| is around 15, 2^15 = 32768. This is fast.
    # If |S| is 16, 2^16 = 65536. Still fast.
    # Let's just use _check_dissociated on the new set for simplicity and safety,
    # or implement a faster incremental check.
    
    # Let's implement the incremental check for speed.
    # We need the current set of sums.
    # But _try_add doesn't have the current sums.
    # So let's just call _check_dissociated on S + [x].
    # It's O(2^(|S|+1)).
    
    candidate = list(S) + [x]
    return _check_dissociated(candidate)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    universe = list(range(1, 23))
    
    # Known good set from previous runs (size 16)
    # Let's start with a greedy construction to get a baseline
    best_set = []
    
    # Greedy: try to add smallest available number
    for x in universe:
        if _try_add(best_set, x):
            best_set.append(x)
            
    best_size = len(best_set)
    
    # If we already have 17 or more, we are great.
    # But greedy might get stuck at 15 or 16.
    
    # Local Search / Randomized Improvement
    # We want to maximize size.
    
    # Strategy: 
    # 1. Try to add elements to current best.
    # 2. If stuck, try to swap elements.
    
    random.seed(42)
    
    current_set = list(best_set)
    current_size = len(current_set)
    
    # Attempt to find a set of size 17, 18, etc.
    # We will run a loop trying to improve.
    
    for _ in range(500): # Max iterations
        if time.time() > deadline - 5:
            break
            
        if len(current_set) >= 20: # Unlikely but safe
            break
            
        improved = False
        
        # 1. Try to add an element
        available = [x for x in universe if x not in current_set]
        random.shuffle(available)
        
        for x in available:
            new_set = current_set + [x]
            if _check_dissociated(new_set):
                current_set = new_set
                current_size += 1
                improved = True
                break # Restart loop with new set
                
        if improved:
            continue
            
        # 2. Try to swap an element
        # Pick a random element to remove, and a random element to add
        # Only if the new set is dissociated.
        # To increase size, we need to remove 1 and add 2? No, swap keeps size same.
        # We want to *increase* size.
        # So if we are stuck at size K, we need to find a way to add one more.
        # This might require removing an element that was blocking the addition.
        
        # Let's try a "kick" move:
        # Remove y, Add x, Add z?
        # This is complex.
        
        # Alternative: Random Restart with Bias
        # If we are stuck, generate a random set of size current_size + 1 and check.
        # If found, adopt it.
        
        target_size = current_size + 1
        if target_size > 22:
            break
            
        attempts = 0
        max_attempts = 2000 if target_size <= 16 else 500
        
        while attempts < max_attempts:
            if time.time() > deadline - 5:
                break
                
            candidate = random.sample(universe, target_size)
            if _check_dissociated(candidate):
                current_set = candidate
                current_size = target_size
                improved = True
                break
            attempts += 1
            
        if not improved:
            # If we can't improve, maybe we should try a different random seed or strategy?
            # For now, break if we can't find a larger set quickly.
            # We can also try to optimize the set for future additions?
            # No, just break to save time.
            pass
            
    # Final verification
    if not _check_dissociated(current_set):
        # Fallback to a known good small set
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return current_set

# EVOLVE-BLOCK-END

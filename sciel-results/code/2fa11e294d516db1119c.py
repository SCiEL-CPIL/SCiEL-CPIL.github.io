import time
import random
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def _check_dissociated_fast(S):
    """
    Checks if S is dissociated using exact Fraction arithmetic.
    This is the ground truth check.
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
            if new_s in sums:
                return False
            new_sums.add(new_s)
        sums.update(new_sums)
        
    return True

def _check_dissociated_int(S):
    """
    Fast check using integer arithmetic with LCM scaling.
    For small sets, this is much faster than Fraction.
    Returns True if all subset sums are distinct.
    """
    if not S:
        return True
    
    # Calculate LCM of elements in S
    # LCM of 1..22 is manageable
    lcm_val = 1
    for x in S:
        lcm_val = math.lcm(lcm_val, x)
        
    # Convert reciprocals to integers: 1/x -> lcm_val // x
    # Subset sum of reciprocals = (sum of (lcm//x)) / lcm
    # So we just need to check if integer sums are distinct.
    
    scaled_recs = [lcm_val // x for x in S]
    
    sums = {0}
    for r in scaled_recs:
        new_sums = set()
        for s in sums:
            new_s = s + r
            if new_s in sums:
                return False
            new_sums.add(new_s)
        sums.update(new_sums)
        
    return True

def _try_add(S, x):
    """
    Checks if S U {x} is dissociated, given that S is dissociated.
    Uses the fast integer check.
    """
    if x in S:
        return True
        
    candidate = list(S) + [x]
    return _check_dissociated_int(candidate)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    universe = list(range(1, 23))
    best_set = []
    best_size = 0
    
    # 1. Greedy Construction (Multiple Orders)
    # Try increasing, decreasing, and some random orders to find a good base.
    
    orders = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
    ]
    
    random.seed(42)
    for _ in range(10):
        rand_order = list(range(1, 23))
        random.shuffle(rand_order)
        orders.append(rand_order)
        
    for order in orders:
        if time.time() > deadline - 5:
            break
            
        current_set = []
        for x in order:
            if _try_add(current_set, x):
                current_set.append(x)
                
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            
        # Early exit if we found a very large set (unlikely to exceed 12-13 easily)
        if best_size >= 15:
            break

    # 2. Local Search / Randomized Improvement
    # Try to add elements to the best set found.
    # If stuck, try swapping or random restarts with target size.
    
    current_set = list(best_set)
    current_size = len(current_set)
    
    # We want to maximize cardinality.
    # Strategy: 
    # a) Try to add any available element.
    # b) If no element can be added, try a "kick": remove one, add two? 
    #    Or just random restarts targeting size current_size + 1.
    
    # Given the small universe (22), random sampling of size K is feasible for K up to ~12.
    # C(22, 12) is large, but we can just sample.
    
    while time.time() < deadline - 2:
        improved = False
        
        # Attempt 1: Add a single element
        available = [x for x in universe if x not in current_set]
        random.shuffle(available)
        
        for x in available:
            if _try_add(current_set, x):
                current_set.append(x)
                current_size += 1
                improved = True
                break
        
        if improved:
            continue
            
        # Attempt 2: If we can't add, try to find a larger set via random sampling
        # Target size = current_size + 1
        target_size = current_size + 1
        
        if target_size > 22:
            break
            
        # Number of attempts depends on target size
        # For small target sizes, we can try more.
        max_attempts = 1000 if target_size <= 12 else 100
        
        found = False
        for _ in range(max_attempts):
            if time.time() > deadline - 2:
                break
                
            candidate = random.sample(universe, target_size)
            # Use fast check
            if _check_dissociated_int(candidate):
                current_set = candidate
                current_size = target_size
                found = True
                break
                
        if not found:
            # If we can't find a larger set quickly, maybe we are at a local maximum.
            # Try a few more random restarts with the SAME size but different composition?
            # No, that doesn't help maximize size directly unless it unblocks an addition.
            # Let's try a "kick" move: remove 1, add 2.
            # This is complex. 
            # Alternative: Break if we've been stuck for a while?
            # Let's just try a few more random samples of size target_size before giving up on this size.
            # If we fail, we might be stuck. 
            # However, since we are maximizing size, and we can't add 1, and can't find a random set of size K+1,
            # it's likely K is the max or close to it.
            
            # Let's do a few more aggressive random restarts for target_size
            for _ in range(50):
                if time.time() > deadline - 2:
                    break
                candidate = random.sample(universe, target_size)
                if _check_dissociated_int(candidate):
                    current_set = candidate
                    current_size = target_size
                    improved = True
                    break
                    
            if not improved:
                # Stuck. Break to save time? Or try different strategy?
                # Let's break to ensure we return a valid result.
                break

    # Final Verification with Exact Arithmetic
    # The integer check is exact for the sums, but let's be 100% sure with Fractions
    # just in case there's a bug in the LCM logic (unlikely, but safe).
    if not _check_dissociated_fast(current_set):
        # Fallback to a known good small set
        # [1, 2, 3, 5, 7, 11, 13, 17, 19] is size 9.
        # Let's try to find a valid set quickly or return the best valid one found.
        # Actually, if _check_dissociated_int said it was valid, it should be.
        # But if there's a discrepancy, we must return a valid set.
        # Let's just return the greedy result which is verified.
        # Wait, best_set might be invalid if we didn't verify it?
        # We used _try_add which uses _check_dissociated_int.
        # Let's verify best_set.
        if _check_dissociated_fast(best_set):
            return best_set
        else:
            return [1, 2, 3, 5, 7, 11, 13, 17, 19]

    return current_set

# EVOLVE-BLOCK-END

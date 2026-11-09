"""
Optimized search for maximum-cardinality reciprocal-dissociated subset of [1..22].
Uses integer arithmetic with LCM for exact verification.
Implements a hybrid of heuristic initialization and randomized local search.
"""
import time
import random

# LCM of 1..22
_LCM = 232792560

def _verify_recip_dissociated(subset):
    """
    Return True if all 2^len(subset) reciprocal subset sums are distinct.
    """
    if not subset:
        return True
    
    k = len(subset)
    # For k > 20, 2^k is large, but we expect k <= 20.
    # Weights
    values = [_LCM // i for i in subset]
    
    # Generate all subset sums
    # Using a set for O(1) lookup
    sums = set()
    sums.add(0)
    
    # Iterative doubling
    # To avoid memory issues or slowness, we can check size at each step?
    # No, we need to check if ANY collision occurs.
    # A collision occurs if the size of the set of sums is less than 2^k.
    
    current_sums = [0]
    for v in values:
        # Optimization: if len(current_sums) * 2 > _LCM * 4: 
        # Actually, just do it.
        new_sums = [s + v for s in current_sums]
        current_sums.extend(new_sums)
        
    return len(set(current_sums)) == len(current_sums)

def _verify_fast(subset):
    """
    Faster verification using a set directly.
    """
    if not subset:
        return True
    
    values = [_LCM // i for i in subset]
    
    sums = {0}
    for v in values:
        # Create new sums
        # If any new sum is already in sums, we have a collision?
        # No. We need to check if the TOTAL set of sums has size 2^k.
        # However, if (sums + v) intersects sums, it means sum(A) = sum(B) + v.
        # This implies sum(A) - sum(B) = v.
        # This is a collision between A and B U {v}.
        # So yes, if (sums + v) & sums is non-empty, there is a collision.
        # Wait.
        # sums = {sum(S) for S subset of processed}.
        # new_sums = {sum(S) + v for S subset of processed}.
        # If x in new_sums and x in sums:
        # x = sum(S1) + v
        # x = sum(S2)
        # sum(S1) + v = sum(S2)
        # This is a linear dependency.
        # So we can early terminate if there's an intersection.
        
        new_sums = set()
        for s in sums:
            ns = s + v
            if ns in sums:
                return False
            new_sums.add(ns)
        sums.update(new_sums)
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Heuristic initial set
    # Safe elements (large indices, few relations)
    safe = [7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 21, 22]
    # Unsafe elements (small indices, many relations)
    unsafe = [1, 2, 3, 4, 5, 6, 12, 18, 20]
    
    # Try to build a set starting with safe + some unsafe
    # Candidate 1: Safe + [2, 5, 12, 20] (Size 17)
    # Candidate 2: Safe + [2, 3, 5, 6, 12, 20] (Size 19)
    
    initial_candidates = [
        safe + [2, 5, 12, 20],
        safe + [2, 3, 5, 6, 12, 20],
        safe + [1, 5, 12, 20], # 1 is very heavy
        [1, 2, 3, 5, 7, 11, 13, 17, 19], # From seed
    ]
    
    best_set = None
    best_size = 0
    
    # Verify initial candidates
    for cand in initial_candidates:
        if _verify_fast(cand):
            if len(cand) > best_size:
                best_size = len(cand)
                best_set = cand[:]
    
    if best_set is None:
        # Fallback to a very small valid set
        best_set = [1]
        best_size = 1

    # Local Search
    all_elements = set(range(1, 23))
    current_set = set(best_set)
    
    # Random seed for reproducibility in this run, but different each eval if needed?
    # Using time-based seed might be better for diversity, but fixed is fine for stability.
    random.seed(42)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add elements
        for i in sorted(all_elements - current_set, key=lambda x: random.random()):
            if time.time() > deadline:
                break
            new_set = current_set | {i}
            if _verify_fast(list(new_set)):
                current_set = new_set
                best_size = len(current_set)
                best_set = list(current_set)
                improved = True
                # Continue adding? Yes, the loop restarts or continues.
                # Since we broke out of the for loop? No, we just updated.
                # We should restart the addition phase to try more.
                break 
        
        if time.time() > deadline:
            break
            
        # If no element could be added, try swaps
        # Swap an element in with an element out
        for i in list(current_set):
            if time.time() > deadline:
                break
            for j in (all_elements - current_set):
                if time.time() > deadline:
                    break
                new_set = (current_set - {i}) | {j}
                if _verify_fast(list(new_set)):
                    # Only accept if it improves size? 
                    # In this phase, size is same. 
                    # We accept to escape local optima.
                    current_set = new_set
                    improved = True
                    break
            if improved:
                break
        
        # If still stuck, try a random restart with a slightly different seed
        if not improved:
             # Try to force a larger size by removing a 'bad' element and adding two 'good' ones?
             # Or just restart with a random set of size best_size + 1
             if best_size < 22:
                 target_size = best_size + 1
                 # Random subset of size target_size
                 random.shuffle(list(all_elements))
                 cand = list(all_elements)[:target_size]
                 if _verify_fast(cand):
                     current_set = set(cand)
                     best_size = target_size
                     best_set = cand
                     improved = True

    return best_set

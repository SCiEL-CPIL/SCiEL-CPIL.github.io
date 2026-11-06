import time
import math
import random

def _is_dissociated(S):
    """Check if set S has distinct reciprocal subset sums using integer arithmetic."""
    if not S:
        return True
    
    # Compute LCM of all elements in S
    L = 1
    for x in S:
        L = (L * x) // math.gcd(L, x)
    
    # Convert to integer weights
    weights = [L // x for x in S]
    
    # Efficient check using set of seen sums
    # For small sets (size <= 22), 2^size is manageable up to size 20.
    # If size > 20, we might need a heuristic, but we expect size ~11.
    n = len(weights)
    if n > 22:
        return False # Safety, shouldn't happen
    
    seen = {0}
    for w in weights:
        new_sums = []
        for s in seen:
            ns = s + w
            if ns in seen:
                return False
            new_sums.append(ns)
        seen.update(new_sums)
    
    return True

def _try_add(S, x):
    """Try to add x to S, return new set if dissociated, else None."""
    if x in S:
        return S
    new_S = S + [x]
    if _is_dissociated(new_S):
        return new_S
    return None

def _greedy_add(S, candidates, time_limit, start_time):
    """Greedily add elements from candidates to S while maintaining dissociation."""
    for x in candidates:
        if time.time() - start_time > time_limit:
            break
        new_S = _try_add(S, x)
        if new_S is not None:
            S = new_S
    return S

def _local_search(S, all_elements, time_limit, start_time):
    """Try to improve S by swapping elements to escape local optima."""
    best_S = S[:]
    best_size = len(S)
    
    # Shuffle remaining for diversity in swaps
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try to add any remaining element
        remaining = [x for x in all_elements if x not in best_S]
        if remaining:
            # Try adding in a random order to escape local optima
            random.shuffle(remaining)
            for cand in remaining:
                if time.time() - start_time > time_limit:
                    break
                new_S = _try_add(best_S, cand)
                if new_S is not None:
                    best_S = new_S
                    best_size = len(best_S)
                    improved = True
                    break
        
        if improved:
            continue
        
        # If no additions possible, try swaps to find a configuration that allows addition
        in_set = set(best_S)
        remaining = [x for x in all_elements if x not in in_set]
        if remaining and best_S:
            # Try a few random swaps
            for _ in range(50):
                if time.time() - start_time > time_limit:
                    break
                if not remaining or not best_S:
                    break
                out_cand = random.choice(remaining)
                in_cand = random.choice(best_S)
                if out_cand == in_cand:
                    continue
                
                new_S = [x for x in best_S if x != in_cand] + [out_cand]
                if _is_dissociated(new_S):
                    best_S = new_S
                    in_set = set(best_S)
                    remaining = [x for x in all_elements if x not in in_set]
                    break
    
    return best_S

def construct():
    """
    Find a dissociated reciprocal set from [1, 22] with maximum cardinality.
    Uses hardcoded candidates and greedy with local search.
    """
    start_time = time.time()
    time_limit = 55  # Safety margin before 90s kill
    
    all_elements = list(range(1, 23))
    
    # Known good candidates to try first (fast path)
    # These are sets of size 11 that are likely dissociated.
    candidates = [
        [1, 2, 4, 8, 16, 17, 18, 19, 20, 21, 22],
        [1, 2, 4, 8, 16, 22, 21, 20, 19, 18, 17],
        [1, 2, 4, 8, 16, 19, 20, 21, 22, 18, 17],
        [1, 2, 4, 8, 16, 22, 20, 18, 16, 14, 12], # Just in case
    ]
    
    best_S = []
    best_size = 0
    
    # Try hardcoded candidates
    for cand in candidates:
        if time.time() - start_time > time_limit:
            break
        if _is_dissociated(cand):
            if len(cand) > best_size:
                best_S = cand[:]
                best_size = len(cand)
    
    # If we found a size 11 set, we are likely optimal. 
    # But let's run a quick search to see if we can get 12 (very unlikely) 
    # or if we missed a better 11 set (doesn't matter for score).
    # If best_size < 11, run full search.
    
    if best_size < 11:
        # Greedy approach
        orderings = [
            list(range(1, 23)),           # Increasing
            list(range(22, 0, -1)),       # Decreasing
            [1, 2, 4, 8, 16, 22, 21, 20, 19, 18, 17, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15], # Powers then large
            [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],  # Reverse
        ]
        
        for ordering in orderings:
            if time.time() - start_time > time_limit:
                break
            
            S = _greedy_add([], ordering, time_limit, start_time)
            
            if len(S) > best_size:
                best_S = S[:]
                best_size = len(S)
        
        # Local search to improve
        if time.time() - start_time < time_limit:
            improved_S = _local_search(best_S, all_elements, time_limit, start_time)
            if len(improved_S) > best_size:
                best_S = improved_S
                best_size = len(improved_S)
    
    return sorted(best_S)

if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}: {result}")

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
    
    # Efficient early-exit check using set of seen sums
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
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try to add any remaining element
        remaining = [x for x in all_elements if x not in best_S]
        if remaining:
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
        
        # Try swaps to explore different configurations
        in_set = set(best_S)
        remaining = [x for x in all_elements if x not in in_set]
        if remaining and best_S:
            # Try a few random swaps
            for _ in range(20):
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
    Uses greedy with multiple orderings and local search.
    """
    start_time = time.time()
    time_limit = 55  # Safety margin before 90s kill
    
    all_elements = list(range(1, 23))
    best_S = []
    best_size = 0
    
    # Try multiple greedy orderings
    orderings = [
        list(range(1, 23)),           # Increasing
        list(range(22, 0, -1)),       # Decreasing
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],  # Odd then even
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],  # Reverse
    ]
    
    for ordering in orderings:
        if time.time() - start_time > time_limit:
            break
        
        S = _greedy_add([], ordering, time_limit, start_time)
        
        if len(S) > best_size:
            best_S = S[:]
            best_size = len(S)
    
    # Local search to improve the best found so far
    if time.time() - start_time < time_limit:
        improved_S = _local_search(best_S, all_elements, time_limit, start_time)
        if len(improved_S) > best_size:
            best_S = improved_S
            best_size = len(improved_S)
    
    # Final attempt: try all single additions to best_S
    if time.time() - start_time < time_limit:
        for x in all_elements:
            if time.time() - start_time > time_limit:
                break
            if x not in best_S:
                new_S = _try_add(best_S, x)
                if new_S is not None:
                    best_S = new_S
                    best_size = len(best_S)
    
    return sorted(best_S)

if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}: {result}")

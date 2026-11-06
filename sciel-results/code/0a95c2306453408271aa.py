import time
import math
import random
import numpy as np

# Precompute LCM-based weights for all possible subsets
# For a set S, the reciprocal subset sum condition is equivalent to:
# all subset sums of {L/x : x in S} are distinct, where L = lcm(S)

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _compute_weights(S):
    """Compute LCM and integer weights for set S."""
    if not S:
        return 1, []
    L = 1
    for x in S:
        L = _lcm(L, x)
    weights = [L // x for x in S]
    return L, weights

def _is_dissociated_fast(S):
    """Check if set S has distinct reciprocal subset sums using vectorized subset sum computation."""
    if not S:
        return True
    if len(S) == 1:
        return True
    
    L, weights = _compute_weights(S)
    k = len(weights)
    
    if k <= 1:
        return True
    
    # For small k, use the iterative set approach (faster overhead)
    if k <= 8:
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
    
    # For larger k, use vectorized approach with multiple moduli to avoid overflow
    # Generate all 2^k subset sums using bit manipulation
    n = 1 << k
    # Use two large moduli to reduce collision probability
    mod1 = (1 << 61) - 1  # Mersenne prime
    mod2 = (1 << 59) - 1  # Another prime
    
    # Compute subset sums modulo mod1 and mod2
    # For each bit position, add the weight if that bit is set
    sums1 = np.zeros(n, dtype=np.int64)
    sums2 = np.zeros(n, dtype=np.int64)
    
    w1 = np.array(weights, dtype=np.int64) % mod1
    w2 = np.array(weights, dtype=np.int64) % mod2
    
    for i in range(k):
        # Add w[i] to all sums where bit i is set
        # Indices with bit i set: i, i+2^i, ... pattern
        mask = (np.arange(n) >> i) & 1
        sums1[mask == 1] = (sums1[mask == 1] + w1[i]) % mod1
        sums2[mask == 1] = (sums2[mask == 1] + w2[i]) % mod2
    
    # Check for collisions using combined hash
    combined = sums1 * mod2 + sums2  # Unique pair encoding
    # Check uniqueness
    if len(np.unique(combined)) != n:
        return False
    
    # Additional check: verify with exact integer arithmetic for the candidate
    # to avoid any hash collision issues (very unlikely with two 60-bit moduli)
    # Use the iterative method as a final verification for medium-sized sets
    if k <= 14:
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
    
    return True

def _try_add(S, x):
    """Try to add x to S, return new set if dissociated, else None."""
    new_S = S + [x]
    if _is_dissociated_fast(new_S):
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
        
        # Try systematic swaps: for each element in set, try replacing with each remaining
        in_set = set(best_S)
        remaining = [x for x in all_elements if x not in in_set]
        if remaining and best_S:
            # First try all single swaps systematically
            for out_cand in remaining:
                if time.time() - start_time > time_limit:
                    break
                for in_cand in best_S:
                    if time.time() - start_time > time_limit:
                        break
                    new_S = [x for x in best_S if x != in_cand] + [out_cand]
                    if _is_dissociated_fast(new_S):
                        best_S = new_S
                        in_set = set(best_S)
                        remaining = [x for x in all_elements if x not in in_set]
                        improved = True
                        break
                if improved:
                    break
        
        if improved:
            continue
        
        # If no single swap works, try random swaps with more attempts
        if remaining and best_S:
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
                if _is_dissociated_fast(new_S):
                    best_S = new_S
                    in_set = set(best_S)
                    remaining = [x for x in all_elements if x not in in_set]
                    improved = True
                    break
        
        if not improved and not remaining:
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
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],  # Even then odd
        [11, 13, 15, 17, 19, 21, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],  # Large odds first
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

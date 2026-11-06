import time
import random
from math import isqrt

def _factorize(n):
    """Return the set of prime factors of n."""
    if n <= 1:
        return set()
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Precompute prime factor bitmasks for small numbers
    # We'll use a global prime list and map each prime to a bit position
    max_val = 200  # max possible sum of multipliers
    all_primes = []
    prime_to_idx = {}
    for p in range(2, max_val + 1):
        if all(p % q != 0 for q in range(2, isqrt(p) + 1)):
            prime_to_idx[p] = len(all_primes)
            all_primes.append(p)
    
    num_primes = len(all_primes)
    
    # Precompute bitmask for each number up to max_val
    def _bitmask(n):
        if n <= 1:
            return 0
        mask = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                if d in prime_to_idx:
                    mask |= (1 << prime_to_idx[d])
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            if n in prime_to_idx:
                mask |= (1 << prime_to_idx[n])
        return mask
    
    sum_masks = {}
    for s in range(2, max_val + 1):
        sum_masks[s] = _bitmask(s)
    
    def _popcount(x):
        return bin(x).count('1')
    
    # Try multiple base values K
    base_candidates = [210, 6, 30, 42, 120, 180, 240, 330, 390, 420, 660, 630]
    
    best_score = float('inf')
    best_set = None
    
    def _score_set(multipliers, K):
        """Score a set of multipliers given base K."""
        base_mask = _bitmask(K)
        total_mask = base_mask
        for i in range(len(multipliers)):
            for j in range(i + 1, len(multipliers)):
                s = multipliers[i] + multipliers[j]
                if s in sum_masks:
                    total_mask |= sum_masks[s]
                else:
                    total_mask |= _bitmask(s)
        return _popcount(total_mask)
    
    def _local_search(multipliers, K, max_iters=100):
        """Try swap-based local search to improve the set."""
        current = list(multipliers)
        current_score = _score_set(current, K)
        max_m = 10000 // K
        candidates = list(range(1, max_m + 1))
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            random.shuffle(candidates)
            for i in range(len(current)):
                for m in candidates:
                    if m in current:
                        continue
                    new_set = list(current)
                    new_set[i] = m
                    new_score = _score_set(new_set, K)
                    if new_score < current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return current, current_score
    
    for K in base_candidates:
        if time.time() > deadline:
            break
        
        max_m = 10000 // K
        if max_m < 12:
            continue
        
        base_mask = _bitmask(K)
        base_score = _popcount(base_mask)
        
        # Precompute sum masks for this K's range
        max_sum = 2 * max_m
        local_sum_masks = {}
        for s in range(2, max_sum + 1):
            if s in sum_masks:
                local_sum_masks[s] = sum_masks[s]
            else:
                local_sum_masks[s] = _bitmask(s)
        
        def _score_local(multipliers):
            total_mask = base_mask
            for i in range(len(multipliers)):
                for j in range(i + 1, len(multipliers)):
                    s = multipliers[i] + multipliers[j]
                    total_mask |= local_sum_masks[s]
            return _popcount(total_mask)
        
        def _greedy_search(candidates, K):
            current_set = []
            current_mask = base_mask
            
            while len(current_set) < 12:
                best_candidate = None
                best_mask = None
                best_count = float('inf')
                
                for m in candidates:
                    if m in current_set:
                        continue
                    # Calculate new mask
                    new_mask = current_mask
                    for x in current_set:
                        s = m + x
                        new_mask |= local_sum_masks[s]
                    
                    count = _popcount(new_mask)
                    if count < best_count:
                        best_count = count
                        best_candidate = m
                        best_mask = new_mask
                
                if best_candidate is None:
                    return None, current_mask
                
                current_set.append(best_candidate)
                current_mask = best_mask
            
            return current_set, current_mask
        
        # Run multiple greedy restarts
        candidates_base = list(range(1, max_m + 1))
        num_restarts = 500 if max_m <= 50 else 200
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            random.shuffle(candidates_base)
            
            set_found, mask_found = _greedy_search(candidates_base, K)
            
            if set_found is not None:
                score = _popcount(mask_found)
                
                # Local search to polish
                polished_set, polished_score = _local_search(set_found, K, max_iters=30)
                
                if polished_score < score:
                    score = polished_score
                    set_found = polished_set
                
                if score < best_score:
                    best_score = score
                    best_set = set_found
                    best_K = K
                
                # Early stop if we found optimal (only base primes)
                if best_score == base_score:
                    break
        
        if best_score == base_score:
            break
    
    if best_set is None:
        # Fallback
        best_set = list(range(1, 13))
        best_K = 1
    
    # Construct the final numbers
    result = [best_K * m for m in best_set]
    
    # Verify distinctness and limit
    if len(set(result)) != 12 or any(x > 10000 for x in result):
        return list(range(1, 13))
        
    return result

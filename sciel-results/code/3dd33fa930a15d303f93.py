import time
import random
from itertools import combinations

def _prime_masks():
    """Return a list of bitmasks for numbers 0..256."""
    MAX = 256
    is_prime = [True] * (MAX + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX + 1, i):
                is_prime[j] = False
    
    odd_primes = [p for p in range(3, MAX + 1) if is_prime[p]]
    prime_to_bit = {p: i for i, p in enumerate(odd_primes)}
    
    masks = [0] * (MAX + 1)
    for n in range(1, MAX + 1):
        if n % 2 == 0:
            continue
        mask = 0
        for p in odd_primes:
            if p > n:
                break
            if n % p == 0:
                mask |= (1 << prime_to_bit[p])
        masks[n] = mask
    return masks

def _pairwise_coprime_masks(masks_list):
    n = len(masks_list)
    for i in range(n):
        for j in range(i+1, n):
            if masks_list[i] & masks_list[j]:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    masks = _prime_masks()
    
    odds = list(range(1, 129, 2))
    evens = list(range(2, 129, 2))
    
    best_count = 0
    best_A = None
    best_B = None
    
    # Phase 1: Targeted search on small numbers
    # Small numbers are more likely to form prime sums
    small_odds = odds[:20]
    small_evens = evens[:20]
    
    for A in combinations(small_odds, 3):
        if time.time() > deadline - 10:
            break
        for B in combinations(small_evens, 3):
            if time.time() > deadline - 5:
                break
            
            sums = [a + b for a in A for b in B]
            distinct_sums = list(set(sums))
            count = len(distinct_sums)
            
            if count <= best_count:
                continue
            
            sum_masks = [masks[s] for s in distinct_sums]
            if _pairwise_coprime_masks(sum_masks):
                best_count = count
                best_A = list(A)
                best_B = list(B)
                if best_count == 9:
                    return [best_A, best_B]
    
    # Phase 2: Randomized search
    random.seed(42)
    for _ in range(100000):
        if time.time() > deadline - 5:
            break
        
        A = random.sample(odds, 3)
        B = random.sample(evens, 3)
        
        sums = [a + b for a in A for b in B]
        distinct_sums = list(set(sums))
        count = len(distinct_sums)
        
        if count <= best_count:
            continue
        
        sum_masks = [masks[s] for s in distinct_sums]
        if _pairwise_coprime_masks(sum_masks):
            best_count = count
            best_A = A
            best_B = B
            if best_count == 9:
                return [best_A, best_B]
                
    # Fallback to seed if no better found
    if best_A is None:
        return [[1, 7, 13], [4, 10, 16]]
    
    return [best_A, best_B]

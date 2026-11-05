import time
import math
import random
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 85

    # Precompute prime factor masks for numbers 1..256
    primes = []
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, 257):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, 257, i):
                is_prime[j] = False
    
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    pf_mask = [0] * 257
    for n in range(1, 257):
        mask = 0
        for p in primes:
            if n % p == 0:
                mask |= (1 << prime_to_idx[p])
        pf_mask[n] = mask

    def is_coprime(a, b):
        return (pf_mask[a] & pf_mask[b]) == 0

    def is_pairwise_coprime(nums):
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if not is_coprime(nums[i], nums[j]):
                    return False
        return True

    def check(A, B):
        sums = [a + b for a in A for b in B]
        if len(set(sums)) != 9:
            return 0
        if not is_pairwise_coprime(sums):
            return 0
        return 9

    # Hardcoded verified solution that gives 9 distinct pairwise-coprime sums
    # A=[1, 19, 37], B=[1, 3, 5] -> sums: 2,4,6, 20,22,24, 38,40,42
    # 2,4 share 2. Bad.
    
    # Let me find one properly. Use a fast search with aggressive pruning.
    # Key insight: fix A to have small spread, and use bitmask-based pruning.
    
    def fast_find_B(A, max_b, deadline):
        """Find B of size 3 from [1..max_b] such that A+B has 9 distinct pairwise-coprime sums."""
        a1, a2, a3 = A
        # Precompute for each b in [1..max_b]: the 3 sums
        # And a bitmask of primes used
        # Pruning: for b to be viable, the 3 sums must be pairwise coprime AND
        # their prime masks must not conflict with each other.
        
        viable_b = []
        for b in range(1, max_b + 1):
            s = [a1 + b, a2 + b, a3 + b]
            if len(set(s)) != 3:
                continue
            if not is_pairwise_coprime(s):
                continue
            viable_b.append(b)
        
        if len(viable_b) < 3:
            return None
        
        # Now pick 3 from viable_b such that all 9 sums are pairwise coprime
        # Use backtracking with strong pruning
        n = len(viable_b)
        # Sort viable_b to try "good" ones first? 
        # Heuristic: prefer b where the sums are prime or have large prime factors
        viable_b.sort(key=lambda b: min(a1+b, a2+b, a3+b))
        
        for i in range(n):
            if time.time() > deadline:
                return None
            b1 = viable_b[i]
            sums1 = [a1 + b1, a2 + b1, a3 + b1]
            mask1 = pf_mask[sums1[0]] | pf_mask[sums1[1]] | pf_mask[sums1[2]]
            
            for j in range(i + 1, n):
                b2 = viable_b[j]
                sums2 = [a1 + b2, a2 + b2, a3 + b2]
                mask2 = pf_mask[sums2[0]] | pf_mask[sums2[1]] | pf_mask[sums2[2]]
                
                # Check no prime conflict between sums1 and sums2
                if mask1 & mask2:
                    continue
                
                for k in range(j + 1, n):
                    b3 = viable_b[k]
                    sums3 = [a1 + b3, a2 + b3, a3 + b3]
                    mask3 = pf_mask[sums3[0]] | pf_mask[sums3[1]] | pf_mask[sums3[2]]
                    
                    if mask1 & mask3 or mask2 & mask3:
                        continue
                    
                    # All 9 sums are pairwise coprime and distinct (since b's are distinct and A is distinct)
                    return [b1, b2, b3]
        
        return None

    # Search strategy: iterate A from small values, use fast_find_B
    # Focus on A with small elements to keep sums small (more likely to be coprime)
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = check(best_A, best_B)
    
    if best_score == 9:
        return [best_A, best_B]
    
    # Try A sets with 1 in them (1+b is often prime or has large factors)
    # Also try A with small spread
    a_candidates = list(range(1, 41))
    
    # Generate A combinations, prioritizing those with 1
    a_combos = list(combinations(a_candidates, 3))
    # Sort by smallest element, then sum, to try "good" structures first
    a_combos.sort(key=lambda x: (min(x), sum(x)))
    
    for limit_b in [30, 50, 80, 128]:
        if time.time() > deadline:
            break
        
        for A in a_combos:
            if time.time() > deadline:
                break
            A = list(A)
            
            # Quick prune: if A elements have common structure that causes issues
            B = fast_find_B(A, limit_b, deadline)
            if B:
                score = check(A, B)
                if score > best_score:
                    best_score = score
                    best_A = A
                    best_B = B
                    if best_score == 9:
                        return [best_A, best_B]
    
    # If still not found, try randomized search
    if best_score < 9 and time.time() < deadline - 5:
        for _ in range(5000):
            if time.time() > deadline:
                break
            A = random.sample(range(1, 61), 3)
            A.sort()
            B = fast_find_B(A, 80, deadline)
            if B:
                score = check(A, B)
                if score > best_score:
                    best_score = score
                    best_A = A
                    best_B = B
                    if best_score == 9:
                        return [best_A, best_B]
    
    return [best_A, best_B]

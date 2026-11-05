"""Rewritten program: bipartite/graph search with prime-mask pruning for pairwise-coprime sumsets."""
import time
from itertools import combinations
from math import gcd

def _build_prime_masks():
    """Build prime factor bitmask for each integer 1..256."""
    MAX_VAL = 256
    # Find all primes up to 256
    primes = []
    is_prime = [True] * (MAX_VAL + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, MAX_VAL + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, MAX_VAL + 1, i):
                is_prime[j] = False
    
    # Map each prime to a bit index
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    
    # Build mask for each number
    masks = [0] * (MAX_VAL + 1)
    for n in range(2, MAX_VAL + 1):
        m = n
        mask = 0
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                mask |= (1 << prime_to_bit[p])
                while m % p == 0:
                    m //= p
        if m > 1:
            mask |= (1 << prime_to_bit[m])
        masks[n] = mask
    return masks, primes

def _pairwise_coprime(values):
    """Check if all distinct values are pairwise coprime."""
    distinct = list(set(values))
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            if gcd(distinct[i], distinct[j]) > 1:
                return False
    return True

def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    masks, primes = _build_prime_masks()
    
    # Odd and even numbers in [1, 128]
    odds = [i for i in range(1, 129) if i % 2 == 1]
    evens = [i for i in range(1, 129) if i % 2 == 0]
    
    best_count = 0
    best_result = [[1, 7, 13], [4, 10, 16]]  # fallback
    
    # Strategy: A all odd, B all even (or vice versa)
    # Try both orientations
    
    for orientation in [0, 1]:
        if time.time() > deadline - 5:
            break
        
        if orientation == 0:
            set1_pool = odds   # A
            set2_pool = evens  # B
        else:
            set1_pool = evens  # A
            set2_pool = odds   # B
        
        # Precompute masks for pools
        pool1_masks = [(x, masks[x]) for x in set1_pool]
        pool2_masks = [(x, masks[x]) for x in set2_pool]
        
        # Try all combinations of 3 from set1_pool
        # This is C(64,3) = 41664 for odds, C(64,3) for evens
        # For each, try all combinations of 3 from set2_pool
        
        # Optimize: for a fixed A, find best B
        # Use greedy/heuristic: prioritize B values that create unique sums with few shared factors
        
        n1 = len(set1_pool)
        n2 = len(set2_pool)
        
        # For efficiency, iterate over A combinations and for each, search B
        # But C(64,3)^2 is too large. Use pruning.
        
        # Better approach: iterate over A triples, then for each, find B triples
        # that maximize distinct coprime sums
        
        # Precompute: for each pair (a, b), the sum and its mask
        # sum_mask[a_idx][b_idx] = (a+b, masks[a+b])
        
        # Actually, let's be smarter. For a fixed A = {a1, a2, a3},
        # the sums are a_i + b_j for i,j in {0,1,2}.
        # We need all distinct sums to be pairwise coprime.
        
        # For a candidate B = {b1, b2, b3}, the 9 sums are determined.
        # We can check validity quickly using masks.
        
        # Let's iterate over A triples, and for each, do a guided search over B.
        
        a_combos = list(combinations(range(n1), 3))
        
        for ai1, ai2, ai3 in a_combos:
            if time.time() > deadline - 3:
                break
            
            a1, a1m = pool1_masks[ai1]
            a2, a2m = pool1_masks[ai2]
            a3, a3m = pool1_masks[ai3]
            
            # For each candidate B triple, check validity
            # To speed up, precompute for each b the sums with a1,a2,a3
            # and their masks
            
            # For a given b, the three sums are a1+b, a2+b, a3+b
            # Their masks are masks[a1+b], masks[a2+b], masks[a3+b]
            
            # For B = {b1, b2, b3}, all 9 sums must have pairwise disjoint prime factors
            # (among distinct values)
            
            # Quick filter: for each b, compute the "conflict mask" with other b's
            # This is complex, so let's just iterate but with early termination
            
            # Heuristic: sort B candidates by how "prime-rich" they are (fewer small factors)
            # Actually, let's just iterate over B triples but with a cap
            
            b_combos = list(combinations(range(n2), 3))
            
            # To limit search, try a subset first
            # Score B candidates: prefer those where a_i + b has few prime factors
            b_scores = []
            for bi in range(n2):
                b_val, b_mask = pool2_masks[bi]
                # Count prime factors in the three sums
                s1 = a1 + b_val
                s2 = a2 + b_val
                s3 = a3 + b_val
                # Prefer sums that are prime or have few factors
                score = bin(masks[s1] | masks[s2] | masks[s3]).count('1')
                b_scores.append((score, bi))
            
            b_scores.sort()
            # Take top 200 B candidates
            top_b = [bi for _, bi in b_scores[:200]]
            
            # Now try triples from top_b
            top_b_combos = list(combinations(top_b, 3))
            
            for bi1, bi2, bi3 in top_b_combos:
                b1, b1m = pool2_masks[bi1]
                b2, b2m = pool2_masks[bi2]
                b3, b3m = pool2_masks[bi3]
                
                # Compute all 9 sums
                sums = [
                    a1 + b1, a1 + b2, a1 + b3,
                    a2 + b1, a2 + b2, a2 + b3,
                    a3 + b1, a3 + b2, a3 + b3
                ]
                
                distinct_sums = list(set(sums))
                count = len(distinct_sums)
                
                if count <= best_count:
                    continue
                
                # Check pairwise coprime
                valid = True
                for i in range(len(distinct_sums)):
                    for j in range(i + 1, len(distinct_sums)):
                        if gcd(distinct_sums[i], distinct_sums[j]) > 1:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    best_count = count
                    if orientation == 0:
                        best_result = [[a1, a2, a3], [b1, b2, b3]]
                    else:
                        best_result = [[b1, b2, b3], [a1, a2, a3]]
        
        if best_count >= 9:
            break  # Can't do better than 9
    
    return best_result

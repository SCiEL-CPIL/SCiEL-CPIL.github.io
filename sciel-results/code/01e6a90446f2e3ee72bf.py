"""Improved program: backtracking search for max-cardinality pairwise-coprime sumsets.

Key structural insight (parity):
  Pairwise-coprime => at most ONE even number in the whole sumset.
  For a 3x3 additive grid a_i + b_j, the number of even sums is
      k*m + (3-k)*(3-m)   (k=#even in A, m=#even in B)
  This equals 1 for NO (k,m) in [0,3]^2, so we must have ZERO even sums.
  That forces (A all odd & B all even) or (A all even & B all odd).
  Hence every sum is ODD and factor 2 never appears; we only need to manage
  odd prime factors.

We use backtracking with prime-mask pruning to efficiently find solutions
with 9 distinct pairwise-coprime sums.
"""

import time
from math import gcd
from itertools import combinations

DEADLINE = 85  # seconds; safety margin before 90s kill

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def construct():
    start_time = time.time()
    
    # Generate odd and even numbers in [1, 128]
    odds = [i for i in range(1, 129) if i % 2 == 1]
    evens = [i for i in range(1, 129) if i % 2 == 0]
    
    # Precompute prime factors for numbers up to 256 using bitmask
    max_sum = 256
    # Map primes to bit positions
    primes = [p for p in range(2, max_sum + 1) if _is_prime(p)]
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    
    # prime_mask[n] = bitmask of prime factors of n
    prime_mask = [0] * (max_sum + 1)
    for p_idx, p in enumerate(primes):
        bit = 1 << p_idx
        for multiple in range(p, max_sum + 1, p):
            prime_mask[multiple] |= bit
    
    def _check_pairwise_coprime(sums):
        """Check if all distinct sums are pairwise coprime using bitmask."""
        distinct_sums = list(set(sums))
        n = len(distinct_sums)
        for i in range(n):
            for j in range(i + 1, n):
                if prime_mask[distinct_sums[i]] & prime_mask[distinct_sums[j]]:
                    return False
        return True
    
    best_count = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Strategy: Use backtracking to build A and B simultaneously
    # For efficiency, we'll try to find 9 distinct sums first
    
    # Precompute: for each odd a, list of (even b, sum=a+b, mask=prime_mask[a+b])
    # This allows quick checking
    
    # Better approach: iterate over A triples, and for each, use backtracking
    # to find the best B triple
    
    def _find_best_B_for_A(A, deadline):
        """Find best B triple (all even) for given A triple (all odd)."""
        best_b_count = 0
        best_b = None
        
        # For each even b, compute the 3 sums a+b for a in A
        # and check if they can be part of a pairwise coprime set
        
        # Backtracking: pick 3 evens from evens list
        # Prune: if any two sums share a prime factor, reject
        
        evens_list = evens  # all even numbers in [1,128]
        
        # Precompute for each even b: the 3 sums and their masks
        b_sums = {}
        for b in evens_list:
            sums = tuple(a + b for a in A)
            masks = tuple(prime_mask[s] for s in sums)
            b_sums[b] = (sums, masks)
        
        # Now backtracking over choosing 3 evens
        # We'll use a simple recursive approach with pruning
        
        def backtrack(chosen, idx):
            nonlocal best_b_count, best_b
            if time.time() > deadline:
                return
            if len(chosen) == 3:
                # Check all 9 sums are pairwise coprime
                all_sums = []
                for b in chosen:
                    all_sums.extend(b_sums[b][0])
                if _check_pairwise_coprime(all_sums):
                    dc = len(set(all_sums))
                    if dc > best_b_count:
                        best_b_count = dc
                        best_b = list(chosen)
                        if dc == 9:
                            return
                return
            
            # Try adding each even from idx onwards
            for i in range(idx, len(evens_list)):
                b = evens_list[i]
                sums, masks = b_sums[b]
                
                # Quick check: new sums must be coprime with all existing sums
                valid = True
                for existing_b in chosen:
                    existing_sums, existing_masks = b_sums[existing_b]
                    for s1, m1 in zip(sums, masks):
                        for s2, m2 in zip(existing_sums, existing_masks):
                            if m1 & m2:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    backtrack(chosen + [b], i + 1)
        
        backtrack([], 0)
        return best_b_count, best_b
    
    # Iterate over A triples (all odd)
    for A in combinations(odds, 3):
        if time.time() - start_time > DEADLINE:
            break
        A = list(A)
        
        b_count, b = _find_best_B_for_A(A, DEADLINE)
        if b is not None and b_count > best_count:
            best_count = b_count
            best_A = A
            best_B = b
            if best_count == 9:
                return [best_A, best_B]
    
    # Also try A even, B odd
    if time.time() - start_time < DEADLINE:
        for A in combinations(evens, 3):
            if time.time() - start_time > DEADLINE:
                break
            A = list(A)
            
            # For A even, B odd, the sums are odd
            # Similar backtracking but with odds for B
            best_b_count = 0
            best_b = None
            
            odds_list = odds
            b_sums = {}
            for b in odds_list:
                sums = tuple(a + b for a in A)
                masks = tuple(prime_mask[s] for s in sums)
                b_sums[b] = (sums, masks)
            
            def backtrack2(chosen, idx):
                nonlocal best_b_count, best_b
                if time.time() > DEADLINE:
                    return
                if len(chosen) == 3:
                    all_sums = []
                    for b in chosen:
                        all_sums.extend(b_sums[b][0])
                    if _check_pairwise_coprime(all_sums):
                        dc = len(set(all_sums))
                        if dc > best_b_count:
                            best_b_count = dc
                            best_b = list(chosen)
                            if dc == 9:
                                return
                    return
                
                for i in range(idx, len(odds_list)):
                    b = odds_list[i]
                    sums, masks = b_sums[b]
                    
                    valid = True
                    for existing_b in chosen:
                        existing_sums, existing_masks = b_sums[existing_b]
                        for s1, m1 in zip(sums, masks):
                            for s2, m2 in zip(existing_sums, existing_masks):
                                if m1 & m2:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        backtrack2(chosen + [b], i + 1)
            
            backtrack2([], 0)
            
            if best_b is not None and best_b_count > best_count:
                best_count = best_b_count
                best_A = A
                best_B = best_b
                if best_count == 9:
                    return [best_A, best_B]
    
    return [best_A, best_B]

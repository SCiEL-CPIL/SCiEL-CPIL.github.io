"""Rewritten program: systematic search for 9 distinct pairwise-coprime sums.

Key insights:
1. Parity constraint: For 3x3 sumset, pairwise coprime requires all sums to be
   odd (since at most one even number can appear, and the count formula
   k*m + (3-k)*(3-m) never equals 1 for k,m in [0,3]).
   This forces (A all odd & B all even) or (A all even & B all odd).
   
2. We want 9 distinct sums, all pairwise coprime (ideally all prime).

3. Use bitmask of prime factors for quick coprime checking.

Strategy:
- Precompute prime factor masks for all numbers up to 256.
- Search over A (3 distinct values) and B (3 distinct values) with the parity
  constraint, checking if all 9 sums are distinct and pairwise coprime.
- Use a deadline guard to ensure we finish within time limits.
"""
import time
from itertools import combinations

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 80.0  # 80 seconds, leaving 10s safety margin

def _prime_factor_mask(n):
    """Return a bitmask of prime factors of n (odd primes only, since we deal with odd numbers)."""
    mask = 0
    if n < 2:
        return mask
    # For our purposes, we only care about odd primes (sums are odd)
    d = 3
    while d * d <= n:
        if n % d == 0:
            mask |= (1 << d)
            while n % d == 0:
                n //= d
        d += 2
    if n > 2:
        mask |= (1 << n)
    return mask

# Precompute masks for all odd numbers up to 256
_MASKS = {}
for n in range(3, 257, 2):
    _MASKS[n] = _prime_factor_mask(n)

def _pairwise_coprime(sums):
    """Check if all distinct values in sums are pairwise coprime."""
    distinct = set(sums)
    masks = [_MASKS.get(s, 0) for s in distinct]
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if masks[i] & masks[j]:
                return False
    return True

def _valid_AB(A, B):
    """Check if A and B produce 9 distinct pairwise-coprime sums."""
    sums = [a + b for a in A for b in B]
    if len(set(sums)) != 9:
        return False
    return _pairwise_coprime(sums)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Parity constraint: A all odd & B all even, OR A all even & B all odd
    # Case 1: A odd, B even
    # Case 2: A even, B odd
    
    best = None
    best_count = 0
    
    odds = list(range(1, 129, 2))   # 1, 3, 5, ..., 127
    evens = list(range(2, 129, 2))  # 2, 4, 6, ..., 128
    
    # We'll search systematically. To maximize distinct sums, we want
    # the sums to spread out. Let's try combinations.
    
    # For efficiency, we can limit the search space.
    # A and B each have 3 distinct values from [1,128].
    
    # Try Case 1: A all odd, B all even
    # And Case 2: A all even, B all odd
    
    # To keep runtime manageable, we'll use a randomized + systematic approach.
    # First, try some structured patterns, then random search.
    
    import random
    
    def check_and_update(A, B):
        nonlocal best, best_count
        if time.time() - _START_TIME > TIME_LIMIT:
            return False
        sums = [a + b for a in A for b in B]
        distinct = set(sums)
        if len(distinct) > best_count and _pairwise_coprime(sums):
            best_count = len(distinct)
            best = (list(A), list(B))
            if best_count == 9:
                return True  # Can't do better
        return False
    
    # Structured search: try to find 9 distinct sums
    # All sums must be odd, so A+B is odd => one of A,B is odd, other even.
    
    # Let's try a focused search. Since we need 9 distinct odd sums in [3, 256],
    # and they must be pairwise coprime, let's prefer primes.
    
    # Generate candidate primes in [3, 256]
    primes = [n for n in range(3, 257, 2) if _MASKS[n] != 0 and _MASKS[n] == (1 << n)]
    # Actually, a number is prime iff its mask is just itself (for odd primes)
    # Let's recompute: a number n is prime if its only prime factor is n
    primes = [n for n in range(3, 257, 2) if _MASKS[n] == (1 << n)]
    
    # Try random search with bias toward primes
    random.seed(42)
    
    found_nine = False
    iterations = 0
    
    while time.time() - _START_TIME < TIME_LIMIT:
        iterations += 1
        
        # Randomly pick A and B with correct parity
        if iterations % 2 == 0:
            # A odd, B even
            A = random.sample(odds, 3)
            B = random.sample(evens, 3)
        else:
            # A even, B odd
            A = random.sample(evens, 3)
            B = random.sample(odds, 3)
        
        if check_and_update(A, B):
            found_nine = True
            break
    
    # If we found 9, great. Otherwise, do a more systematic search.
    if not found_nine and time.time() - _START_TIME < TIME_LIMIT:
        # Systematic search over smaller ranges to find good candidates
        # Try all combinations of 3 odds and 3 evens from a smaller set
        
        # Focus on values that produce sums in a good range
        # Sums range from 3 to 256, all odd
        # Let's try A from odds in [1, 50] and B from evens in [2, 50]
        small_odds = list(range(1, 51, 2))
        small_evens = list(range(2, 51, 2))
        
        for A in combinations(small_odds, 3):
            if time.time() - _START_TIME > TIME_LIMIT:
                break
            for B in combinations(small_evens, 3):
                if time.time() - _START_TIME > TIME_LIMIT:
                    break
                if check_and_update(A, B):
                    found_nine = True
                    break
            if found_nine:
                break
        
        if not found_nine:
            # Try the other parity
            for A in combinations(small_evens, 3):
                if time.time() - _START_TIME > TIME_LIMIT:
                    break
                for B in combinations(small_odds, 3):
                    if time.time() - _START_TIME > TIME_LIMIT:
                        break
                    if check_and_update(A, B):
                        found_nine = True
                        break
                if found_nine:
                    break
    
    # If we still haven't found 9, try a broader systematic search
    if not found_nine and time.time() - _START_TIME < TIME_LIMIT:
        # Try all combinations from a medium range
        med_odds = list(range(1, 81, 2))
        med_evens = list(range(2, 81, 2))
        
        # Use a smarter approach: fix A, then search B
        for A in combinations(med_odds, 3):
            if time.time() - _START_TIME > TIME_LIMIT:
                break
            for B in combinations(med_evens, 3):
                if time.time() - _START_TIME > TIME_LIMIT:
                    break
                if check_and_update(A, B):
                    found_nine = True
                    break
            if found_nine:
                break
    
    # Fallback: return a known good solution or the best found
    if best is not None:
        return best
    
    # Default fallback (should not happen)
    return [[1, 7, 13], [4, 10, 16]]

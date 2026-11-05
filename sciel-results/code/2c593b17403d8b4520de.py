"""Rewritten program: exhaustive search for max-cardinality pairwise-coprime sumsets.

Key structural insight (parity):
  Pairwise-coprime => at most ONE even number in the whole sumset.
  For a 3x3 additive grid a_i + b_j, the number of even sums is
      k*m + (3-k)*(3-m)   (k=#even in A, m=#even in B)
  This equals 1 for NO (k,m) in [0,3]^2, so we must have ZERO even sums.
  That forces (A all odd & B all even) or (A all even & B all odd).
  Hence every sum is ODD and factor 2 never appears; we only need to manage
  odd prime factors.

We search over all valid (A,B) pairs where A and B each have 3 distinct integers
in [1,128], all sums are odd (so one set is all-odd, the other all-even), and
all distinct sums are pairwise coprime. We maximize the number of distinct sums.
"""

import time
from math import gcd
from itertools import combinations, product

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

def _pairwise_coprime(values):
    """Check if all distinct values in the list are pairwise coprime."""
    vals = list(set(values))
    n = len(vals)
    for i in range(n):
        for j in range(i+1, n):
            if gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _valid_pair(A, B):
    """Check if A+B has all distinct values pairwise coprime."""
    sums = [a + b for a in A for b in B]
    return _pairwise_coprime(sums)

def _distinct_count(A, B):
    return len(set(a + b for a in A for b in B))

def construct():
    start_time = time.time()
    
    # Generate odd and even numbers in [1, 128]
    odds = [i for i in range(1, 129) if i % 2 == 1]
    evens = [i for i in range(1, 129) if i % 2 == 0]
    
    # We need A all odd, B all even OR A all even, B all odd
    # Both give the same structure; let's try A odd, B even first
    
    best_count = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Precompute: for each pair of odd numbers, check which even numbers work
    # Actually, let's do a smarter search.
    
    # Strategy: iterate over combinations of 3 odds for A, and for each,
    # find the best B (3 evens) that maximizes distinct coprime sums.
    
    # To speed up, we can precompute for each odd a and even b whether
    # gcd(a+b, ...) works, but the pairwise condition is global.
    
    # Let's use a pruning approach: for a given A, check B candidates.
    
    # Number of 3-combinations of 64 odds = C(64,3) = 41664
    # Number of 3-combinations of 64 evens = C(64,3) = 41664
    # Total pairs to check: 41664 * 41664 = ~1.7e9 - too many.
    
    # Better: for each A (41664 options), find best B.
    # For a fixed A, the sums a_i + b_j must be pairwise coprime (distinct ones).
    # There are 9 sums, but some may repeat. The distinct ones must be coprime.
    
    # Let's try a different approach: since we want to maximize distinct sums,
    # and the maximum possible is 9, let's search for solutions with 9 distinct
    # sums first, then 8, etc.
    
    # Actually, let's just do a smarter exhaustive search with pruning.
    
    # For each A (odd triple), collect all valid B candidates.
    # A sum a+b is "bad" if it shares a factor with another sum.
    
    # Let's precompute prime factors for numbers up to 256.
    max_sum = 256
    prime_factors = [set() for _ in range(max_sum + 1)]
    for p in range(2, max_sum + 1):
        if _is_prime(p):
            for multiple in range(p, max_sum + 1, p):
                prime_factors[multiple].add(p)
    
    def _sums_pairwise_coprime(A, B):
        """Check if all distinct sums in A+B are pairwise coprime using prime factors."""
        sums = [a + b for a in A for b in B]
        distinct_sums = list(set(sums))
        n = len(distinct_sums)
        for i in range(n):
            for j in range(i + 1, n):
                if prime_factors[distinct_sums[i]] & prime_factors[distinct_sums[j]]:
                    return False
        return True
    
    # Search over A triples (all odd)
    # To make this faster, let's limit the search space by focusing on
    # "good" candidates. 
    
    # Actually, let's just iterate and use the deadline.
    count = 0
    for A in combinations(odds, 3):
        if time.time() - start_time > DEADLINE:
            break
        A = list(A)
        
        # For this A, try to find the best B
        # We'll iterate over B triples (all even)
        for B in combinations(evens, 3):
            if time.time() - start_time > DEADLINE:
                break
            B = list(B)
            
            if _sums_pairwise_coprime(A, B):
                dc = _distinct_count(A, B)
                if dc > best_count:
                    best_count = dc
                    best_A = A
                    best_B = B
                    if best_count == 9:
                        return [best_A, best_B]
    
    # Also try A even, B odd (should give same results by symmetry)
    # But let's check if we have time
    if time.time() - start_time < DEADLINE:
        for A in combinations(evens, 3):
            if time.time() - start_time > DEADLINE:
                break
            A = list(A)
            for B in combinations(odds, 3):
                if time.time() - start_time > DEADLINE:
                    break
                B = list(B)
                if _sums_pairwise_coprime(A, B):
                    dc = _distinct_count(A, B)
                    if dc > best_count:
                        best_count = dc
                        best_A = A
                        best_B = B
                        if best_count == 9:
                            return [best_A, best_B]
    
    return [best_A, best_B]

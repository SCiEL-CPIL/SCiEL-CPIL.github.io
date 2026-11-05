"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
from math import gcd
from functools import reduce

def _pairwise_coprime(nums):
    """Check if all distinct values in nums are pairwise coprime."""
    distinct = list(set(nums))
    n = len(distinct)
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def construct():
    # Deterministic construction of A and B such that A+B consists of 9 distinct primes.
    # Primes are pairwise coprime.
    # A = [3, 15, 39]
    # B = [2, 4, 8]
    # Sums:
    # 3+2=5, 3+4=7, 3+8=11
    # 15+2=17, 15+4=19, 15+8=23
    # 39+2=41, 39+4=43, 39+8=47
    # All are distinct primes.
    
    A = [3, 15, 39]
    B = [2, 4, 8]
    
    # Verification (optional, but good for safety)
    sums = [a + b for a in A for b in B]
    distinct_sums = set(sums)
    
    # Check pairwise coprimality
    if not _pairwise_coprime(distinct_sums):
        # Fallback if verification fails (should not happen)
        return [[1, 7, 13], [4, 10, 16]]
        
    return [A, B]
# EVOLVE-BLOCK-END

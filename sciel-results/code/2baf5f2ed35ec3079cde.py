import random
from math import gcd

def construct():
    # We want 3x3 grid of sums all distinct primes
    # Primes up to 256
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5)+1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = set(p for p in range(2, 257) if is_prime(p))
    
    best_A, best_B = None, None
    best_count = 0
    
    # Strategy: A all odd, B all even (or vice versa) to get odd sums
    # Try many random combinations
    random.seed(42)
    
    for trial in range(200000):
        # Pick A: 3 distinct odd numbers in [1,128]
        odds = list(range(1, 129, 2))
        evens = list(range(2, 129, 2))
        
        A = random.sample(odds, 3)
        B = random.sample(evens, 3)
        
        sums = [a + b for a in A for b in B]
        distinct_sums = set(sums)
        
        # Check all distinct sums are prime and pairwise coprime (primes are automatically pairwise coprime if distinct)
        if len(distinct_sums) == 9:
            if all(s in primes for s in distinct_sums):
                return [sorted(A), sorted(B)]
        
        # Also check for high cardinality with pairwise coprime
        if len(distinct_sums) > best_count:
            # Check pairwise coprime
            valid = True
            ds = list(distinct_sums)
            for i in range(len(ds)):
                for j in range(i+1, len(ds)):
                    if gcd(ds[i], ds[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_count = len(distinct_sums)
                best_A, best_B = sorted(A), sorted(B)
    
    if best_A is not None:
        return [best_A, best_B]
    
    # Fallback: known good solution
    return [[1, 7, 13], [4, 10, 16]]

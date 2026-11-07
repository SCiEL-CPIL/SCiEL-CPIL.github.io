"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start = time.time()
    deadline = start + 80.0  # 80s safety margin within 90s limit
    
    def is_prime(n):
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
    
    def prime_factors(n):
        """Return set of distinct prime factors of n."""
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    # Precompute prime factors for all numbers 2..256
    pf = [set() for _ in range(257)]
    for n in range(2, 257):
        pf[n] = prime_factors(n)
    
    # Start with all primes
    primes = [a for a in range(2, 257) if is_prime(a)]
    result = list(primes)
    result_set = set(result)
    
    # Candidate composites to try adding
    composites = [a for a in range(2, 257) if not is_prime(a)]
    
    # Sort composites: prefer those with fewer/smaller prime factors (less likely to cause violations)
    def composite_score(x):
        # Prefer composites whose prime factors are large (less likely to divide products of others)
        # Actually, we want composites that are "hard to divide" - i.e., have large prime factors
        # A composite a divides b*c iff all prime factors of a appear in b*c
        # So composites with large prime factors are safer to include
        factors = pf[x]
        # Score: sum of prime factors (higher = safer, less likely to divide b*c)
        return -sum(factors)
    
    composites.sort(key=composite_score)
    
    def can_add(x, current_set, current_list):
        """Check if x can be added to current_list without violating constraints."""
        if x in current_set:
            return False
        
        n = len(current_list)
        # Check: for all pairs (a,b) in current_list, a != b:
        #   1. x does not divide a*b
        #   2. a does not divide x*b
        #   3. b does not divide x*a
        
        for i in range(n):
            a = current_list[i]
            for j in range(n):
                if i == j:
                    continue
                b = current_list[j]
                # Check x ∤ a*b
                if (a * b) % x == 0:
                    return False
                # Check a ∤ x*b
                if (x * b) % a == 0:
                    return False
                # Check b ∤ x*a
                if (x * a) % b == 0:
                    return False
        
        return True
    
    # Greedily add composites
    for x in composites:
        if time.time() > deadline - 5.0:
            break
        if can_add(x, result_set, result):
            result.append(x)
            result_set.add(x)
    
    # Verify final result
    n = len(result)
    for i in range(n):
        a = result[i]
        for j in range(n):
            if j == i:
                continue
            b = result[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = result[k]
                if (b * c) % a == 0:
                    # Violation found - shouldn't happen
                    return primes
    
    return result
# EVOLVE-BLOCK-END

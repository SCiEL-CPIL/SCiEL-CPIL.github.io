import time
import math
import itertools
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # First few primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
    def tau_perp(n):
        """Count adjacent coprime pairs in sorted divisors of n."""
        if n <= 1:
            return 0
        divs = sorted(set(_divisors(n)))
        count = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                count += 1
        return count
    
    def _divisors(n):
        """Generate all divisors of n."""
        if n == 1:
            return [1]
        divs = set([1])
        p = 2
        while p * p <= n:
            if n % p == 0:
                power = 1
                while n % p == 0:
                    n //= p
                    power *= p
                    for d in list(divs):
                        divs.add(d * power)
            p += 1 if p == 2 else 2
        if n > 1:
            for d in list(divs):
                divs.add(d * n)
        return list(divs)
    
    def score_fast(n):
        """Faster tau_perp using the divisor structure."""
        if n <= 1:
            return 0
        # Get prime factorization
        factors = []
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors.append((p, exp))
            p += 1 if p == 2 else 2
        if temp > 1:
            factors.append((temp, 1))
        
        # Generate divisors and count coprime adjacent pairs
        divs = [1]
        for p, e in factors:
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        
        divs = sorted(set(divs))
        count = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                count += 1
        return count
    
    # Start with primorial of first 15 primes as baseline
    best_n = 1
    for p in primes[:15]:
        best_n *= p
    best_score = score_fast(best_n)
    
    # Local search: try adjusting exponents
    # Use greedy hill-climbing on exponent vectors
    def get_exponents(n, max_prime=61):
        """Get exponent vector for n with respect to given primes."""
        exps = {}
        for p in primes:
            if p > max_prime:
                break
            while n % p == 0:
                exps[p] = exps.get(p, 0) + 1
                n //= p
        if n > 1:
            exps[n] = exps.get(n, 0) + 1
        return exps
    
    def n_from_exps(exps):
        """Reconstruct n from exponent dict."""
        result = 1
        for p, e in exps.items():
            result *= p ** e
        return result
    
    # Hill climbing on exponent adjustments
    def local_search(n, max_time):
        current_n = n
        current_score = score_fast(current_n)
        improved = True
        while improved and time.time() < max_time:
            improved = False
            exps = get_exponents(current_n, max_prime=61)
            # Try increasing each exponent by 1
            for p in primes[:12]:
                if time.time() >= max_time:
                    break
                new_exps = dict(exps)
                new_exps[p] = new_exps.get(p, 0) + 1
                candidate = n_from_exps(new_exps)
                if candidate > 10**18:
                    continue
                s = score_fast(candidate)
                if s > current_score:
                    current_n = candidate
                    current_score = s
                    improved = True
            # Try decreasing each exponent by 1 (if > 0)
            for p in list(exps.keys()):
                if time.time() >= max_time:
                    break
                if exps[p] <= 1:
                    continue
                new_exps = dict(exps)
                new_exps[p] = new_exps[p] - 1
                candidate = n_from_exps(new_exps)
                if candidate < 2:
                    continue
                s = score_fast(candidate)
                if s > current_score:
                    current_n = candidate
                    current_score = s
                    improved = True
        return current_n, current_score
    
    # Run local search from baseline
    best_n, best_score = local_search(best_n, deadline - 5.0)
    
    # Try different starting points
    # Squarefree with more primes
    for k in range(8, 16):
        if time.time() > deadline - 5.0:
            break
        candidate = 1
        for p in primes[:k]:
            candidate *= p
            if candidate > 10**18:
                break
        if 2 <= candidate <= 10**18:
            s = score_fast(candidate)
            if s > best_score:
                best_n = candidate
                best_score = s
    
    # Try numbers with mixed exponents
    # n = 2^a * 3^b * 5^c * ... 
    # Search over small exponent combinations
    if time.time() < deadline - 5.0:
        # Try some structured candidates
        candidates_to_try = []
        
        # Powers of small primes times primorial
        for a in range(1, 20):
            for b in range(1, 12):
                if time.time() > deadline - 5.0:
                    break
                base = (2 ** a) * (3 ** b)
                if base > 10**18:
                    break
                # Multiply by next primes
                candidate = base
                for p in primes[2:10]:
                    if candidate * p > 10**18:
                        break
                    candidate *= p
                if 2 <= candidate <= 10**18:
                    candidates_to_try.append(candidate)
        
        for c in candidates_to_try:
            if time.time() > deadline - 5.0:
                break
            s = score_fast(c)
            if s > best_score:
                best_n = c
                best_score = s
                # Do local search from here
                if time.time() < deadline - 5.0:
                    best_n, best_score = local_search(best_n, deadline - 5.0)
    
    # Final local search pass
    if time.time() < deadline - 2.0:
        best_n, best_score = local_search(best_n, deadline - 1.0)
    
    return best_n

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import isqrt

def _sieve_primes(n):
    """Return list of primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def _factorize(n, primes):
    """Return dict of prime->exponent for n."""
    factors = {}
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            exp = 0
            while n % p == 0:
                n //= p
                exp += 1
            factors[p] = exp
    if n > 1:
        factors[n] = 1
    return factors

def construct():
    n = 256
    primes = _sieve_primes(n)
    
    # Compute prime factor dicts for each number in 2..n
    prime_factors = {}
    for i in range(2, n + 1):
        prime_factors[i] = _factorize(i, primes)
    
    def is_valid_set(S):
        """Check if S is a valid set."""
        S_list = list(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                for k in range(j + 1, m):
                    c = S_list[k]
                    if (b * c) % a == 0:
                        return False
        return True

    def can_add(a, S_list):
        """Check if a can be added to list S_list without violating the condition."""
        m = len(S_list)
        # Check: a ∤ b*c for all distinct b,c in S
        for i in range(m):
            for j in range(i + 1, m):
                b = S_list[i]
                c = S_list[j]
                if (b * c) % a == 0:
                    return False
        # Check: for each b in S, b ∤ a*c for all c in S, c != b
        for i in range(m):
            b = S_list[i]
            for j in range(m):
                if j == i:
                    continue
                c = S_list[j]
                if (a * c) % b == 0:
                    return False
        return True

    # Strategy: Start with all primes, then try to swap primes for composites
    # to see if we can increase the count.
    S = [p for p in primes]
    
    # We want to maximize size.
    # Try adding composites. If a composite x conflicts with primes P(x), 
    # we can remove P(x) and add x.
    # Net change: -len(P(x)) + 1.
    # This is negative if len(P(x)) > 1.
    # So simple 1-for-k swaps reduce size.
    # We need to find groups of composites that allow net gain.
    
    # Let's try a greedy construction with a specific ordering that might find a larger set
    # if one exists.
    
    # Heuristic: Numbers with large prime factors are less likely to divide products of others.
    # Also, primes are very safe.
    
    # Let's try multiple greedy orders and pick the best.
    
    def greedy_solve(order_key):
        candidates = list(range(2, n + 1))
        candidates.sort(key=order_key)
        S_greedy = []
        for a in candidates:
            if can_add(a, S_greedy):
                S_greedy.append(a)
        return S_greedy
    
    # Order 1: Primes first (descending), then composites by largest prime factor
    def key1(x):
        pf = prime_factors[x]
        is_prime = len(pf) == 1
        largest_pf = max(pf.keys()) if pf else 0
        return (0 if is_prime else 1, -largest_pf, -x)
        
    # Order 2: Largest prime factor descending
    def key2(x):
        pf = prime_factors[x]
        largest_pf = max(pf.keys()) if pf else 0
        return (-largest_pf, -x)
        
    # Order 3: Descending value
    def key3(x):
        return -x
        
    # Order 4: Ascending value (sometimes helps?)
    def key4(x):
        return x
        
    S1 = greedy_solve(key1)
    S2 = greedy_solve(key2)
    S3 = greedy_solve(key3)
    S4 = greedy_solve(key4)
    
    best_S = max([S1, S2, S3, S4], key=len)
    
    # Try local search: try to add any excluded number, removing conflicting ones if beneficial
    # (This is complex, so we stick to greedy for now, but ensure we return the best found)
    
    return list(best_S)
# EVOLVE-BLOCK-END

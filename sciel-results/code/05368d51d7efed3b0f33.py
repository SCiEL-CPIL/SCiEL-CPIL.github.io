"""Program for Large integer sets with no two-other-product divisibility."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'n': 256}
DEADLINE_MARGIN = 15.0  # seconds before the 90s wall-clock limit


def _sieve_primes(n):
    """Return list of primes <= n."""
    if n < 2:
        return []
    is_prime = bytearray(b'\x01') * (n + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(n) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start::step] = bytearray(len(is_prime[start::step]))
    return [i for i in range(2, n + 1) if is_prime[i]]


def _is_prime(n, primes):
    """Check if n is prime using the precomputed primes list."""
    if n < 2:
        return False
    if n in primes_set:
        return True
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            return False
    return True


def _check_validity(candidate_set, primes_set, primes):
    """Check if a set of integers satisfies the divisibility constraint."""
    n = len(candidate_set)
    # Build a set for O(1) membership
    cand_set = set(candidate_set)
    # For each pair (b, c), compute b*c and check if any a in candidate_set divides it
    # More efficiently: for each a in candidate_set, check if there exist b, c in candidate_set
    # with b != c, b != a, c != a, such that a | b*c
    # Equivalent: for each a, for each prime power p^k || a, check if for every prime power
    # dividing a, at least one of b or c has enough of that prime.
    
    # Simpler approach: for each a, factorize it. Then check all pairs (b,c).
    # But that's O(n^3) which for n~100 is 1 million, fine.
    
    # Actually let's do: for each a, check if there exist b != a, c != a, b != c such that a | b*c
    # We can iterate over b in candidate_set (b != a), and for each b, we need c != a, c != b
    # such that a/gcd(a,b) | c. Let g = gcd(a,b), then we need (a//g) | c.
    # So for each a, for each b != a, let need = a // gcd(a,b). Check if there exists
    # c in candidate_set, c != a, c != b, such that c % need == 0.
    
    from math import gcd
    for a in candidate_set:
        for b in candidate_set:
            if b == a:
                continue
            g = gcd(a, b)
            need = a // g
            if need == 1:
                # Any c != a, c != b works
                # Need at least one c != a, c != b in candidate_set
                if n >= 3:
                    return False
            else:
                # Check if there's a c in candidate_set with c % need == 0, c != a, c != b
                for c in candidate_set:
                    if c == a or c == b:
                        continue
                    if c % need == 0:
                        return False
    return True


def _greedy_construct(n):
    """Greedy construction: start with all numbers, iteratively remove the most conflicting."""
    # Start with all numbers 2..n
    candidates = list(range(2, n + 1))
    primes = _sieve_primes(n)
    primes_set = set(primes)
    
    # Use the primes + prime powers strategy as a strong initial set
    # Then try to add more numbers greedily
    
    # Step 1: Build initial set from primes and prime powers
    initial = set()
    for p in primes:
        initial.add(p)
        pk = p * p
        while pk <= n:
            initial.add(pk)
            pk *= p
            if pk > n:
                break
    
    # Try to add numbers from the remaining candidates
    remaining = [x for x in candidates if x not in initial]
    # Sort remaining by some heuristic: prefer numbers with large prime factors
    def _score(x):
        # Score by largest prime factor
        lp = x
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                while x % p == 0:
                    x //= p
        return max(x, 1)
    
    remaining.sort(key=_score, reverse=True)
    
    current_set = list(initial)
    current_set_set = set(current_set)
    
    for x in remaining:
        trial = current_set + [x]
        if _check_validity(trial, primes_set, primes):
            current_set = trial
            current_set_set.add(x)
    
    return current_set


def construct():
    """Construct a valid set of integers in 2..256 maximizing cardinality."""
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    n = P['n']
    primes = _sieve_primes(n)
    primes_set = set(primes)
    
    # Strategy 1: Primes + Prime Powers
    initial = set()
    for p in primes:
        initial.add(p)
        pk = p * p
        while pk <= n:
            initial.add(pk)
            pk *= p
            if pk > n:
                break
    
    # Convert to list
    result = list(initial)
    
    # Check validity of the initial set
    if not _check_validity(result, primes_set, primes):
        # Fallback: just primes
        result = list(primes)
    
    # Try to greedily add more numbers
    remaining = [x for x in range(2, n + 1) if x not in initial]
    
    def _largest_prime_factor(x):
        lp = x
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                while x % p == 0:
                    x //= p
        return max(x, 1)
    
    remaining.sort(key=_largest_prime_factor, reverse=True)
    
    for x in remaining:
        if time.time() > deadline:
            break
        trial = result + [x]
        if _check_validity(trial, primes_set, primes):
            result = trial
    
    # Ensure we return a list of Python ints
    result = [int(x) for x in result]
    
    return result


# EVOLVE-BLOCK-END

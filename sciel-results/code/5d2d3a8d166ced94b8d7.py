"""Rewritten solution: primes up to 256 via sieve, with a small greedy attempt to add composites if possible."""

# EVOLVE-BLOCK-START
import time

_DEADLINE = 60.0

def _sieve_primes(limit):
    """Return list of primes up to limit (inclusive)."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _valid_set(S):
    """Check if set S satisfies the condition: for every three distinct a,b,c in S, a does not divide b*c."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = S_list[i], S_list[j], S_list[k]
                if b * c % a == 0:
                    return False
    return True

def construct():
    start_time = time.time()
    
    limit = 256
    primes = _sieve_primes(limit)
    
    # Start with all primes
    S = set(primes)
    
    # Try to add composites one by one, greedily, checking validity
    composites = [x for x in range(2, limit + 1) if x not in S]
    
    # Sort composites by some heuristic - maybe by number of prime factors (fewer first?)
    # or just try all and see which can be added
    def can_add(x, current_S):
        """Check if x can be added to current_S while maintaining validity."""
        # For every pair (b, c) in current_S, check x does not divide b*c
        # And for every a in current_S, check a does not divide x * (other) for all other != a, x
        # Actually need to check all triples involving x
        current_list = list(current_S)
        for i in range(len(current_list)):
            for j in range(i + 1, len(current_list)):
                a, b = current_list[i], current_list[j]
                # Triple (x, a, b): check x does not divide a*b
                if (a * b) % x == 0:
                    return False
                # Triple (a, x, b): check a does not divide x*b
                if (x * b) % a == 0:
                    return False
                # Triple (b, x, a): check b does not divide x*a
                if (x * a) % b == 0:
                    return False
        return True
    
    # Try adding composites
    # Sort by value or some criterion
    for x in sorted(composites):
        if time.time() - start_time > _DEADLINE - 5:
            break
        if can_add(x, S):
            S.add(x)
    
    result = sorted(S)
    # Final validation
    if not _valid_set(S):
        # Fallback to primes only
        result = primes
    
    return result
# EVOLVE-BLOCK-END

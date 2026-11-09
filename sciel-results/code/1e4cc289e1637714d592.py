import time
from math import isqrt

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, isqrt(n) + 1, 2):
        if n % i == 0:
            return False
    return True

def _valid_triple(a, b, c):
    """Check that a does not divide b*c (a,b,c distinct)."""
    return (b * c) % a != 0

def _is_valid_set(S):
    """Check if set S satisfies the property."""
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = S_list[i], S_list[j], S_list[k]
                if (b * c) % a == 0:
                    return False
                if (a * c) % b == 0:
                    return False
                if (a * b) % c == 0:
                    return False
    return True

def _can_add(x, S_list):
    """Check if x can be added to S_list (as a set) maintaining validity."""
    n = len(S_list)
    # x as 'a': x ∤ b*c for all distinct b,c in S
    for i in range(n):
        for j in range(i+1, n):
            if (S_list[i] * S_list[j]) % x == 0:
                return False
    # x as 'b': a ∤ x*c for all distinct a,c in S (a≠c)
    for i in range(n):
        for j in range(i+1, n):
            if (x * S_list[j]) % S_list[i] == 0:
                return False
            if (x * S_list[i]) % S_list[j] == 0:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    all_nums = list(range(2, 257))
    primes = [x for x in all_nums if _is_prime(x)]
    
    # Start with primes
    S = list(primes)
    
    # Try adding composites
    composites = [x for x in all_nums if not _is_prime(x)]
    # Sort by some heuristic - maybe by number of divisors or just descending
    composites.sort(reverse=True)
    
    for x in composites:
        if time.time() > deadline:
            break
        if _can_add(x, S):
            S.append(x)
    
    # Local search: try to improve
    # ...
    
    return S

import math
import time
from collections import defaultdict

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n using trial division."""
    if n <= 0:
        return 0
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            exp = 0
            while temp % d == 0:
                temp //= d
                exp += 1
            # sigma(p^e) = (p^(e+1) - 1) / (p - 1)
            result *= (d ** (exp + 1) - 1) // (d - 1)
        d += 1
    if temp > 1:
        result *= (temp + 1)  # sigma(p) = p + 1
    return result

def _divisors(n):
    """Generate all divisors of n."""
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return divs

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 120.0  # Safety margin from 150s
    
    N = 5 * 10**6
    sigma = [0] * (N + 1)
    
    # Direct sieve for sigma
    sig = sigma
    n_limit = N
    for i in range(1, n_limit + 1):
        for j in range(i, n_limit + 1, i):
            sig[j] += i
            
    if time.time() - start_time > deadline - 30:
        # Too slow, return empty
        return {'M': 1, 'k': [1]}

    # Now group k by M = k * sigma(k)
    groups = defaultdict(list)
    max_M = 10**18
    
    for k in range(1, N + 1):
        m = k * sig[k]
        if m <= max_M:
            groups[m].append(k)
        else:
            if k * k > max_M:
                break

    # Find best group
    best_M = 0
    best_k_list = []
    
    for m, k_list in groups.items():
        if len(k_list) > len(best_k_list):
            best_M = m
            best_k_list = k_list
            
    # Now try to find additional k values for best_M by checking all divisors
    if best_M > 0:
        # Check time
        elapsed = time.time() - start_time
        if elapsed < deadline - 20:
            # Get all divisors of best_M
            divs = _divisors(best_M)
            # For each divisor k, check if k <= sqrt(best_M) and k * sigma(k) == best_M
            sqrt_M = math.isqrt(best_M)
            found_k = set(best_k_list)
            for k in divs:
                if k <= sqrt_M and k not in found_k:
                    # Check if k * sigma(k) == best_M
                    s = _sigma(k)
                    if k * s == best_M:
                        found_k.add(k)
            
            if len(found_k) > len(best_k_list):
                best_k_list = list(found_k)
    
    return {'M': int(best_M), 'k': [int(x) for x in best_k_list]}

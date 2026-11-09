"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _sigma(n):
    """Compute sigma(n) - sum of divisors of n."""
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            # Count power of p in n
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            # Sum of geometric series: 1 + p + p^2 + ... + p^exp
            result *= (p**(exp + 1) - 1) // (p - 1)
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (temp + 1)
    return result

def _factorize(n):
    """Return prime factorization as dict {prime: exponent}."""
    factors = {}
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            factors[p] = exp
        p += 1 if p == 2 else 2
    if temp > 1:
        factors[temp] = 1
    return factors

def _all_divisors_from_factors(factors):
    """Generate all divisors from prime factorization."""
    divisors = [1]
    for p, exp in factors.items():
        new_divs = []
        for d in divisors:
            for e in range(exp + 1):
                new_divs.append(d * (p ** e))
        divisors = new_divs
    return divisors

def _find_k_for_M(M, max_k=None):
    """Find all k such that k * sigma(k) = M and k <= isqrt(M)."""
    sqrt_M = math.isqrt(M)
    if max_k is None:
        max_k = sqrt_M
    
    # k must divide M, and k <= sqrt(M)
    # Also sigma(k) = M/k, so M/k must be an integer >= k (since sigma(k) >= k+1 for k>1, sigma(1)=1)
    solutions = []
    
    # k must be a divisor of M
    # Iterate through possible k values that divide M
    # k must satisfy: k | M and sigma(k) = M/k
    
    # Generate divisors of M up to sqrt(M)
    temp = M
    p = 2
    divisors = []
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            divisors.append((p, exp))
        p += 1 if p == 2 else 2
    if temp > 1:
        divisors.append((temp, 1))
    
    # Generate all divisors
    all_divs = [1]
    for p, exp in divisors:
        new_divs = []
        for d in all_divs:
            for e in range(exp + 1):
                new_divs.append(d * (p ** e))
        all_divs = new_divs
    
    # Filter: k <= sqrt(M) and k * sigma(k) == M
    for k in all_divs:
        if k > max_k:
            continue
        if M % k != 0:
            continue
        expected_sigma = M // k
        actual_sigma = _sigma(k)
        if actual_sigma == expected_sigma:
            solutions.append(k)
    
    return sorted(solutions)

def _search_best_M(deadline):
    """Search for M with many solutions."""
    best_M = 333312
    best_solutions = [336, 372, 434]
    best_count = 3
    
    # Strategy: Look at M values that are products of many small primes
    # to maximize the number of divisors k such that k*sigma(k) = M
    
    # Try M values based on highly composite structures
    # M = product of (p_i * sigma(p_i^a_i)) for various combinations
    
    # First, try known good M values and their multiples
    candidate_Ms = []
    
    # Generate candidates from small prime powers
    # For each combination of prime powers, compute M = k * sigma(k)
    # and count how many k give the same M
    
    # Better approach: enumerate k up to some limit, compute M = k*sigma(k),
    # then find M values with many k's
    
    # Since we need k <= sqrt(M), and M <= 10^18, k can be up to 10^9
    # But we can't enumerate all k up to 10^9
    
    # Alternative: For each M candidate (divisor-rich numbers), find all k
    
    # Let's try M values that are highly composite
    # M = 2^a * 3^b * 5^c * 7^d * ...
    
    # Generate M candidates by taking products of small primes to various powers
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    def gen_M_candidates(primes, max_power, current_idx, current_M, candidates):
        if current_idx == len(primes):
            if 1 <= current_M <= 10**18:
                candidates.append(current_M)
            return
        p = primes[current_idx]
        for e in range(max_power + 1):
            if current_M * (p ** e) > 10**18:
                break
            gen_M_candidates(primes, max_power, current_idx + 1, current_M * (p ** e), candidates)
    
    # Generate M candidates with limited prime exponents
    # Focus on small primes to keep M manageable
    candidates = []
    gen_M_candidates([2, 3, 5, 7, 11], 10, 0, 1, candidates)
    
    # Also add some known good values
    candidates.append(333312)
    
    # Sort by number of divisors (more divisors = more potential k values)
    def num_divisors(n):
        count = 0
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                count *= (exp + 1)
            p += 1 if p == 2 else 2
        if temp > 1:
            count *= 2
        return count
    
    # Sort candidates by divisor count, descending
    candidates.sort(key=num_divisors, reverse=True)
    
    # Limit candidates to avoid timeout
    candidates = candidates[:2000]
    
    for M in candidates:
        if time.time() > deadline:
            break
        if M < 1 or M > 10**18:
            continue
        
        solutions = _find_k_for_M(M)
        count = len(solutions)
        
        if count > best_count:
            best_count = count
            best_M = M
            best_solutions = solutions
    
    return best_M, best_solutions

def construct():
    deadline = time.time() + 140  # Leave 10s safety margin
    
    best_M, best_solutions = _search_best_M(deadline)
    
    # Ensure types are correct
    M = int(best_M)
    k_list = [int(k) for k in best_solutions]
    
    # Validate: all k_i <= isqrt(M) and k_i * sigma(k_i) == M
    sqrt_M = math.isqrt(M)
    valid_k = []
    for k in k_list:
        if 1 <= k <= sqrt_M:
            if k * _sigma(k) == M:
                valid_k.append(k)
    
    # Deduplicate and sort
    valid_k = sorted(set(valid_k))
    
    # If no valid solutions, use fallback
    if not valid_k:
        M = 333312
        valid_k = [336, 372, 434]
    
    # Limit to 1000
    valid_k = valid_k[:1000]
    
    return {'M': M, 'k': valid_k}
# EVOLVE-BLOCK-END

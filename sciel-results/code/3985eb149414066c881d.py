import math
import time
from collections import defaultdict

def _sigma_from_factors(factors):
    """
    Compute sigma(n) given its prime factorization.
    factors: list of (prime, exponent)
    """
    result = 1
    for p, exp in factors:
        if exp == 0:
            continue
        # sum_{i=0}^{exp} p^i = (p^(exp+1) - 1) / (p - 1)
        power = 1
        sum_val = 1
        for _ in range(exp):
            power *= p
            sum_val += power
        result *= sum_val
    return result

def _get_divisors_with_factors(prime_exp_list):
    """
    Yield all divisors of M given M's prime factorization.
    prime_exp_list: list of (prime, exponent) in M
    Returns list of tuples (divisor, list_of_factor_exponents_for_divisor)
    """
    divs = [(1, [0] * len(prime_exp_list))]
    for i, (p, exp) in enumerate(prime_exp_list):
        new_divs = []
        # Powers of p: 1, p, p^2, ..., p^exp
        powers = [1]
        for _ in range(exp):
            powers.append(powers[-1] * p)
        
        for d, exps in divs:
            for j in range(exp + 1):
                new_exps = exps[:]
                new_exps[i] = j
                new_divs.append((d * powers[j], new_exps))
        divs = new_divs
    return divs

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Primes to use for constructing M
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Generate highly composite numbers M
    # We want M <= 10^18
    # Recursive generation of exponents a_1 >= a_2 >= ... >= a_n
    # Start with max exponent for 2
    
    max_val = 10**18
    
    # We will use a stack or recursion to generate candidate M's
    # To avoid duplicates and ensure we cover a good space, we limit the depth
    # and the exponents.
    
    candidates = []
    
    def gen_candidates(idx, current_M, current_exps, max_exp):
        if time.time() > deadline:
            return
        if idx >= len(primes):
            if current_M > 1:
                candidates.append((current_M, current_exps[:]))
            return
        
        p = primes[idx]
        # Try exponents from max_exp down to 0
        # But we must ensure current_M * p^e <= max_val
        e = max_exp
        while True:
            if current_M * (p ** e) > max_val:
                e -= 1
                if e < 0:
                    return
                continue
            
            next_M = current_M * (p ** e)
            gen_candidates(idx + 1, next_M, current_exps + [e], e)
            
            e -= 1
            if e < 0:
                break
            
    # Start generation
    # We need to be careful with the number of candidates.
    # Let's limit the search to a reasonable number of candidates or time.
    
    # Initial call: idx=0, M=1, exps=[], max_exp=60 (since 2^60 > 10^18)
    # But 2^60 is approx 1.15e18. So max_exp for 2 is 59.
    
    # Optimization: Don't generate too many.
    # We can just generate the first N candidates or run until time limit.
    
    # Let's run the generator but break if we have enough candidates or time runs out.
    # We'll collect candidates in a list, but if it gets too big, we process them on the fly?
    # No, let's just generate a set of "top" highly composite numbers.
    
    # Actually, generating all HCNs up to 10^18 is feasible. There are not that many.
    # Number of HCNs up to 10^18 is in the thousands.
    
    gen_candidates(0, 1, [], 60)
    
    # Sort candidates? Maybe not necessary.
    # Process each candidate
    
    for M, exps in candidates:
        if time.time() > deadline:
            break
        
        # Get factorization of M
        prime_exp_list = [(primes[i], exps[i]) for i in range(len(primes)) if exps[i] > 0]
        
        # Get all divisors of M
        divs = _get_divisors_with_factors(prime_exp_list)
        
        valid_ks = []
        
        for k, k_exps in divs:
            # k <= sqrt(M) is required by the problem statement?
            # "distinct k_i <= sqrt(M)"
            # If k > sqrt(M), then sigma(k) < sqrt(M), so k*sigma(k) < M?
            # Wait. If k > sqrt(M), then M/k < sqrt(M) = sigma(k).
            # sigma(k) >= k+1 > sqrt(M) for k > 1.
            # So if k > sqrt(M), sigma(k) > sqrt(M) > M/k, so k*sigma(k) > M.
            # Thus, only k <= sqrt(M) can satisfy k*sigma(k) = M.
            # So we can filter k <= M // k (i.e., k*k <= M)
            
            if k * k > M:
                continue
                
            # Compute sigma(k) from k_exps
            # k_exps corresponds to primes in prime_exp_list
            k_factors = [(prime_exp_list[i][0], k_exps[i]) for i in range(len(k_exps)) if k_exps[i] > 0]
            
            sigma_k = _sigma_from_factors(k_factors)
            
            if k * sigma_k == M:
                valid_ks.append(k)
        
        if len(valid_ks) > best_count:
            best_count = len(valid_ks)
            best_M = M
            best_k = valid_ks
            
    # Fallback if no better found (should not happen as seed is valid)
    if best_count <= 3:
        # Use the seed result
        best_M = 333312
        best_k = [336, 372, 434]
        best_count = 3

    return {'M': int(best_M), 'k': [int(x) for x in best_k]}

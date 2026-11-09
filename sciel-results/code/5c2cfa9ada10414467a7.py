import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # seconds, leaving 5s safety margin

    # Fallback result
    best_result = {'M': 333312, 'k': [336, 372, 434]}
    best_count = 3

    def elapsed():
        return time.time() - start_time

    # Verify a candidate solution
    def verify_solution(M, k):
        """Verify that k * sigma(k) == M by exact factorization."""
        if k < 1 or k > math.isqrt(M):
            return False
        # Compute sigma(k)
        n = k
        sigma_val = 1
        d = 2
        temp = n
        if temp % 2 == 0:
            p_power = 1
            p_sum = 1
            while temp % 2 == 0:
                temp //= 2
                p_power *= 2
                p_sum += p_power
            sigma_val *= p_sum
        while d * d <= temp:
            if temp % d == 0:
                p_power = 1
                p_sum = 1
                while temp % d == 0:
                    temp //= d
                    p_power *= d
                    p_sum += p_power
                sigma_val *= p_sum
            d += 1
        if temp > 1:
            sigma_val *= (1 + temp)
        return k * sigma_val == M

    # Strategy 1: Brute-force sieve for small k
    # Use numpy if available for speed
    try:
        import numpy as np
        use_numpy = True
    except ImportError:
        use_numpy = False

    if use_numpy:
        # Sieve sigma up to max_k
        # Start with a reasonable max_k; we'll adjust based on time
        max_k = 5000000  # 5 million
        sigma = np.zeros(max_k + 1, dtype=np.int64)
        for i in range(1, max_k + 1):
            sigma[i::i] += i

        if elapsed() > DEADLINE:
            return best_result

        k_arr = np.arange(1, max_k + 1, dtype=np.int64)
        M_arr = k_arr * sigma[1:max_k + 1]
        # Valid: M <= 10^18 and k <= sigma(k) (i.e., k <= sqrt(M) since M = k*sigma(k))
        valid = (M_arr <= 10**18) & (k_arr <= sigma[1:max_k + 1])
        
        valid_idx = np.where(valid)[0]
        if len(valid_idx) > 0:
            M_valid = M_arr[valid_idx]
            k_valid = k_arr[valid_idx]
            
            # Group by M
            unique_M, inverse, counts = np.unique(M_valid, return_inverse=True, return_counts=True)
            
            # Find top M values
            top_n = min(100, len(unique_M))
            top_indices = np.argsort(counts)[::-1][:top_n]
            
            for idx in top_indices:
                cnt = int(counts[idx])
                if cnt > best_count:
                    best_M = int(unique_M[idx])
                    mask = inverse == idx
                    ks = [int(x) for x in k_valid[mask].tolist()[:1000]]
                    # Verify a few solutions
                    all_valid = all(verify_solution(best_M, k) for k in ks[:20])
                    if all_valid:
                        best_result = {'M': best_M, 'k': ks}
                        best_count = cnt

        if elapsed() > DEADLINE:
            return best_result

    # Strategy 2: Targeted construction of M values with many solutions
    # Key insight: if M = k * sigma(k), and we want many k with the same M,
    # we need M to be divisible by many different k values where sigma(k) = M/k.
    # 
    # A powerful approach: construct M from its prime factorization.
    # If M = product of many small primes, it has many divisors.
    # For each divisor k of M, check if sigma(k) = M/k.
    #
    # Better: enumerate k values that are divisors of a candidate M,
    # where M is chosen to maximize the number of valid k.
    
    # Let's try a different approach: for a given M, find all k such that
    # k | M and sigma(k) = M/k.
    # 
    # We can enumerate highly composite numbers M and for each, find divisors k
    # where sigma(k) = M/k.
    
    # Generate candidate M values: products of small primes
    # M = 2^a * 3^b * 5^c * 7^d * ...
    # For each such M, enumerate divisors k and check sigma(k) == M/k
    
    # But this is expensive. Instead, let's use a smarter approach:
    # 
    # For small k (up to ~10000), compute M = k * sigma(k).
    # Group by M. This is what we did above.
    # 
    # For larger k, we need a different strategy.
    # 
    # Alternative: for each candidate M (highly composite), find all k | M
    # such that sigma(k) = M/k.
    
    # Let's generate highly composite numbers up to 10^18
    # and for each, find valid k values.
    
    # Actually, let's try a hybrid: 
    # 1. For k up to K1, do brute force (done above)
    # 2. For larger k, use the fact that if k * sigma(k) = M, then k | M
    #    and sigma(k) = M/k. So k must be a divisor of M.
    #    We can enumerate M values that are "rich" in divisors.
    
    # Let's try: generate M = lcm(1, 2, ..., n) for various n,
    # then find all k | M such that sigma(k) = M/k.
    
    # Actually, a better approach: 
    # For each k, M = k * sigma(k). If we want many k with the same M,
    # M must be a common value of k * sigma(k) for many k.
    # 
    # The brute force above should find these if k is not too large.
    # Let's extend the search with a second pass for medium k.
    
    if elapsed() < DEADLINE - 30:
        # Second pass: medium k range
        # Use a different approach: for each M candidate, find k values
        # 
        # Let's try: for k in range(K1, K2), compute M = k * sigma(k)
        # and group. But we need sigma(k) for larger k.
        # 
        # Use a segmented sieve for sigma.
        
        K1 = 5000000
        K2 = 15000000  # 15 million total
        
        if use_numpy and elapsed() < DEADLINE - 30:
            # Extend sigma sieve
            # This is expensive; let's be careful
            max_k2 = K2
            sigma2 = np.zeros(max_k2 + 1, dtype=np.int64)
            for i in range(1, max_k2 + 1):
                sigma2[i::i] += i
            
            if elapsed() > DEADLINE - 10:
                return best_result
            
            k_arr2 = np.arange(K1 + 1, max_k2 + 1, dtype=np.int64)
            M_arr2 = k_arr2 * sigma2[K1 + 1:max_k2 + 1]
            valid2 = (M_arr2 <= 10**18) & (k_arr2 <= sigma2[K1 + 1:max_k2 + 1])
            
            valid_idx2 = np.where(valid2)[0]
            if len(valid_idx2) > 0:
                M_valid2 = M_arr2[valid_idx2]
                k_valid2 = k_arr2[valid_idx2]
                
                unique_M2, inverse2, counts2 = np.unique(M_valid2, return_inverse=True, return_counts=True)
                
                top_n2 = min(100, len(unique_M2))
                top_indices2 = np.argsort(counts2)[::-1][:top_n2]
                
                for idx in top_indices2:
                    cnt = int(counts2[idx])
                    if cnt > best_count:
                        best_M = int(unique_M2[idx])
                        mask = inverse2 == idx
                        ks = [int(x) for x in k_valid2[mask].tolist()[:1000]]
                        all_valid = all(verify_solution(best_M, k) for k in ks[:20])
                        if all_valid:
                            best_result = {'M': best_M, 'k': ks}
                            best_count = cnt

    # Strategy 3: Targeted M construction
    # For specific M values, directly find all k such that k * sigma(k) = M.
    # 
    # Key: if M = k * sigma(k), then k | M and sigma(k) = M/k.
    # So we need to find divisors k of M where sigma(k) = M/k.
    # 
    # Let's try M values that are products of small primes.
    
    if elapsed() < DEADLINE - 10:
        # Generate candidate M values
        # M = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * ...
        
        # For each candidate M, find all divisors k where sigma(k) = M/k
        # 
        # To find divisors of M, we factor M and enumerate divisors.
        
        # Let's try a few specific M values known to have many solutions
        # or construct M values with many divisors.
        
        # M = lcm(1, 2, ..., n) has many divisors
        # For n = 40, lcm(1..40) is large but manageable
        
        # Actually, let's try: for M in a set of highly composite numbers,
        # find all k | M such that sigma(k) = M/k.
        
        # Generate highly composite numbers
        def is_highly_composite(n):
            """Check if n has more divisors than any smaller number."""
            # This is expensive; skip for now
            return True
        
        # Instead, let's try specific M values
        # M = 2^a * 3^b * 5^c for various (a,b,c)
        
        # For each M, enumerate divisors k and check sigma(k) == M/k
        # 
        # To enumerate divisors of M = p1^a1 * p2^a2 * ..., 
        # we generate all combinations of exponents.
        
        def enumerate_divisors(prime_powers):
            """Generate all divisors from prime power list [(p, a), ...]."""
            divisors = [1]
            for p, a in prime_powers:
                new_divs = []
                p_power = 1
                for exp in range(a + 1):
                    for d in divisors:
                        new_divs.append(d * p_power)
                    p_power *= p
                divisors = new_divs
            return divisors
        
        def sigma_from_factorization(factorization):
            """Compute sigma from prime factorization [(p, a), ...]."""
            result = 1
            for p, a in factorization:
                result *= (p ** (a + 1) - 1) // (p - 1)
            return result
        
        # Try M values with many small prime factors
        # M = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f
        # 
        # For each such M, find all k | M where sigma(k) = M/k.
        
        # This is still expensive. Let's limit the search.
        
        # Try a few specific M values
        # M = 2^10 * 3^5 * 5^3 * 7^2 * 11 * 13
        # = 1024 * 243 * 125 * 49 * 11 * 13
        # = 1024 * 243 = 248832
        # 248832 * 125 = 31104000
        # 31104000 * 49 = 1524096000
        # 1524096000 * 11 = 16765056000
        # 16765056000 * 13 = 217945728000
        
        test_M_values = []
        
        # Generate M values as products of small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19]
        
        # Try M = product of first n primes
        for n in range(2, 9):
            M = 1
            for i in range(n):
                M *= small_primes[i]
            if M <= 10**18:
                test_M_values.append(M)
        
        # Try M = 2^a * 3^b * 5^c * 7^d for various exponents
        for a in range(5, 15, 2):
            for b in range(3, 10, 2):
                for c in range(2, 6):
                    for d in range(1, 4):
                        M = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                        if M <= 10**18:
                            test_M_values.append(M)
        
        # Deduplicate
        test_M_values = list(set(test_M_values))
        
        # For each M, find all k | M such that sigma(k) = M/k
        for M in test_M_values:
            if elapsed() > DEADLINE - 5:
                break
            
            # Factor M
            temp = M
            factorization = []
            d = 2
            while d * d <= temp:
                if temp % d == 0:
                    a = 0
                    while temp % d == 0:
                        temp //= d
                        a += 1
                    factorization.append((d, a))
                d += 1
            if temp > 1:
                factorization.append((temp, 1))
            
            # Enumerate divisors
            divisors = enumerate_divisors(factorization)
            
            # For each divisor k, check if sigma(k) = M/k
            valid_ks = []
            for k in divisors:
                if k > math.isqrt(M):
                    continue
                if M % k != 0:
                    continue
                target = M // k
                # Compute sigma(k)
                # Factor k
                temp_k = k
                k_factorization = []
                d2 = 2
                while d2 * d2 <= temp_k:
                    if temp_k % d2 == 0:
                        a2 = 0
                        while temp_k % d2 == 0:
                            temp_k //= d2
                            a2 += 1
                        k_factorization.append((d2, a2))
                    d2 += 1
                if temp_k > 1:
                    k_factorization.append((temp_k, 1))
                
                sigma_k = sigma_from_factorization(k_factorization)
                
                if sigma_k == target:
                    valid_ks.append(k)
            
            if len(valid_ks) > best_count:
                # Verify
                all_valid = all(verify_solution(M, k) for k in valid_ks[:20])
                if all_valid:
                    best_result = {'M': M, 'k': valid_ks[:1000]}
                    best_count = len(valid_ks)
    
    return best_result

def construct():
    import time
    from math import isqrt
    
    deadline = time.time() + 110  # 110s to be safe
    
    def generate_powerful(limit):
        """Generate all powerful numbers up to limit."""
        powerful_set = {1}
        # For each prime p, p^k for k >= 2
        # Powerful numbers are of form ∏ p_i^{e_i} with e_i >= 2
        # We can generate by: for each m, m² * n where n is powerful and coprime to m... 
        # Actually, a powerful number n can be written as a² * b where b is powerful and 
        # all prime factors of b divide a... no.
        # Simpler: n is powerful iff n = a² * b where b is powerful and gcd(a,b)=1? No.
        # Actually: n is powerful iff for every prime p | n, p² | n.
        # Generation: iterate over possible "core" squares.
        # n = x² * y where y is powerful and every prime dividing y also divides x? No.
        
        # Better approach: 
        # A powerful number n satisfies n = ∏ p^{e_p} with e_p >= 2.
        # We can generate by: for each possible square part s = x², and for each powerful 
        # number y that is coprime to x... this is getting complex.
        
        # Simplest: enumerate all n up to limit and check if powerful.
        # But that's O(limit * factorization) which is too slow for large limit.
        
        # Alternative: generate by recursion over primes.
        # For each prime p, the exponent can be 0, 2, 3, 4, ...
        # But we need to be careful about the product staying <= limit.
        
        # Let's use a different approach: 
        # n is powerful iff n = a² * b where b is powerful and gcd(a,b) = 1? 
        # No, that's not right either.
        
        # Actually, the standard way: n is powerful iff n = m² * k where k is powerful 
        # and every prime dividing k divides m... no.
        
        # Let me just use: for each a from 1 to isqrt(limit), a² is a candidate. 
        # Then for each a, multiply by p^k where p is prime, k >= 1, and the result 
        # stays powerful and <= limit. But this requires tracking which primes are used.
        
        # Simpler practical approach: 
        # Generate all numbers of the form x² * y where y is powerful and y <= limit / x².
        # But y must be powerful, and the product must be powerful.
        # x² * y is powerful iff y is powerful (since x² contributes even exponents, 
        # and y contributes exponents >= 2, so total exponents are >= 2).
        # Wait, that's true! If y is powerful, then x² * y is powerful for any x.
        # Because for any prime p: if p | x, then v_p(x²) >= 2, and v_p(y) >= 0, so v_p(x²*y) >= 2.
        # If p ∤ x, then v_p(x²) = 0, and v_p(y) >= 2 (if p | y) or 0 (if p ∤ y).
        # So x² * y is powerful iff y is powerful.
        
        # So: generate all powerful numbers y up to limit, then for each y, 
        # for each x with x² * y <= limit, add x² * y.
        # But this generates duplicates and is still complex.
        
        # Let me just do: for each n from 1 to limit, check if n is powerful by factoring.
        # For limit up to ~10^7, this is feasible with a sieve-like approach.
        
        # Actually, let's use the property: n is powerful iff n = a² * b where b is powerful 
        # and b <= n. We can generate by iterating over possible b (powerful) and a.
        
        # Practical approach for limit up to ~10^8:
        # 1. Generate all powerful numbers up to limit using a recursive/multiplicative method.
        # 2. Or: for each a from 1 to isqrt(limit), a² is powerful. Then for each prime p 
        #    not dividing a, we can multiply by p², p³, etc. But this is complex.
        
        # Let me use a simpler method: 
        # A number n is powerful iff for every prime p, if p|n then p²|n.
        # We can check this by: for each n, factor it and verify all exponents >= 2.
        # For n up to 10^7, we can precompute smallest prime factor (SPF) and then factor quickly.
        
        # But for limit up to 10^8, SPF array is 10^8 integers = 400MB, too much.
        # For limit up to 10^7, SPF is 10^7 integers = 40MB, feasible.
        
        # Let's set limit based on available memory and time.
        # For now, let's generate powerful numbers up to a moderate limit using a different method.
        
        # Method: 
        # powerful = {1}
        # For each prime p:
        #   For each existing powerful number m in powerful:
        #     For k = 2, 3, 4, ... while m * p^k <= limit:
        #       Add m * p^k to powerful
        # But this doesn't work because we need to combine multiple primes.
        
        # Correct recursive approach:
        # def gen_powerful(idx, current, primes, limit):
        #   if idx == len(primes): add current
        #   else:
        #     # don't use primes[idx]
        #     gen_powerful(idx+1, current, primes, limit)
        #     # use primes[idx]^k for k >= 2
        #     p = primes[idx]
        #     pk = p*p
        #     while current * pk <= limit:
        #       gen_powerful(idx+1, current * pk, primes, limit)
        #       pk *= p
        
        # This works but the number of primes up to limit is large.
        # However, for powerful numbers, only primes up to sqrt(limit) can appear with exponent >= 2 
        # in a "new" way... actually no, a prime p > sqrt(limit) can appear with exponent 2 if p² <= limit.
        # So primes up to sqrt(limit) suffice for the recursive generation? No, p can be up to limit^(1/2) 
        # for exponent 2, but for higher exponents, p is smaller.
        
        # Actually, in the recursive generation, we need all primes p where p² <= limit, i.e., p <= sqrt(limit).
        # For limit = 10^7, sqrt(limit) ~ 3162, so about 450 primes. The recursion depth is 450, 
        # but most branches terminate quickly.
        
        # Let me implement this.
        
        import math
        
        # Sieve primes up to isqrt(limit)
        sqrt_limit = isqrt(limit)
        if sqrt_limit < 2:
            return [1]
        
        # Sieve of Eratosthenes
        sieve = bytearray(b'\x01') * (sqrt_limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, isqrt(sqrt_limit) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        primes = [i for i in range(2, sqrt_limit + 1) if sieve[i]]
        
        powerful_set = set()
        
        def gen(idx, current):
            if current > limit:
                return
            if idx == len(primes):
                powerful_set.add(current)
                return
            # Don't use primes[idx]
            gen(idx + 1, current)
            # Use primes[idx]^k for k >= 2
            p = primes[idx]
            pk = p * p
            while current * pk <= limit:
                gen(idx + 1, current * pk)
                pk *= p
        
        gen(0, 1)
        return sorted(powerful_set)
    
    # Generate powerful numbers
    # Start with a moderate limit and increase if time permits
    limit = 5000000
    powerful = generate_powerful(limit)
    powerful_set = set(powerful)
    
    # Count representations
    sum_count = {}
    best_N = 2
    best_count = 0
    best_pairs = []
    
    K = len(powerful)
    
    # Iterate over pairs
    for i in range(K):
        a = powerful[i]
        if time.time() > deadline:
            break
        for j in range(i, K):
            b = powerful[j]
            s = a + b
            if s > 10**18:
                break
            if s not in sum_count:
                sum_count[s] = 0
                sum_count[s] = 1
            else:
                sum_count[s] += 1
            
            if sum_count[s] > best_count:
                best_count = sum_count[s]
                best_N = s
                # We'll reconstruct pairs later
    
    # Reconstruct pairs for best_N
    if best_N > 2:
        pairs = []
        for a in powerful:
            if 2 * a > best_N:
                break
            b = best_N - a
            if b in powerful_set:
                pairs.append([a, b])
        pairs.sort()
        return {'N': best_N, 'pairs': pairs}
    else:
        return {'N': 2, 'pairs': [[1, 1]]}

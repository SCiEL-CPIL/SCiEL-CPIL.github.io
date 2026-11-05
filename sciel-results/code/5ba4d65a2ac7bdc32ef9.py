import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s margin
    
    N = 1000000
    max_n = N + 1
    
    # Sieve of smallest prime factors
    spf = list(range(max_n + 1))
    for i in range(2, int(max_n**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, max_n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def powerful_part(n):
        if n == 1:
            return 1
        original_n = n
        result = 1
        while n > 1:
            p = spf[n]
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            if count >= 2:
                result *= (p ** count)
        return result
    
    best_n = 10000
    max_prod = 0
    
    # Precompute powerful parts for all n in range?
    # That would take 10^6 * O(log N) operations, which is fine.
    # But we only need P(n) and P(n+1).
    # We can compute P(n) on the fly or precompute an array.
    # Precomputing is faster in Python due to loop overhead reduction?
    # Let's precompute P(n) for n in [10000, 1000000].
    
    P = [1] * (max_n + 1)
    for i in range(2, max_n + 1):
        # This naive loop is O(N log N) which is ~14e6 operations. Fine.
        # But we can use the spf to be faster.
        pass
        
    # Actually, let's just compute on the fly in the main loop to keep memory low and code simple.
    # Or better, precompute P array using a modified sieve logic?
    # No, factorization per number is fine.
    
    for n in range(10000, N):
        if time.time() > deadline:
            break
            
        p_n = powerful_part(n)
        if p_n == 1:
            # If P(n) is 1, product is 1 * P(n+1) <= N.
            # Max possible product is ~1e12.
            # If current max_prod > N, we can skip.
            if max_prod > N:
                continue
                
        p_np1 = powerful_part(n + 1)
        
        prod = p_n * p_np1
        if prod > max_prod:
            max_prod = prod
            best_n = n
            
    return best_n

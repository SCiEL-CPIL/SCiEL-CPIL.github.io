import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    MAX_N = 1000000
    MAX_VAL = MAX_N + 1
    
    # 1. Generate primes up to sqrt(MAX_VAL) ~ 1000
    limit = int(MAX_VAL**0.5) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # 2. Precompute powerful parts
    # powerful[i] will store the powerful part of i
    powerful = [1] * (MAX_VAL + 1)
    
    for p in primes:
        q = p * p
        while q <= MAX_VAL:
            # Iterate through multiples of q
            for m in range(q, MAX_VAL + 1, q):
                # Calculate the exponent of p in m
                temp = m
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                if cnt >= 2:
                    powerful[m] *= (p ** cnt)
            q *= p
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break

    # 3. Find n in [10000, 1000000] maximizing powerful[n] * powerful[n+1]
    best_n = 10000
    best_score = 0
    
    for n in range(10000, MAX_N + 1):
        score = powerful[n] * powerful[n + 1]
        if score > best_score:
            best_score = score
            best_n = n
            
        if time.time() > deadline:
            break
            
    return best_n

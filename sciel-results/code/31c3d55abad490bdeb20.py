import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s buffer for safety
    
    N = 1000000
    
    # 1. Compute primes up to N using sieve
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # 2. Compute Powerful Part P(n) for all n using sieve approach
    # P[n] = product of p^e for all prime powers p^e (e>=2) where p^e || n (exactly divides n)
    P = [1] * (N + 1)
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        # For each power p^e with e >= 2
        pe = p2
        while pe <= N:
            pe_next = pe * p
            if pe_next > N:
                break
            # For numbers divisible by pe but not by pe_next, multiply P by pe
            for m in range(pe, N + 1, pe):
                if m % pe_next != 0:
                    P[m] *= pe
            pe = pe_next
    
    # 3. Find max P(n)*P(n+1) with deadline guard
    max_val = 0
    best_n = 10000
    for n in range(10000, N):
        score = P[n] * P[n + 1]
        if score > max_val:
            max_val = score
            best_n = n
        
        # Deadline check
        if n % 50000 == 0:
            if time.time() > deadline:
                break
    
    return best_n

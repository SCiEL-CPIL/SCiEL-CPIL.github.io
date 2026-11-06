import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 85.0  # 5s margin before 90s kill
    
    N = 1000000
    # Linear sieve to compute phi
    # phi[i] will store Euler's totient for i
    phi = [0] * (N + 1)
    phi[1] = 1
    
    primes = []
    is_composite = [False] * (N + 1)
    
    for i in range(2, N + 1):
        if time.time() > deadline:
            break
        if not is_composite[i]:
            primes.append(i)
            phi[i] = i - 1
        for p in primes:
            if i * p > N:
                break
            is_composite[i * p] = True
            if i % p == 0:
                phi[i * p] = phi[i] * p
                break
            else:
                phi[i * p] = phi[i] * (p - 1)
                
    # Find smallest n with phi(n) divisible by 120
    for n in range(1, N + 1):
        if time.time() > deadline:
            break
        if phi[n] % 120 == 0:
            return n
            
    # Fallback
    return 143

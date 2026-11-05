import time
import numpy as np

def construct():
    start_time = time.time()
    LIMIT = 90  # seconds (evaluator kills at 90s)
    MARGIN = 5  # seconds safety margin
    
    N_MAX = 1000001  # We need P up to 1000001 to evaluate n=1000000
    START = 10000
    
    # 1. Sieve for Primes
    # Simple sieve of Eratosthenes
    is_prime = [True] * (N_MAX + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(N_MAX)) + 1):
        if is_prime[i]:
            for j in range(i*i, N_MAX + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # 2. Compute Powerful Part P[k] using NumPy vectorization
    P = np.ones(N_MAX + 1, dtype=np.int64)
    
    for p in primes:
        q = p * p
        if q > N_MAX:
            break
        # Multiply by p for each power p^k (k>=2) dividing m
        while q <= N_MAX:
            P[q:N_MAX+1:q] *= p
            q *= p
        # Multiply by p one more time for all multiples of p^2
        P[p*p:N_MAX+1:p*p] *= p
            
    # Check time
    if time.time() - start_time > LIMIT - MARGIN:
        return 10000

    # 3. Find n that maximizes P[n] * P[n+1]
    try:
        left = P[START:1000001]
        right = P[START+1:1000002]
        
        products = left * right
        
        max_idx = np.argmax(products)
        best_n = int(START + max_idx)
        
    except Exception:
        best_n = START
        best_score = -1
        for n in range(START, 1000001):
            score = P[n] * P[n + 1]
            if score > best_score:
                best_score = score
                best_n = n

    return best_n

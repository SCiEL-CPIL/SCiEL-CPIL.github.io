import time
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 85.0

    # We need to evaluate n up to 1,000,000.
    # So we need P(n) and P(n+1) for n=1,000,000.
    # Thus we need P up to 1,000,001.
    MAXN = 1_000_001

    # 1. Sieve of Eratosthenes for primes up to MAXN
    is_prime = np.ones(MAXN + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAXN**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False

    primes = np.where(is_prime)[0]

    # 2. Compute h(x)
    # h[x] = product of p where v_p(x) == 1
    # P[x] = x / h[x]
    
    # Using int64 to ensure sufficient range for intermediate products if any, 
    # though h[x] <= x so int32 would suffice for h, but P[x] * P[x+1] can be large.
    # Actually P[x] <= x, so P[x]*P[x+1] can be up to 1e12, which fits in int64.
    
    h = np.ones(MAXN + 1, dtype=np.int64)

    for p in primes:
        p = int(p)
        # Multiply by p for all multiples of p
        # This accumulates the square-free part
        h[p::p] *= p
        
        # Divide by p for all multiples of p^2
        # This removes p from the square-free part if exponent >= 2
        p2 = p * p
        if p2 <= MAXN:
            h[p2::p2] //= p

    if time.time() - t0 > DEADLINE:
        return 10000

    # 3. Compute P[x]
    # P[x] = x // h[x]
    # x_vals = np.arange(MAXN + 1, dtype=np.int64)
    # P = x_vals // h
    
    # To save memory and time, we can just do the division on the fly or create the array.
    # Creating the array is fast in NumPy.
    x_vals = np.arange(MAXN + 1, dtype=np.int64)
    P = x_vals // h

    # 4. Maximize P[n] * P[n+1] for n in [10000, 1000000]
    # n ranges from 10000 to 1000000
    # So we look at P[10000] * P[10001] ... P[1000000] * P[1000001]
    
    start = 10000
    end = 1000000

    # Slice P from start to end+1 (inclusive)
    # We want pairs (start, start+1), ..., (end, end+1)
    # So we need P[start : end+2]
    
    P_slice = P[start:end+2] # Length is (end+2) - start = end - start + 2
    
    # products[i] = P_slice[i] * P_slice[i+1]
    # We want i such that the first element corresponds to n=start+i
    # The product for n is P[n] * P[n+1]
    # So products = P_slice[:-1] * P_slice[1:]
    
    products = P_slice[:-1] * P_slice[1:]

    # Find index of max
    idx = np.argmax(products)
    best_n = int(start + idx)

    # Ensure best_n is in valid range
    if best_n < 10000 or best_n > 1000000:
        best_n = 10000

    return best_n

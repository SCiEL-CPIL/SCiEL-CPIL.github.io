import numpy as np
import time

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
        
        # 2. Compute P[x] directly
        # P[x] = product of p^e for all p^e || x with e >= 2
        
        P = np.ones(MAXN + 1, dtype=np.int64)
        
        for p in primes:
            p = int(p)
            p2 = p * p
            if p2 > MAXN:
                continue
            
            # Accumulate factors for exponents >= 2
            # For each k >= 2, multiply P[n] by p for n divisible by p^k
            # This results in factor p^(v_p(n)-1) for v_p(n) >= 2
            power = p2
            while power <= MAXN:
                P[power::power] *= p
                power *= p
            
            # Correction: multiply by p again for all n divisible by p^2
            # This adds the missing factor of p to reach p^v_p(n)
            P[p2::p2] *= p
            
        if time.time() - t0 > DEADLINE:
            return 10000
        
        # 4. Maximize P[n] * P[n+1] for n in [10000, 1000000]
        # n ranges from 10000 to 1000000
        # So we look at P[10000] * P[10001] ... P[1000000] * P[1000001]
        
        start = 10000
        end = 1000000
        
        # Slice P from start to end+1 (inclusive)
        # products[i] = P[start+i] * P[start+i+1]
        # We want i such that start+i is in [start, end]
        # So i ranges from 0 to end - start
        
        P_slice = P[start:end+2] # Length is (end+2) - start = end - start + 2
        # We want pairs (0,1), (1,2), ..., (len-2, len-1)
        # products = P_slice[:-1] * P_slice[1:]
        
        products = P_slice[:-1] * P_slice[1:]
        
        # Find index of max
        idx = np.argmax(products)
        best_n = int(start + idx)
        
        # Ensure best_n is in valid range
        if best_n < 10000 or best_n > 1000000:
            best_n = 10000
            
        return best_n

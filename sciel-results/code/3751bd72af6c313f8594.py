import numpy as np
import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # 1. Sieve phi
    # Determine max sieve size based on time/memory
    # Let's try 10^8
    N = 10**8
    # If memory is a concern, we can reduce N.
    # 10^8 * 4 bytes = 400MB.
    
    try:
        phi = np.arange(N + 1, dtype=np.int64)
        # Use int64 to be safe, or int32 if sure. 
        # phi(10^8) < 10^8, so int32 is enough.
        phi = phi.astype(np.int32)
        
        for i in range(2, N + 1):
            if phi[i] == i: # i is prime
                # phi[j] -= phi[j] // i for j in multiples of i
                # Vectorized operation
                phi[i::i] -= phi[i::i] // i
                
        # 2. Find max run
        best_k = 1
        best_s = 1
        # Sliding window
        # We want max length k such that phi[s..s+k-1] are distinct.
        # Note: phi[1]=1, phi[2]=1. So run starting at 1 or 2 will break immediately if len>1.
        
        # We can just iterate and keep a set.
        # But a set of size 200 is small.
        # However, doing this for 10^8 elements in Python loop might be slow.
        # 10^8 iterations is ~10-30 seconds in Python.
        # We can optimize by only checking if we have potential to beat best_k.
        
        # Alternative:
        # For each start s, check up to min(200, N-s+1) elements.
        # But that's O(N * 200) = 2*10^10 ops. Too slow.
        
        # Sliding window with a frequency dict is O(N).
        freq = {}
        left = 0
        current_len = 0
        for right in range(N + 1):
            val = phi[right]
            freq[val] = freq.get(val, 0) + 1
            # Shrink window if duplicate
            while freq[val] > 1:
                left_val = phi[left]
                freq[left_val] -= 1
                left += 1
            
            current_len = right - left + 1
            if current_len > best_k:
                best_k = current_len
                best_s = left
            # Cap window size to 200? No, we want max.
            # But if current_len > 200, we can stop checking further for this left?
            # Actually, if we find a run > 200, we are done (max k is 200).
            if best_k >= 200:
                break
                
    except MemoryError:
        # Fallback to smaller sieve or other method
        N = 10**7
        # ... repeat sieve ...

    # 3. Random search in higher range if time permits
    # Use the remaining time to search [N, 10^12]
    # But as argued, this is likely futile for beating a sieve result > 20.
    # We will implement a basic random search just in case.
    
    # Fast phi for single number
    def fast_phi(n):
        if n == 1: return 1
        result = n
        temp = n
        # Trial division up to 10^5
        for p in range(2, 10**5):
            if p * p > temp: break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            result -= result // temp
        return result

    # Miller Rabin and Pollard Rho if needed for very large factors
    # But trial division up to 10^5 handles most composites.
    # If temp > 1, it is either prime or product of two primes > 10^5.
    # If product of two primes > 10^5, then temp > 10^10.
    # n <= 10^12, so temp <= 10^12.
    # If temp is composite, it has a factor <= sqrt(temp) <= 10^6.
    # So trial division up to 10^6 is sufficient to fully factorize any n <= 10^12?
    # No, if n = p * q with p, q ~ 10^6, sqrt(n) ~ 10^6.
    # So trial division up to 10^6 is enough.
    # 10^6 iterations per number is slow.
    # We need primes list.
    
    # Generate primes up to 10^6
    LIMIT = 10**6
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]

    def phi_single(n):
        if n == 1: return 1
        result = n
        temp = n
        for p in primes:
            if p * p > temp: break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            result -= result // temp
        return result

    if time.time() - start_time < DEADLINE - 10:
        # Random search
        random.seed(42)
        # Sample a few points
        for _ in range(1000):
            if time.time() - start_time > DEADLINE - 5:
                break
            s = random.randint(N, 10**12 - 200)
            # Check window
            seen = set()
            k = 0
            for i in range(200):
                v = s + i
                if v > 10**12: break
                pv = phi_single(v)
                if pv in seen:
                    break
                seen.add(pv)
                k += 1
            if k > best_k:
                best_k = k
                best_s = s

    return {'start': best_s, 'length': best_k}

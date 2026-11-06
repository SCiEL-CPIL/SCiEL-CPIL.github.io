import numpy as np
import time
import random

def construct():
    # Deadline guard
    deadline = time.time() + 110
    
    # 1. Sieve phi up to N
    # Start with a conservative N, expand if fast
    N = 10**7
    
    # Initialize phi array
    phi = np.arange(N + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes to get primes up to sqrt(N)
    limit = int(np.sqrt(N)) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, limit + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = False
    primes = np.where(is_prime)[0]
    
    # Update phi values
    for p in primes:
        multiples = np.arange(p, N + 1, p)
        phi[multiples] -= phi[multiples] // p
        
    # 2. Find longest run of distinct values
    best_k = 1
    best_s = 1
    
    # The first value is phi[1] = 1
    # We start scanning from 1
    # Note: phi[0] is 0, but s>=1
    
    # Optimization: Use a set. 
    # To speed up the Python loop, we can't easily vectorize this.
    # But 10^7 iterations is manageable.
    
    seen = {phi[1]}
    current_k = 1
    
    # We iterate from 2 to N
    # To avoid Python overhead, we can convert to list? No, numpy iteration is slower than list iteration?
    # Actually, iterating over numpy array in Python is slow.
    # Converting to list: list(phi) takes time and memory.
    # Let's stick to numpy iteration or use a memoryview?
    # For now, standard loop.
    
    for i in range(2, N + 1):
        if time.time() > deadline:
            break
        
        v = phi[i]
        if v in seen:
            # Reset window
            seen = {v}
            current_k = 1
        else:
            seen.add(v)
            current_k += 1
            if current_k > best_k:
                best_k = current_k
                best_s = i - current_k + 1
                
    # 3. If time remains, try larger N or random search
    # Let's try to extend N if we finished early
    elapsed = time.time()
    if elapsed < deadline - 20:
        # Try N = 2 * 10^7
        # This would require re-computing or extending.
        # Re-computing is simpler.
        pass # Skipping for now to keep code simple and safe.
        
    # Random search for s > N
    if time.time() < deadline - 5:
        # Generate primes up to 10^6 for trial division
        limit2 = 10**6
        is_prime2 = np.ones(limit2 + 1, dtype=bool)
        is_prime2[0] = is_prime2[1] = False
        for i in range(2, int(np.sqrt(limit2)) + 1):
            if is_prime2[i]:
                is_prime2[i*i:limit2+1:i] = False
        primes_large = np.where(is_prime2)[0]
        
        random.seed(42)
        
        def phi_large(n):
            result = n
            temp = n
            for p in primes_large:
                if p * p > temp:
                    break
                if temp % p == 0:
                    while temp % p == 0:
                        temp //= p
                    result -= result // p
            if temp > 1:
                result -= result // temp
            return result
            
        # Do a few random checks
        # We check windows of size best_k + 1
        for _ in range(50):
            if time.time() > deadline:
                break
            s = random.randint(N, 10**12)
            if s + best_k > 10**12:
                continue
            
            seen_rand = set()
            valid = True
            # Check if we can beat best_k
            # We need a run of length best_k + 1
            # So we check best_k + 1 values
            for j in range(best_k + 1):
                v = phi_large(s + j)
                if v in seen_rand:
                    valid = False
                    break
                seen_rand.add(v)
                
            if valid:
                # Found a run of at least best_k + 1
                # Extend it to find max
                current_k_rand = best_k + 1
                # seen_rand already has best_k+1 values
                j = best_k + 1
                while s + j <= 10**12 and j < 200:
                    v = phi_large(s + j)
                    if v in seen_rand:
                        break
                    seen_rand.add(v)
                    current_k_rand += 1
                    j += 1
                    
                if current_k_rand > best_k:
                    best_k = current_k_rand
                    best_s = s

    return {'start': best_s, 'length': best_k}

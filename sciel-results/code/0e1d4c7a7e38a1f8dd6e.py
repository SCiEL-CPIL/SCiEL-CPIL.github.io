import time
import math
import numpy as np
from array import array

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    N = 10**9
    
    # Sieve
    # Use numpy for speed
    try:
        is_prime = np.ones(N + 1, dtype=bool)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(math.isqrt(N)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        
        # Extract primes
        # This creates a numpy array of indices
        primes_np = np.where(is_prime)[0]
        
        # Convert to list? No, keep as numpy array for faster access?
        # Accessing numpy array elements is slightly slower than list in Python?
        # Actually, list is faster for iteration in Python loops.
        # But numpy array is more compact.
        # Let's convert to list for the loop.
        # primes_list = primes_np.tolist()
        
        # Memory check: 50M ints in list is large.
        # Let's stick to numpy array and use .item() or direct indexing.
        # Direct indexing in numpy is slow in Python loops.
        # List is faster.
        
        # Let's try to use a list but only for the last 10,000 primes?
        # No, we need primes[0..i] for the inner sieve.
        # So we need the whole list.
        
        # If memory is an issue, we can use a segmented sieve for the inner loop?
        # No, we need the primes.
        
        # Let's assume list is okay.
        primes_list = primes_np.tolist()
        
    except MemoryError:
        N = 10**8
        is_prime = np.ones(N + 1, dtype=bool)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(math.isqrt(N)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        primes_list = np.where(is_prime)[0].tolist()

    best_k = 1
    best_p = 2
    
    def get_k(p, idx):
        vals = [p*p + j for j in range(1, 201)]
        pr = primes_list
        for j in range(idx + 1):
            q = pr[j]
            base = p*p
            # Start multiple
            m0 = (base + 1 + q - 1) // q
            start_val = m0 * q
            idx_v = start_val - (base + 1)
            
            while idx_v < 200:
                v = vals[idx_v]
                while v % q == 0:
                    v //= q
                vals[idx_v] = v
                idx_v += q
        
        k = 0
        for v in vals:
            if v <= p:
                k += 1
            else:
                break
        return k

    # Iterate last 5000 primes
    # 5000 * 1s = 5000s. Too slow.
    # We need to be faster.
    # We can only check ~100 primes.
    
    # Let's check the last 200 primes.
    for i in range(len(primes_list) - 1, max(len(primes_list) - 200, -1), -1):
        if time.time() > deadline - 5:
            break
        p = primes_list[i]
        k = get_k(p, i)
        if k > best_k:
            best_k = k
            best_p = p
        if best_k == 200:
            break

    return {'p': best_p, 'k': best_k}

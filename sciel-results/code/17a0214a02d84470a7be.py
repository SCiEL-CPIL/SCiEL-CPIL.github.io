import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Sieve of Eratosthenes up to 10^6 (sqrt(10^12))
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    def compute_phi_window(s, W):
        # s: start of window, W: window size
        # Returns list of phi(s+i) for i in range(W)
        rem = [s + i for i in range(W)]
        phi = [s + i for i in range(W)]
        
        for p in primes:
            if p * p > s + W - 1:
                # Optimization: if p*p > max_n, we can break? 
                # No, because rem[i] might be a prime > p. 
                # But if p*p > max_n, then rem[i] can have at most one prime factor > 1.
                # The loop handles all p. We can stop if p > max(rem)? 
                # Actually, we just iterate through all primes up to 10^6.
                pass
            
            # Find first multiple of p in [s, s+W-1]
            # s + i = k*p => i = (k*p - s)
            # k*p >= s => k >= ceil(s/p)
            k_start = (s + p - 1) // p
            first_mult = k_start * p
            i_start = first_mult - s
            
            for i in range(i_start, W, p):
                if rem[i] % p == 0:
                    phi[i] -= phi[i] // p
                    while rem[i] % p == 0:
                        rem[i] //= p
        
        for i in range(W):
            if rem[i] > 1:
                phi[i] -= phi[i] // rem[i]
                
        return phi

    def check_window(s, max_len=200):
        # Check if phi(s), ..., phi(s+max_len-1) are distinct
        # We compute phi for the whole window and then check for duplicates
        phi_vals = compute_phi_window(s, max_len)
        
        seen = set()
        for i in range(max_len):
            v = phi_vals[i]
            if v in seen:
                return i
            seen.add(v)
        return max_len

    best_k = 1
    best_s = 1
    
    # Check small range first to ensure we have a valid baseline
    # s=1, phi(1)=1, phi(2)=1 -> collision at 2. k=1.
    # s=2, phi(2)=1, phi(3)=2, phi(4)=2 -> collision at 4. k=2 (2,3).
    # s=3, phi(3)=2, phi(4)=2 -> collision. k=1.
    # s=4, phi(4)=2, phi(5)=4, phi(6)=2 -> collision at 6. k=2 (4,5).
    # s=5, phi(5)=4, phi(6)=2, phi(7)=6, phi(8)=4 -> collision at 8. k=3 (5,6,7).
    # Let's just run the search.
    
    # To speed up, we can skip some s if we want, but random is fine.
    # We will sample s uniformly from [1, 10^12 - 200]
    
    # Heuristic: Collisions are more likely for small numbers.
    # So large s is better.
    # We will bias towards large s.
    
    while time.time() < deadline:
        # Random s
        # Use a mix of uniform and biased
        if random.random() < 0.9:
            s = random.randint(10**10, 10**12 - 200)
        else:
            s = random.randint(1, 10**10)
            
        k = check_window(s)
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                break
                
    return {'start': best_s, 'length': best_k}

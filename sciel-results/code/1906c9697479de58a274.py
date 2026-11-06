import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s margin, evaluator kills at 90s
    
    N_MAX = 1000000
    N_LIMIT = N_MAX + 1  # We need n+1 up to 1000001
    
    # 1. Sieve of Eratosthenes to find primes up to N_LIMIT using NumPy
    is_prime = np.ones(N_LIMIT + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(N_LIMIT**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0].tolist()
    
    # 2. Precompute powerful parts using NumPy
    val = np.ones(N_LIMIT + 1, dtype=np.float64)
    
    for p in primes:
        p2 = p * p
        if p2 > N_LIMIT:
            break
        
        # Iterate through powers p^k for k >= 2
        power = p2
        next_power = power * p
        
        while power <= N_LIMIT:
            if time.time() > deadline:
                break
            
            # Multiples of 'power' not divisible by 'next_power'
            # These are m ≡ power (mod next_power)
            if next_power > N_LIMIT:
                # All multiples of power
                indices = np.arange(power, N_LIMIT + 1, power)
                val[indices] *= power
            else:
                # m = power * t where t is not divisible by p
                # Equivalently, m ≡ power (mod next_power)
                # m = power + next_power * j for j = 0, 1, 2, ...
                # while m <= N_LIMIT
                indices = np.arange(power, N_LIMIT + 1, next_power)
                # But we also need multiples of power that are NOT at those positions
                # Actually: multiples of power are power, 2*power, 3*power, ...
                # Those divisible by next_power = p*power are: next_power, 2*next_power, ...
                # So we want: multiples of power MINUS multiples of next_power
                
                # Simpler: iterate multiples of power, skip those divisible by next_power
                # Using the congruence: m mod next_power == power (since m = power * t, 
                # m mod next_power = power * (t mod p))
                # We want t mod p != 0, i.e., m mod next_power != 0
                # m = power * t, so m mod next_power = power * (t mod p)
                # This is 0 iff t mod p == 0 iff m is divisible by next_power
                
                # The multiples of power are: power, 2*power, 3*power, ...
                # Among these, those divisible by next_power=p*power are:
                # p*power, 2p*power, 3p*power, ... = next_power, 2*next_power, ...
                
                # So valid indices: arange(power, N_LIMIT+1, power) minus arange(next_power, N_LIMIT+1, next_power)
                all_multiples = np.arange(power, N_LIMIT + 1, power)
                if next_power <= N_LIMIT:
                    excluded = np.arange(next_power, N_LIMIT + 1, next_power)
                    # Use set difference via boolean mask
                    mask = np.zeros(N_LIMIT + 1, dtype=bool)
                    mask[all_multiples] = True
                    mask[excluded] = False
                    indices = np.where(mask)[0]
                else:
                    indices = all_multiples
                
                val[indices] *= power
            
            power = next_power
            next_power *= p
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break

    # 3. Find best n
    best_score = -1
    best_n = 10000
    
    # Use NumPy for vectorized scoring
    scores = val[10000:N_MAX + 1] * val[10001:N_MAX + 2]
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_n = 10000 + best_idx
    
    return int(best_n)

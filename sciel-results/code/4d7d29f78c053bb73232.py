import numpy as np
import time

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def _generate_smooth_numbers(P, limit):
    """
    Generate all P-smooth numbers up to limit.
    P is the largest prime allowed.
    """
    primes = [p for p in range(2, P + 1) if _is_prime(p)]
    
    smooth_nums = {1}
    
    # Determine max powers
    max_powers = []
    for p in primes:
        power = 1
        while power * p <= limit:
            power *= p
        max_powers.append(power)
        
    # Recursive generation
    def gen(idx, current):
        if idx == len(primes):
            if current <= limit:
                smooth_nums.add(current)
            return
        p = primes[idx]
        max_pow = max_powers[idx]
        for i in range(max_pow + 1):
            gen(idx + 1, current * (p ** i))
            
    gen(0, 1)
    return smooth_nums

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds limit to be safe
    
    N_MAX = 1000000
    N_MIN = 10000
    
    # Try P from 2 upwards
    # We stop when we find a solution or run out of time
    # Based on inspiration, P=13 works.
    # Let's try up to P=50 to be safe, but break early if found.
    
    for P in range(2, 51):
        if time.time() > deadline - 5:
            break
            
        smooth_set = _generate_smooth_numbers(P, N_MAX + 1)
        
        # Sort to check for consecutive pairs
        smooth_list = sorted(smooth_set)
        
        for i in range(len(smooth_list) - 1):
            n = smooth_list[i]
            if n > N_MAX:
                break
            if n < N_MIN:
                continue
            if smooth_list[i+1] == n + 1:
                return n

    # Fallback: Brute force with LPF sieve if no smooth pair found in time or up to P=50
    # This ensures we always return something valid
    # Precompute LPF
    N = N_MAX + 1
    lpf = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve for largest prime factor
    for i in range(2, N + 1):
        if lpf[i] == 0: # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    min_val = float('inf')
    best_n = N_MIN
    
    # Check remaining time
    if time.time() > deadline - 2:
        return best_n
        
    for n in range(N_MIN, N_MAX + 1):
        val = max(lpf[n], lpf[n+1])
        if val < min_val:
            min_val = val
            best_n = n
            if min_val == 2:
                return best_n
                
    return best_n

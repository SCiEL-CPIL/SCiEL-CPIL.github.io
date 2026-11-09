import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 70  # safety margin before 90s kill
    
    N = 123457
    
    # Try increasing smoothness bounds B
    # For each B, generate all B-smooth numbers up to N
    # Check if any pair sums to N
    
    def generate_smooth_numbers(B, limit):
        """Generate all numbers <= limit with all prime factors <= B."""
        # First find all primes <= B
        primes = []
        for p in range(2, B + 1):
            is_p = True
            for q in primes:
                if q * q > p:
                    break
                if p % q == 0:
                    is_p = False
                    break
            if is_p:
                primes.append(p)
        
        # Generate smooth numbers using recursive generation
        smooth = set()
        
        def gen(idx, current):
            if current > limit:
                return
            if current > 0:
                smooth.add(current)
            if idx >= len(primes):
                return
            p = primes[idx]
            val = current
            while val <= limit:
                gen(idx + 1, val)
                val *= p
        
        # Start with 1
        smooth.add(1)
        gen(0, 1)
        
        return sorted(smooth)
    
    # Try increasing B values
    B_candidates = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499]
    
    for B in B_candidates:
        if time.time() > deadline:
            break
        
        smooth = generate_smooth_numbers(B, N)
        smooth_set = set(smooth)
        
        # Check if any pair (a, N-a) are both B-smooth
        for a in smooth:
            if a >= N:
                break
            b = N - a
            if b in smooth_set:
                return [a, b]
    
    # Fallback: linear scan with SPF
    spf = list(range(N + 1))
    for i in range(2, isqrt(N) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        n = i
        max_pf = 1
        while n > 1:
            p = spf[n]
            max_pf = p
            while n % p == 0:
                n //= p
        lpf[i] = max_pf
    
    best_val = max(lpf[1], lpf[N - 1])
    best_a = 1
    for a in range(2, N):
        if time.time() > deadline:
            break
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_val:
            best_val = val
            best_a = a
    
    return [best_a, N - best_a]

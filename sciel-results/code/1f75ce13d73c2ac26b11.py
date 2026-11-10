import random
import time
import math

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for overhead/termination
    
    # 1. Generate primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # 2. Efficient phi function
    def phi(x):
        if x == 1:
            return 1
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result
    
    # 3. Batch phi computation for a range - more efficient than individual calls
    def phi_range(start, length):
        """Compute phi for start, start+1, ..., start+length-1"""
        result = []
        for i in range(length):
            result.append(phi(start + i))
        return result

    # 4. Check if a range has all distinct phi values
    def check_range_distinct(start, max_length):
        """Check consecutive integers starting from start, return max distinct run length"""
        seen = set()
        k = 0
        for i in range(max_length):
            val = phi(start + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k

    best_k = 1
    best_s = 1
    max_k = 200
    max_s = 10**12 - max_k + 1
    
    # 5. Search strategy: combine random sampling with targeted exploration
    # Key insight: numbers with few small prime factors tend to have larger, more distinct phi values
    # Focus on regions where numbers are "smooth" in a good way
    
    # First, try some deterministic good spots (based on number theory heuristics)
    # Numbers that are products of large primes or have prime powers tend to have distinct totients
    deterministic_starts = [
        1,
        10**6,
        10**8,
        10**10,
        10**12 - 200,
        999999999999,
        123456789012,
        987654321098,
    ]
    
    for s in deterministic_starts:
        if s < 1 or s > max_s:
            continue
        k = check_range_distinct(s, max_k)
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == max_k:
                return {'start': best_s, 'length': best_k}
    
    # Random search with smarter sampling
    # Weight towards regions with higher density of "good" numbers
    while time.time() < deadline:
        # Pick a random start, but bias towards certain ranges
        # Use logarithmic spacing to cover the range more evenly
        log_min = 0
        log_max = math.log10(max_s)
        log_s = random.uniform(log_min, log_max)
        s = int(10**log_s)
        s = max(1, min(s, max_s))
        
        k = check_range_distinct(s, max_k)
        
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == max_k:
                break
        elif k >= best_k - 2:
            # If close to best, explore neighborhood
            for offset in range(-50, 51, 10):
                s2 = s + offset
                if 1 <= s2 <= max_s:
                    k2 = check_range_distinct(s2, max_k)
                    if k2 > best_k:
                        best_k = k2
                        best_s = s2
                        if best_k == max_k:
                            return {'start': best_s, 'length': best_k}
    
    return {'start': best_s, 'length': best_k}

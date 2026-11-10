import math
import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    # 1. Generate smallest prime factor (SPF) table up to 10^6 for fast factorization
    limit = 10**6
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Fast phi function using SPF
    def phi(n):
        if n <= limit:
            # Use SPF for fast factorization
            result = n
            temp = n
            while temp > 1:
                p = spf[temp]
                result -= result // p
                while temp % p == 0:
                    temp //= p
            return result
        else:
            # For n > limit, use trial division with primes from SPF
            result = n
            temp = n
            for p in range(2, int(temp**0.5) + 1):
                if spf[p] == p:  # p is prime
                    if temp % p == 0:
                        while temp % p == 0:
                            temp //= p
                        result -= result // p
                    if temp == 1:
                        break
            if temp > 1:
                result -= result // temp
            return result
    
    # 3. Search for a valid start s
    max_k = 200
    best_s = 1
    best_k = 1
    
    # Strategy: Focus on regions where numbers are likely to have distinct totients
    # Larger numbers tend to have more distinct prime factors, reducing collisions
    # Try multiple strategies:
    
    # Strategy 1: Random sampling in high ranges (10^11 to 10^12)
    # Strategy 2: Targeted search around known good regions
    # Strategy 3: Quick check of small values as fallback
    
    candidates = set()
    
    # High-range random samples (most likely to find long runs)
    for _ in range(3000):
        s = random.randint(10**11, 10**12 - max_k)
        candidates.add(s)
    
    # Medium-range samples
    for _ in range(1000):
        s = random.randint(10**9, 10**11)
        candidates.add(s)
    
    # Small values as quick fallback
    for s in range(1, 1000):
        candidates.add(s)
    
    # Convert to list and shuffle for variety
    candidate_list = list(candidates)
    random.shuffle(candidate_list)
    
    for s in candidate_list:
        if time.time() > deadline:
            break
        
        seen = set()
        k = 0
        for i in range(max_k):
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == max_k:
                break
        # Early exit if we've found a very good result and time is running low
        if best_k >= 50 and time.time() > deadline - 20:
            break
    
    # Fallback
    if best_s + best_k - 1 > 10**12 or best_k < 1:
        best_s = 1
        best_k = 1

    return {'start': best_s, 'length': best_k}

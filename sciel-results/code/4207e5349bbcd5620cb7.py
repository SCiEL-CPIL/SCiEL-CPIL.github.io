import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit, 5s buffer for safety
    
    MAX_SUM = 20000
    
    # 1. Precompute prime factors for all sums up to MAX_SUM
    # Using Smallest Prime Factor (SPF) sieve
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    pf_cache = [set() for _ in range(MAX_SUM + 1)]
    for i in range(2, MAX_SUM + 1):
        x = i
        while x > 1:
            p = spf[x]
            pf_cache[i].add(p)
            while x % p == 0:
                x //= p
    
    def evaluate(arr):
        primes = set()
        n = len(arr)
        for i in range(n):
            ai = arr[i]
            for j in range(i+1, n):
                s = ai + arr[j]
                primes.update(pf_cache[s])
        return primes

    best_score = float('inf')
    best_arr = None
    
    # Initial Candidates
    candidates = []
    # Small even numbers
    candidates.append(list(range(2, 26, 2)))
    # Multiples of 6
    candidates.append(list(range(6, 79, 6)))
    # Multiples of 30
    candidates.append(list(range(30, 361, 30)))
    # Powers of 2
    candidates.append([2**i for i in range(1, 13)])
    # All 1 mod 30
    candidates.append([1 + 30*i for i in range(12)])
    # All 0 mod 30
    candidates.append([30*i for i in range(1, 13)])
    
    # Random
    for _ in range(100):
        candidates.append(sorted(random.sample(range(1, 10001), 12)))
        
    for arr in candidates:
        if max(arr) > 10000:
            continue
        if len(set(arr)) != 12:
            continue
        primes = evaluate(arr)
        score = len(primes)
        if score < best_score:
            best_score = score
            best_arr = arr[:]
            
    if best_arr is None:
        best_arr = list(range(1, 13))
        
    current_arr = best_arr[:]
    current_score = best_score
    
    # Local Search
    T = 5.0
    cooling = 0.9995
    
    while time.time() < deadline:
        idx = random.randint(0, 11)
        old_val = current_arr[idx]
        
        # Generate new value
        if random.random() < 0.7:
            # Local move
            delta = random.randint(-10, 10)
            new_val = old_val + delta
            if new_val < 1 or new_val > 10000:
                continue
        else:
            # Global move
            new_val = random.randint(1, 10000)
            
        if new_val == old_val:
            continue
            
        # Check distinctness
        if new_val in current_arr:
             continue
             
        current_arr[idx] = new_val
        
        new_primes = evaluate(current_arr)
        new_score = len(new_primes)
        
        if new_score < current_score:
            best_score = new_score
            best_arr = current_arr[:]
            current_score = new_score
        else:
            delta_score = new_score - current_score
            if delta_score > 0:
                prob = math.exp(-delta_score / T)
                if random.random() > prob:
                    current_arr[idx] = old_val
            # If delta_score == 0, we accept
            
        T *= cooling
        if T < 0.05:
            T = 0.05

    return best_arr

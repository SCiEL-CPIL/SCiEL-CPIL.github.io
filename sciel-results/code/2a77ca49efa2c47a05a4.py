import time
import random
import math
import sys

# Precompute Smallest Prime Factors (SPF) for fast factorization
MAX_VAL = 20000
spf = [0] * (MAX_VAL + 1)
for i in range(2, MAX_VAL + 1):
    if spf[i] == 0:
        for j in range(i, MAX_VAL + 1, i):
            if spf[j] == 0:
                spf[j] = i

def get_prime_factors(n):
    factors = set()
    if n < 2:
        return factors
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def count_distinct_primes(arr):
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes.update(get_prime_factors(s))
    return len(primes)

def generate_smooth_numbers(limit, primes):
    """Generate all numbers <= limit composed of the given primes."""
    smooths = {1}
    for p in primes:
        new_smooths = set()
        for s in smooths:
            val = s
            while val * p <= limit:
                val *= p
                new_smooths.add(val)
        smooths.update(new_smooths)
    return sorted([s for s in smooths if s <= limit])

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Generate candidate pool of smooth numbers
    base_primes = [2, 3, 5, 7]
    candidates = generate_smooth_numbers(10000, base_primes)
    
    # Ensure we have enough candidates
    if len(candidates) < 20:
        candidates = list(range(1, 10001))
        
    # 1. Greedy Construction for Initial Solution
    current_arr = []
    current_primes = set()
    available = candidates[:]
    
    for _ in range(12):
        if not available:
            break
        best_candidate = None
        best_prime_count = float('inf')
        best_new_primes = None
        
        # Sample if available is too large to check all
        if len(available) > 200:
            sample = random.sample(available, 200)
        else:
            sample = available
            
        for c in sample:
            # Calculate primes if c is added
            new_primes = set(current_primes)
            for a in current_arr:
                s = a + c
                new_primes.update(get_prime_factors(s))
            
            count = len(new_primes)
            if count < best_prime_count:
                best_prime_count = count
                best_candidate = c
                best_new_primes = new_primes
        
        if best_candidate is not None:
            current_arr.append(best_candidate)
            current_primes = best_new_primes
            # Remove used candidate to ensure distinctness
            if best_candidate in available:
                available.remove(best_candidate)
                
    # If greedy didn't fill 12, fill with random from candidates
    while len(current_arr) < 12:
        if not available:
             # If we ran out of smooth candidates, pick random integers
             new_val = random.randint(1, 10000)
             while new_val in current_arr:
                 new_val = random.randint(1, 10000)
             current_arr.append(new_val)
        else:
             # Pick random from available
             idx = random.randint(0, len(available) - 1)
             c = available[idx]
             current_arr.append(c)
             available.pop(idx)
             # Update current_primes (simple recalc for consistency)
             current_primes = set()
             for i in range(len(current_arr)):
                 for j in range(i+1, len(current_arr)):
                     current_primes.update(get_prime_factors(current_arr[i] + current_arr[j]))

    best_arr = current_arr[:]
    best_cost = count_distinct_primes(best_arr)
    
    # 2. Simulated Annealing
    current_arr = best_arr[:]
    current_cost = best_cost
    T = 10.0
    cooling_rate = 0.9995
    
    while time.time() < deadline:
        if T < 0.01:
            break
            
        idx = random.randint(0, 11)
        old_val = current_arr[idx]
        
        # Generate new value
        if random.random() < 0.7 and candidates:
            new_val = random.choice(candidates)
        else:
            new_val = random.randint(1, 10000)
            
        if new_val == old_val:
            continue
        if new_val in current_arr:
            continue
            
        current_arr[idx] = new_val
        new_cost = count_distinct_primes(current_arr)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_arr = current_arr[:]
            current_cost = new_cost
            # Small temperature boost on improvement
            T = min(T * 1.05, 20.0)
        else:
            delta = new_cost - current_cost
            if delta > 0:
                prob = math.exp(-delta / T)
            else:
                prob = 1.0
                
            if random.random() < prob:
                current_cost = new_cost
            else:
                current_arr[idx] = old_val
                
        T *= cooling_rate
        
    return best_arr
# EVOLVE-BLOCK-END

import time
import random
import math

# Precompute Smallest Prime Factor (SPF) up to 20000
MAX_VAL = 10000
MAX_SUM = 20000
spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM**0.5) + 1):
    if spf[i] == i:  # i is prime
        for j in range(i*i, MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

def get_prime_factors(n):
    """Return set of prime factors of n using SPF."""
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

# Precompute prime factors as bitmasks for all numbers up to MAX_SUM
# We'll use the first 64 primes as our "universe" for bitmask representation
primes_list = []
for i in range(2, MAX_SUM + 1):
    if spf[i] == i:
        primes_list.append(i)
        if len(primes_list) >= 64:
            break

prime_to_bit = {p: (1 << i) for i, p in enumerate(primes_list)}

# Precompute bitmask of prime factors for each number
pf_mask = [0] * (MAX_SUM + 1)
for i in range(2, MAX_SUM + 1):
    mask = 0
    n = i
    while n > 1:
        p = spf[n]
        if p in prime_to_bit:
            mask |= prime_to_bit[p]
        while n % p == 0:
            n //= p
    pf_mask[i] = mask

def count_bits(x):
    """Count set bits in an integer."""
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = None
    best_score = float('inf')
    
    def calc_score_mask(nums):
        """Calculate score using bitmask OR operation."""
        mask = 0
        n = len(nums)
        for i in range(n):
            xi = nums[i]
            for j in range(i+1, n):
                s = xi + nums[j]
                mask |= pf_mask[s]
        return count_bits(mask)
    
    # Initial structured attempts
    inits = [
        list(range(2, 26, 2)),
        list(range(6, 79, 6)),
        [2**i for i in range(1, 13)],
        [3**i for i in range(1, 13)],
        [6*i for i in range(1, 13)],
    ]
    
    for init in inits:
        if all(0 < x <= MAX_VAL for x in init) and len(set(init)) == 12:
            sc = calc_score_mask(init)
            if sc < best_score:
                best_score = sc
                best_set = init[:]
    
    # Random initialization
    current = random.sample(range(1, MAX_VAL + 1), 12)
    current_score = calc_score_mask(current)
    if current_score < best_score:
        best_score = current_score
        best_set = current[:]
    
    # Local search with bitmask-based scoring
    no_improve = 0
    max_no_improve = 500
    
    while no_improve < max_no_improve and time.time() < deadline:
        i = random.randint(0, 11)
        old_val = current[i]
        
        # Generate candidate with bias towards smooth numbers
        if random.random() < 0.3:
            # Try a multiple of a small smooth number
            base = random.choice([2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 48, 60, 72, 90, 120])
            mult = random.randint(1, MAX_VAL // base)
            new_val = base * mult
        else:
            new_val = random.randint(1, MAX_VAL)
            
        if new_val == old_val or new_val in current:
            no_improve += 1
            continue
            
        current[i] = new_val
        new_score = calc_score_mask(current)
        
        if new_score < current_score:
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_set = current[:]
                if best_score <= 3:
                    break
        else:
            current[i] = old_val
            no_improve += 1
    
    # If we haven't found a good solution, try more restarts
    if best_score > 4:
        for r in range(20):
            if time.time() > deadline:
                break
                
            current = random.sample(range(1, MAX_VAL + 1), 12)
            current_score = calc_score_mask(current)
            
            if current_score < best_score:
                best_score = current_score
                best_set = current[:]
                if best_score <= 3:
                    break
            
            no_improve = 0
            max_no_improve = 200
            
            while no_improve < max_no_improve and time.time() < deadline:
                i = random.randint(0, 11)
                old_val = current[i]
                
                if random.random() < 0.4:
                    base = random.choice([2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 48, 60, 72, 90, 120])
                    mult = random.randint(1, MAX_VAL // base)
                    new_val = base * mult
                else:
                    new_val = random.randint(1, MAX_VAL)
                    
                if new_val == old_val or new_val in current:
                    no_improve += 1
                    continue
                    
                current[i] = new_val
                new_score = calc_score_mask(current)
                
                if new_score < current_score:
                    current_score = new_score
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_set = current[:]
                        if best_score <= 3:
                            break
                else:
                    current[i] = old_val
                    no_improve += 1
    
    return best_set

# EVOLVE-BLOCK-END

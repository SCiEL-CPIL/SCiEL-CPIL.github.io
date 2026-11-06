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
    
    # KEY INSIGHT from inspiration: If all numbers are multiples of 6, then all sums are divisible by 6.
    # So prime factors always include 2 and 3. We need to minimize additional primes.
    # Let nums[i] = 6 * k[i], then sum = 6*(k[i]+k[j])
    # Prime factors of sum = {2, 3} ∪ prime_factors(k[i]+k[j])
    # So we need to find 12 distinct k[i] to minimize union of prime_factors(k[i]+k[j])
    
    # Strategy: Search over k values (smaller numbers), then multiply by 6
    # k[i] must be distinct positive integers, and 6*k[i] <= 10000, so k[i] <= 1666
    
    MAX_K = MAX_VAL // 6  # 1666
    
    # Try structured initializations for k values
    k_inits = [
        [2*i for i in range(1, 13)],       # all even
        [2*i+1 for i in range(0, 12)],     # all odd
        list(range(1, 13)),                 # mixed parity
        [6*i for i in range(1, 13)],       # multiples of 6
        [3*i for i in range(1, 13)],       # multiples of 3
        [10*i for i in range(1, 13)],      # multiples of 10
        [12*i for i in range(1, 13)],      # multiples of 12
        [15*i for i in range(1, 13)],      # multiples of 15
    ]
    
    for k_init in k_inits:
        if all(0 < k <= MAX_K for k in k_init) and len(set(k_init)) == 12:
            nums = [6*k for k in k_init]
            sc = calc_score_mask(nums)
            if sc < best_score:
                best_score = sc
                best_set = nums[:]
    
    # Also try some direct initializations
    direct_inits = [
        list(range(2, 26, 2)),
        list(range(6, 79, 6)),
        [2**i for i in range(1, 13)],
        [3**i for i in range(1, 13)],
        [6*i for i in range(1, 13)],
        [12*i for i in range(1, 13)],
        [30*i for i in range(1, 13)],
        [60*i for i in range(1, 13)],
        [120*i for i in range(1, 13)],
    ]
    
    for init in direct_inits:
        if all(0 < x <= MAX_VAL for x in init) and len(set(init)) == 12:
            sc = calc_score_mask(init)
            if sc < best_score:
                best_score = sc
                best_set = init[:]
    
    # Local search on k values with multiple restarts
    # We'll do several restarts with different strategies
    
    def local_search_k(k_init, max_no_improve=500):
        """Local search on k values (multiplied by 6 for actual numbers)."""
        current_k = k_init[:]
        current_score = calc_score_mask([6*k for k in current_k])
        best_k = current_k[:]
        best_sc = current_score
        no_improve = 0
        
        while no_improve < max_no_improve and time.time() < deadline:
            i = random.randint(0, 11)
            old_k = current_k[i]
            
            # Generate new k value with bias towards small, smooth numbers
            r = random.random()
            if r < 0.3:
                # Small even numbers (to keep sums even)
                new_k = random.choice([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 42, 48, 60, 72, 84, 96, 120])
                if random.random() < 0.3:
                    new_k *= random.randint(1, MAX_K // new_k)
            elif r < 0.5:
                # Small odd numbers
                new_k = random.choice([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39])
                if random.random() < 0.3:
                    new_k *= random.randint(1, MAX_K // new_k)
            elif r < 0.7:
                # Multiples of 6
                new_k = 6 * random.randint(1, MAX_K // 6)
            elif r < 0.85:
                # Multiples of 3
                new_k = 3 * random.randint(1, MAX_K // 3)
            else:
                new_k = random.randint(1, MAX_K)
            
            if new_k == old_k or new_k in current_k:
                no_improve += 1
                continue
            
            current_k[i] = new_k
            nums = [6*k for k in current_k]
            new_score = calc_score_mask(nums)
            
            if new_score < current_score:
                current_score = new_score
                no_improve = 0
                if current_score < best_sc:
                    best_sc = current_score
                    best_k = current_k[:]
                    if best_sc <= 4:
                        break
            else:
                current_k[i] = old_k
                no_improve += 1
        
        return best_k, best_sc
    
    # Multiple restarts with different initialization strategies
    restart_strategies = [
        lambda: [2*random.randint(1, 100) for _ in range(12)],           # all even
        lambda: [2*random.randint(1, 100)+1 for _ in range(12)],         # all odd
        lambda: [6*random.randint(1, 277) for _ in range(12)],           # multiples of 6
        lambda: [3*random.randint(1, 555) for _ in range(12)],           # multiples of 3
        lambda: [random.randint(1, MAX_K) for _ in range(12)],           # random
        lambda: [12*random.randint(1, 138) for _ in range(12)],          # multiples of 12
        lambda: [10*random.randint(1, 166) for _ in range(12)],          # multiples of 10
    ]
    
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Choose a strategy
        strategy_idx = random.randint(0, len(restart_strategies) - 1)
        k_init = restart_strategies[strategy_idx]()
        
        # Ensure distinctness
        attempts = 0
        while len(set(k_init)) < 12 and attempts < 100:
            dup_idx = random.randint(0, 11)
            k_init[dup_idx] = random.randint(1, MAX_K)
            attempts += 1
        
        if len(set(k_init)) < 12:
            continue
        
        best_k, best_sc = local_search_k(k_init, max_no_improve=300)
        
        if best_sc < best_score:
            best_score = best_sc
            best_set = [6*k for k in best_k]
            if best_score <= 4:
                break
    
    # If still not good, try direct local search on original values
    if best_score > 5:
        current = random.sample(range(1, MAX_VAL + 1), 12)
        current_score = calc_score_mask(current)
        if current_score < best_score:
            best_score = current_score
            best_set = current[:]
        
        no_improve = 0
        max_no_improve = 500
        
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
                    if best_score <= 4:
                        break
            else:
                current[i] = old_val
                no_improve += 1
    
    return best_set

# EVOLVE-BLOCK-END

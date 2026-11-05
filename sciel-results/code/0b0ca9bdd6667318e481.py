import time
import random
import math

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

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 40.0
    
    max_k = 1666
    
    def is_smooth(n, primes_set):
        while n > 1:
            p = spf[n]
            if p not in primes_set:
                return False
            while n % p == 0:
                n //= p
        return True

    allowed_primes_sets = [
        {2, 3},       
        {2, 3, 5},    
        {2, 3, 5, 7}, 
        {2, 3, 5, 7, 11} 
    ]
    
    for primes_set in allowed_primes_sets:
        if time.time() > deadline - 5:
            break
            
        found = None
        
        # Use a greedy randomized approach with restarts for speed
        # Instead of full backtracking which might be slow in Python if it fails
        
        for _ in range(100): # 100 restarts
            if time.time() > deadline - 2:
                break
                
            current_k_list = []
            # Start with a random small k
            start_k = random.randint(1, 50)
            current_k_list.append(start_k)
            
            success = True
            for target_size in range(2, 13):
                valid_candidates = []
                for k in range(current_k_list[-1] + 1, max_k + 1):
                    valid = True
                    for x in current_k_list:
                        if not is_smooth(k + x, primes_set):
                            valid = False
                            break
                    if valid:
                        valid_candidates.append(k)
                        
                if not valid_candidates:
                    success = False
                    break
                
                # Pick a random candidate, biased towards smaller ones?
                # Smaller ones might leave more room?
                # Or pick the one with highest "degree" in remaining?
                # For now, random from valid candidates.
                # To avoid getting stuck in a dead end, maybe pick the first one?
                # Greedy first-fit is often fast.
                # Let's try picking the first valid candidate.
                chosen_k = valid_candidates[0]
                current_k_list.append(chosen_k)
                
            if success:
                found = current_k_list
                break
                
        if found:
            return [6 * k for k in found]
            
    # Fallback: Local Search
    init_arr = [30 * (i + 1) for i in range(12)]
    best_arr = init_arr[:]
    best_cost = count_distinct_primes(best_arr)
    current_arr = init_arr[:]
    current_cost = best_cost
    
    T = 5.0
    cooling_rate = 0.99
    
    while time.time() < deadline:
        if T < 0.1:
            break
        idx = random.randint(0, 11)
        old_val = current_arr[idx]
        if random.random() < 0.5:
            new_val = random.randint(1, 333) * 30
        else:
            new_val = random.randint(1, 10000)
            
        if new_val in current_arr and new_val != old_val:
            continue
            
        current_arr[idx] = new_val
        new_cost = count_distinct_primes(current_arr)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_arr = current_arr[:]
        else:
            delta = new_cost - current_cost
            if random.random() < math.exp(-delta / T):
                current_cost = new_cost
            else:
                current_arr[idx] = old_val
        T *= cooling_rate
        
    return best_arr
# EVOLVE-BLOCK-END

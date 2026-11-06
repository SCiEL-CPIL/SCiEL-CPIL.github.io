import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # 1. Precompute Smallest Prime Factor (SPF) and Prime Factors up to 20000
    MAX_VAL = 20000
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Cache prime factors for all numbers up to MAX_VAL
    prime_factors_cache = [set() for _ in range(MAX_VAL + 1)]
    prime_factors_cache[1] = set()
    for i in range(2, MAX_VAL + 1):
        n = i
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        prime_factors_cache[i] = factors

    def count_distinct_primes(arr):
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                primes.update(prime_factors_cache[s])
        return len(primes)

    # 2. Simulated Annealing
    best_score = 999
    best_arr = None
    
    # Heuristic Initialization: Multiples of 6
    # This ensures 2 and 3 are always in the prime set.
    # We aim to minimize additional primes.
    
    num_restarts = 20
    steps_per_restart = 2000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with random multiples of 6
        # Range 1..1666 for k, so a = 6k <= 9996
        arr = [6 * random.randint(1, 1666) for _ in range(12)]
        # Ensure distinct
        while len(set(arr)) < 12:
            arr[random.randint(0, 11)] = 6 * random.randint(1, 1666)
        
        current_score = count_distinct_primes(arr)
        if current_score < best_score:
            best_score = current_score
            best_arr = list(arr)
            
        temp = 1.0
        cooling = 0.999
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            i = random.randint(0, 11)
            old_val = arr[i]
            
            # Generate new value
            # Strategy: Mix of local moves (small change) and global moves (random)
            if random.random() < 0.1:
                new_val = 6 * random.randint(1, 1666)
            else:
                # Local move: change k by a small amount
                k = old_val // 6
                delta_k = random.randint(-10, 10)
                new_k = k + delta_k
                if new_k < 1:
                    new_k = 1
                if new_k > 1666:
                    new_k = 1666
                new_val = 6 * new_k
            
            if new_val == old_val:
                continue
            if new_val in arr:
                # Try to find a distinct one nearby
                found = False
                for _ in range(5):
                    new_val = 6 * random.randint(1, 1666)
                    if new_val not in arr and new_val != old_val:
                        found = True
                        break
                if not found:
                    continue
                    
            arr[i] = new_val
            new_score = count_distinct_primes(arr)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_arr = list(arr)
                # Cool down
                temp *= cooling
            else:
                delta = new_score - current_score
                if delta > 0:
                    prob = math.exp(-delta / temp)
                    if random.random() < prob:
                        current_score = new_score
                        temp *= cooling
                    else:
                        arr[i] = old_val
                else:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_arr = list(arr)
            
            if current_score == 3:
                # 3 is likely the theoretical minimum for this structure (2,3,5)
                # We can stop early if we find a very good solution
                # But let's run a bit more to see if we can get 2? 
                # 2 primes means sums are only 2,3 or 2,5 or 3,5.
                # If sums are multiples of 6, they have 2,3. 
                # If all sums are powers of 2 times 3? 
                # Unlikely to be < 3.
                pass

    if best_arr:
        return best_arr
    else:
        # Fallback
        return list(range(1, 13))
# EVOLVE-BLOCK-END

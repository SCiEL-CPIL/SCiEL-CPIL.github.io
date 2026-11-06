import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit to be safe within 90s
    
    # 1. Pre-compute prime factors for all numbers up to 20000
    # Max sum is 10000 + 9999 = 19999. Let's go up to 20000.
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1)) # Smallest Prime Factor
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i: # i is prime
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Pre-compute set of prime factors for each number
    prime_factors = [set() for _ in range(MAX_SUM + 1)]
    for i in range(2, MAX_SUM + 1):
        x = i
        while x > 1:
            p = spf[x]
            prime_factors[i].add(p)
            while x % p == 0:
                x //= p
    
    # 2. Define candidate pool
    # Using multiples of 6 ensures 2 and 3 are in the factor set of all sums.
    # This reduces the problem to minimizing other primes.
    # Pool size ~1666
    pool = list(range(6, 10001, 6))
    
    # 3. Evaluation function
    def evaluate(a_list):
        union_primes = set()
        n = len(a_list)
        for i in range(n):
            ai = a_list[i]
            for j in range(i+1, n):
                s = ai + a_list[j]
                union_primes.update(prime_factors[s])
        return len(union_primes)
    
    best_score = float('inf')
    best_set = None
    
    # 4. Search Loop
    while time.time() < deadline:
        # Random restart with local search
        # Initialize with a random subset of the pool
        current_set = random.sample(pool, 12)
        current_score = evaluate(current_set)
        
        if current_score < best_score:
            best_score = current_score
            best_set = current_set[:]
            if best_score <= 3: # Theoretical lower bound for multiples of 6 is 2 (only 2,3) but sums > 12 usually introduce others. 3 is {2,3,5}. 4 is {2,3,5,7}.
                # We can break early if we hit a very good score, but let's keep searching a bit to be sure or find even better?
                # Actually, if we find 3, it's likely optimal for this pool structure.
                # Let's continue for a bit longer to see if 2 is possible (unlikely) or just return.
                # Given time constraints, if we find 3, we can probably stop or do a very fast polish.
                # But 2 is impossible for distinct positive integers > 0?
                # If sums are only 2,3 primes, sums are $2^a 3^b$.
                # $a_i + a_j = 2^a 3^b$.
                # This is a very restrictive condition.
                # Let's just record it and continue if time permits, but 3 is a great result.
                pass

        # Local Search (Hill Climbing with some noise)
        no_improvement_count = 0
        last_improvement_time = time.time()
        
        while time.time() < deadline:
            # Check if we should break out of local search to restart
            if time.time() - last_improvement_time > 2.0:
                break
            
            # Pick a random index to swap
            i = random.randint(0, 11)
            old_val = current_set[i]
            
            # Pick a random new value from pool
            # To avoid picking the same value, we can sample from a filtered list or just check
            # Sampling from pool is O(1) but might collide.
            # Given pool size 1666 and set size 12, collision probability is low.
            new_val = random.choice(pool)
            
            if new_val == old_val:
                continue
            
            # Check if new_val is already in the set
            if new_val in current_set:
                continue
            
            # Perform swap
            current_set[i] = new_val
            
            new_score = evaluate(current_set)
            
            # Accept if better or equal (to allow plateaus)
            # Or with small probability if worse (annealing)
            # Let's stick to hill climbing for speed, restarts handle local minima.
            if new_score <= current_score:
                current_score = new_score
                no_improvement_count = 0
                last_improvement_time = time.time()
                
                if current_score < best_score:
                    best_score = current_score
                    best_set = current_set[:]
            else:
                # Revert
                current_set[i] = old_val
                no_improvement_count += 1
                
                # If stuck for too long, break to restart
                if no_improvement_count > 100:
                    break
                    
    # Return the best set found
    if best_set:
        return best_set
    else:
        # Fallback
        return list(range(1, 13))

# EVOLVE-BLOCK-END

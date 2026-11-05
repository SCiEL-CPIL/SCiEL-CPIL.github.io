import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    # 1. Generate pool of 5-smooth numbers (primes 2, 3, 5) up to 10000
    pool = set()
    for a in range(0, 15):
        v2 = 2 ** a
        if v2 > 10000:
            break
        for b in range(0, 15):
            v23 = v2 * (3 ** b)
            if v23 > 10000:
                break
            for c in range(0, 15):
                val = v23 * (5 ** c)
                if val > 10000:
                    break
                pool.add(val)
    pool = sorted(list(pool))
    pool_size = len(pool)
    
    # Precompute smallest prime factor (SPF) up to 20000
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Map pool values to indices for fast lookup
    val_to_idx = {v: i for i, v in enumerate(pool)}
    
    # Precompute factors for all pairs in the pool
    pair_primes = [[None for _ in range(pool_size)] for _ in range(pool_size)]
    
    for i in range(pool_size):
        vi = pool[i]
        for j in range(i + 1, pool_size):
            vj = pool[j]
            s = vi + vj
            factors = set()
            n = s
            while n > 1:
                p = spf[n]
                factors.add(p)
                while n % p == 0:
                    n //= p
            # Store as tuple for hashability and faster iteration
            f_tuple = tuple(factors)
            pair_primes[i][j] = f_tuple
            pair_primes[j][i] = f_tuple
            
    # If time is running out during precomputation, abort and use a simpler method
    if time.time() > deadline - 5.0:
         # Fallback: just return a random set
         return random.sample(pool, 12)
         
    # Initialize with a structured subset: take the 12 smallest 5-smooth numbers
    # This gives a good starting point since small numbers have fewer prime factors in sums
    current_set = pool[:12]
    current_idx_set = list(range(12))
    
    # Maintain prime counts
    prime_counts = {}
    
    # Initialize counts
    for i in range(12):
        for j in range(i + 1, 12):
            idx_i = current_idx_set[i]
            idx_j = current_idx_set[j]
            primes = pair_primes[idx_i][idx_j]
            for p in primes:
                if p in prime_counts:
                    prime_counts[p] += 1
                else:
                    prime_counts[p] = 1
                    
    current_score = len(prime_counts)
    best_score = current_score
    best_set = current_set[:]
    best_idx_set = current_idx_set[:]
    
    # Simulated Annealing with improved parameters
    temp = 2.0  # Start hotter to allow more exploration
    cooling = 0.99995
    min_temp = 0.005
    
    restarts = 0
    max_restarts = 8
    
    # Track how many steps we've done
    step_count = 0
    
    while time.time() < deadline:
        if restarts > max_restarts:
            break
            
        # Pick random index to swap
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        old_idx = current_idx_set[idx]
        
        # Smart move: pick a new value that's close in magnitude to old_val
        # This helps maintain smoothness of the set
        # 70% chance to pick nearby, 30% chance to pick anywhere
        if random.random() < 0.7:
            # Pick from a window around old_idx
            window_size = max(5, pool_size // 10)
            low = max(0, old_idx - window_size)
            high = min(pool_size - 1, old_idx + window_size)
            new_idx = random.randint(low, high)
        else:
            new_idx = random.randint(0, pool_size - 1)
            
        new_val = pool[new_idx]
        
        if new_idx == old_idx:
            continue
        if new_val in current_set:
            continue
            
        # Apply change: remove old_idx, add new_idx
        # 1. Remove contributions of old_idx with all other elements
        for k in range(12):
            if k == idx:
                continue
            other_idx = current_idx_set[k]
            primes = pair_primes[old_idx][other_idx]
            for p in primes:
                prime_counts[p] -= 1
                if prime_counts[p] == 0:
                    del prime_counts[p]
                    
        # 2. Add contributions of new_idx with all other elements
        for k in range(12):
            if k == idx:
                continue
            other_idx = current_idx_set[k]
            primes = pair_primes[new_idx][other_idx]
            for p in primes:
                if p in prime_counts:
                    prime_counts[p] += 1
                else:
                    prime_counts[p] = 1
                    
        # Update current state
        current_set[idx] = new_val
        current_idx_set[idx] = new_idx
        new_score = len(prime_counts)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current_set[:]
                best_idx_set = current_idx_set[:]
        else:
            # Accept worse solution with probability p
            delta = new_score - current_score
            if delta < 0:
                accept_prob = 1.0
            else:
                accept_prob = math.exp(-delta / temp)
            
            if random.random() < accept_prob:
                current_score = new_score
            else:
                # Revert
                # Remove new_idx contributions
                for k in range(12):
                    if k == idx:
                        continue
                    other_idx = current_idx_set[k]
                    primes = pair_primes[new_idx][other_idx]
                    for p in primes:
                        prime_counts[p] -= 1
                        if prime_counts[p] == 0:
                            del prime_counts[p]
                            
                # Add old_idx contributions back
                for k in range(12):
                    if k == idx:
                        continue
                    other_idx = current_idx_set[k]
                    primes = pair_primes[old_idx][other_idx]
                    for p in primes:
                        if p in prime_counts:
                            prime_counts[p] += 1
                        else:
                            prime_counts[p] = 1
                            
                current_set[idx] = old_val
                current_idx_set[idx] = old_idx
                # current_score remains unchanged
                
        temp *= cooling
        step_count += 1
        
        if temp < min_temp:
            temp = min_temp
            restarts += 1
            # Restart with a new random set
            current_set = random.sample(pool, 12)
            current_idx_set = [val_to_idx[v] for v in current_set]
            prime_counts = {}
            for i in range(12):
                for j in range(i + 1, 12):
                    idx_i = current_idx_set[i]
                    idx_j = current_idx_set[j]
                    primes = pair_primes[idx_i][idx_j]
                    for p in primes:
                        if p in prime_counts:
                            prime_counts[p] += 1
                        else:
                            prime_counts[p] = 1
            current_score = len(prime_counts)
            if current_score < best_score:
                best_score = current_score
                best_set = current_set[:]
                best_idx_set = current_idx_set[:]
            temp = 2.0  # Reset to hot temperature
    
    # Final local search pass: try to improve best_set by single swaps
    if time.time() < deadline - 2.0:
        local_best = best_set[:]
        local_idx_set = best_idx_set[:]
        local_prime_counts = {}
        
        # Initialize local counts
        for i in range(12):
            for j in range(i + 1, 12):
                idx_i = local_idx_set[i]
                idx_j = local_idx_set[j]
                primes = pair_primes[idx_i][idx_j]
                for p in primes:
                    if p in local_prime_counts:
                        local_prime_counts[p] += 1
                    else:
                        local_prime_counts[p] = 1
                        
        local_score = len(local_prime_counts)
        
        # Try each position, try to find a better swap
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for idx in range(12):
                old_idx = local_idx_set[idx]
                old_val = local_best[idx]
                
                # Try swapping with each pool element
                best_new_idx = -1
                best_new_score = local_score
                
                for new_idx in range(pool_size):
                    if new_idx == old_idx:
                        continue
                    new_val = pool[new_idx]
                    if new_val in local_best:
                        continue
                    
                    # Calculate score change
                    # Remove old_idx contributions
                    temp_counts = dict(local_prime_counts)
                    for k in range(12):
                        if k == idx:
                            continue
                        other_idx = local_idx_set[k]
                        primes = pair_primes[old_idx][other_idx]
                        for p in primes:
                            temp_counts[p] -= 1
                            if temp_counts[p] == 0:
                                del temp_counts[p]
                    
                    # Add new_idx contributions
                    for k in range(12):
                        if k == idx:
                            continue
                        other_idx = local_idx_set[k]
                        primes = pair_primes[new_idx][other_idx]
                        for p in primes:
                            if p in temp_counts:
                                temp_counts[p] += 1
                            else:
                                temp_counts[p] = 1
                                
                    candidate_score = len(temp_counts)
                    
                    if candidate_score < best_new_score:
                        best_new_score = candidate_score
                        best_new_idx = new_idx
                
                # Apply best swap if found
                if best_new_idx != -1 and best_new_score < local_score:
                    old_idx = local_idx_set[idx]
                    
                    # Remove old_idx contributions
                    for k in range(12):
                        if k == idx:
                            continue
                        other_idx = local_idx_set[k]
                        primes = pair_primes[old_idx][other_idx]
                        for p in primes:
                            local_prime_counts[p] -= 1
                            if local_prime_counts[p] == 0:
                                del local_prime_counts[p]
                    
                    # Add new_idx contributions
                    for k in range(12):
                        if k == idx:
                            continue
                        other_idx = local_idx_set[k]
                        primes = pair_primes[best_new_idx][other_idx]
                        for p in primes:
                            if p in local_prime_counts:
                                local_prime_counts[p] += 1
                            else:
                                local_prime_counts[p] = 1
                                
                    local_best[idx] = pool[best_new_idx]
                    local_idx_set[idx] = best_new_idx
                    local_score = best_new_score
                    improved = True
                    
                    if local_score < best_score:
                        best_score = local_score
                        best_set = local_best[:]
                        best_idx_set = local_idx_set[:]
        
    return best_set

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# EVOLVE-BLOCK-END

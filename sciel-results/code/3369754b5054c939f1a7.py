import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 256
    
    # Precompute factor masks
    primes = []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    
    factor_mask = [0] * (n + 1)
    for i in range(2, n + 1):
        mask = 0
        x = i
        for p in primes:
            if p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_to_bit[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_to_bit[x])
        factor_mask[i] = mask
    
    # factor_mask[1] = 0

    def is_compatible(val, current_set):
        fm = factor_mask[val]
        if fm == 0:
            return True
        for v in current_set:
            if (fm & factor_mask[v]) != 0:
                return False
        return True

    def local_search(initial_set, time_limit):
        best_set = set(initial_set)
        best_sum = sum(best_set)
        candidates = list(range(n, 0, -1))
        end_time = time.time() + time_limit
        
        improved = True
        while improved and time.time() < end_time:
            improved = False
            
            # Phase 1: Add
            for val in candidates:
                if time.time() >= end_time:
                    break
                if val in best_set:
                    continue
                if is_compatible(val, best_set):
                    best_set.add(val)
                    best_sum += val
                    improved = True
            
            # Phase 2: Swap 1-for-1
            for remove_val in list(best_set):
                if time.time() >= end_time:
                    break
                reduced_set = best_set - {remove_val}
                reduced_sum = best_sum - remove_val
                
                for add_val in candidates:
                    if time.time() >= end_time:
                        break
                    if add_val in reduced_set:
                        continue
                    if is_compatible(add_val, reduced_set):
                        new_sum = reduced_sum + add_val
                        if new_sum > best_sum:
                            best_set = reduced_set
                            best_set.add(add_val)
                            best_sum = new_sum
                            improved = True
                            break
                            
            # Phase 3: Swap 1-for-2 (Remove 1, Add 2)
            if time.time() < end_time:
                for remove_val in list(best_set):
                    if time.time() >= end_time:
                        break
                    reduced_set = best_set - {remove_val}
                    reduced_sum = best_sum - remove_val
                    
                    # Try to add two
                    for add1 in candidates:
                        if time.time() >= end_time:
                            break
                        if add1 in reduced_set:
                            continue
                        if not is_compatible(add1, reduced_set):
                            continue
                        
                        temp_set = reduced_set | {add1}
                        temp_sum = reduced_sum + add1
                        
                        for add2 in candidates:
                            if time.time() >= end_time:
                                break
                            if add2 in temp_set:
                                continue
                            if not is_compatible(add2, temp_set):
                                continue
                            
                            new_sum = temp_sum + add2
                            if new_sum > best_sum:
                                best_set = temp_set
                                best_set.add(add2)
                                best_sum = new_sum
                                improved = True
                                break

        return best_set, best_sum

    # Generate multiple initial solutions
    
    # 1. Prime Power Heuristic
    init_set_1 = {1}
    for p in primes:
        power = p
        while power * p <= n:
            power *= p
        if is_compatible(power, init_set_1):
            init_set_1.add(power)
            
    # 2. Greedy Largest Heuristic
    init_set_2 = {1}
    for val in range(n, 1, -1):
        if is_compatible(val, init_set_2):
            init_set_2.add(val)
            
    best_overall_set = set()
    best_overall_sum = 0
    
    # Allocate time: 40% for Init 1, 40% for Init 2, 20% for Random Restarts
    total_time = 80.0
    t1 = total_time * 0.4
    t2 = total_time * 0.4
    t3 = total_time * 0.2
    
    # Run local search on Init 1
    s1, sum1 = local_search(init_set_1, t1)
    if sum1 > best_overall_sum:
        best_overall_set = s1
        best_overall_sum = sum1
        
    # Run local search on Init 2
    s2, sum2 = local_search(init_set_2, t2)
    if sum2 > best_overall_sum:
        best_overall_set = s2
        best_overall_sum = sum2
        
    # Random Restarts
    if time.time() < deadline - 5.0:
        # Estimate time per restart
        # We have ~16s left. Let's do up to 5 restarts of 3s each?
        # Or just loop until time is up, but ensure local_search gets enough time.
        # If we call local_search with a small time limit, it might not converge.
        # Better to do fewer restarts with more time.
        
        # Let's do 2 more restarts with 8s each
        for _ in range(2):
            if time.time() >= deadline - 5.0:
                break
            
            rand_set = {1}
            cands = list(range(n, 1, -1))
            random.shuffle(cands)
            
            for val in cands:
                if is_compatible(val, rand_set):
                    rand_set.add(val)
                    
            rs, rsum = local_search(rand_set, 8.0)
            if rsum > best_overall_sum:
                best_overall_set = rs
                best_overall_sum = rsum

    return [int(x) for x in best_overall_set]

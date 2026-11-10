import time
import random

def _factor_primes(n):
    primes = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            primes.add(d)
            n //= d
        d += 1
    if n > 1:
        primes.add(n)
    return primes

def _is_smooth_fast(n, primes):
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1

def construct():
    start_time = time.time()
    best_set = None
    best_count = float('inf')
    
    # Candidate prime sets to try, in order of size
    prime_candidates = [
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13]
    ]
    
    def evaluate(nums):
        nonlocal best_set, best_count
        all_primes = set()
        for i in range(12):
            for j in range(i + 1, 12):
                all_primes |= _factor_primes(nums[i] + nums[j])
        count = len(all_primes)
        if count < best_count:
            best_count = count
            best_set = list(nums)

    # For each prime set, try to find a configuration where all sums are smooth
    for P in prime_candidates:
        if time.time() - start_time > 50:
            break
            
        # Pre-compute smoothness for potential sums? 
        # Max sum is 20000.
        # We can just use _is_smooth_fast.
        
        # Local Search / Simulated Annealing
        # Initial guess: random numbers
        # We want to minimize violations
        
        def count_violations(x, P):
            v = 0
            for i in range(12):
                for j in range(i + 1, 12):
                    if not _is_smooth_fast(x[i] + x[j], P):
                        v += 1
            return v
        
        # Try multiple restarts
        found = False
        for restart in range(200):
            if time.time() - start_time > 50:
                break
                
            # Random initialization
            x = random.sample(range(1, 10001), 12)
            current_cost = count_violations(x, P)
            
            if current_cost == 0:
                evaluate(x)
                found = True
                break
            
            # Simulated Annealing
            T = 100.0
            for step in range(5000):
                if time.time() - start_time > 50:
                    break
                    
                # Pick a random index to change
                k = random.randint(0, 11)
                old_val = x[k]
                # Try a few random new values
                for _ in range(10):
                    new_val = random.randint(1, 10000)
                    if new_val == old_val:
                        continue
                    # Check uniqueness
                    if new_val in x:
                        continue
                    
                    # Calculate cost change
                    # Only need to check pairs involving k
                    delta_cost = 0
                    for i in range(12):
                        if i == k:
                            continue
                        old_sum = old_val + x[i]
                        new_sum = new_val + x[i]
                        
                        old_smooth = _is_smooth_fast(old_sum, P)
                        new_smooth = _is_smooth_fast(new_sum, P)
                        
                        if not old_smooth and not new_smooth:
                            pass
                        elif old_smooth and not new_smooth:
                            delta_cost += 1
                        elif not old_smooth and new_smooth:
                            delta_cost -= 1
                        else:
                            pass
                    
                    new_cost = current_cost + delta_cost
                    
                    if new_cost < current_cost:
                        x[k] = new_val
                        current_cost = new_cost
                        if current_cost == 0:
                            evaluate(x)
                            found = True
                            break
                    else:
                        # Accept with probability exp(-delta/T)
                        if random.random() < pow(2.718, -delta_cost / T):
                            x[k] = new_val
                            current_cost = new_cost
                            if current_cost == 0:
                                evaluate(x)
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
                
                T *= 0.99
                if T < 0.1:
                    break
        
        if found:
            # If we found a solution for this P, it's likely optimal for this prime set size
            # But we should still check if a larger P allows a smaller *actual* prime union?
            # No, if sums are P-smooth, actual union is subset of P.
            # If we found a solution for P={2,3,5}, the union is <= 3.
            # If we found a solution for P={2,3,5,7}, the union is <= 4.
            # We want the minimum union.
            # If we found a solution for P={2,3,5}, we can return immediately?
            # Not necessarily, because the actual union might be {2,3} (size 2)?
            # If actual union is {2,3}, then all sums are 3-smooth.
            # So if we find a solution for P={2,3,5}, we evaluate it.
            # If best_count becomes 2 or 3, that's very good.
            # We should continue to next P only if we didn't find a solution for current P?
            # Or if we found a solution but want to see if a different P yields a smaller union?
            # A solution for P={2,3,5} has union subset of {2,3,5}.
            # A solution for P={2,3,5,7} has union subset of {2,3,5,7}.
            # It is possible that a P={2,3,5,7} solution has union {2,3} (size 2) while no P={2,3,5} solution exists?
            # If union is {2,3}, all sums are 3-smooth. 3-smooth implies 5-smooth.
            # So if a solution with union {2,3} exists, it is also a solution for P={2,3,5}.
            # So if we fail to find a P={2,3,5} solution, we cannot have a solution with union subset of {2,3,5}.
            # Therefore, if we find a solution for P={2,3,5}, we have the best possible union size for that set of primes.
            # We should stop if best_count == 2? Or 3?
            # The minimum possible number of distinct primes is likely 2 or 3.
            # If we find best_count == 2, we can stop.
            if best_count <= 2:
                break

    if best_set is None:
        # Fallback
        return list(range(1, 13))
        
    return best_set

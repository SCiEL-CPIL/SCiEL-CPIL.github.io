import time
import random
from itertools import combinations

# Precompute smallest prime factor for numbers up to 20000 (max sum = 2*10000)
MAX_SUM = 20000
spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM**0.5) + 1):
    if spf[i] == i:
        for j in range(i*i, MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

# Precompute prime factors for all numbers up to MAX_SUM
factors_cache = [set() for _ in range(MAX_SUM + 1)]
for i in range(2, MAX_SUM + 1):
    n = i
    while n > 1:
        p = spf[n]
        factors_cache[i].add(p)
        while n % p == 0:
            n //= p

def count_distinct_primes(arr):
    """Count distinct prime factors across all pairwise sums of distinct elements."""
    all_primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            all_primes |= factors_cache[s]
    return len(all_primes)

def is_smooth(n, target_primes):
    """Check if n is smooth with respect to target_primes."""
    if n <= 1:
        return True
    return factors_cache[n].issubset(target_primes)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_score = 999
    best_arr = list(range(1, 13)) # Fallback
    
    # List of first few primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Try to find a solution with a specific number of primes k
    # We iterate k from 6 upwards. If we find a valid set for k, we return it.
    # This guarantees a score of at least 9/k.
    
    for k in range(6, 10):
        if time.time() > deadline:
            break
            
        # Try different combinations of primes of size k
        # We prefer smaller primes as they are more likely to be factors of sums
        prime_indices = list(range(min(15, len(primes))))
        
        # To avoid combinatorial explosion, we don't try all C(15, k).
        # We try a few promising sets.
        # Set 1: First k primes
        # Set 2: First k-1 primes + 1 larger
        # Set 3: Random subsets
        
        target_sets = set()
        
        # Add the "first k" set
        target_sets.add(frozenset(primes[:k]))
        
        # Add a few random sets
        random.seed(42)
        for _ in range(5):
            selected_indices = random.sample(prime_indices, k)
            target_sets.add(frozenset(primes[i] for i in selected_indices))
        
        for target_primes in target_sets:
            if time.time() > deadline:
                break
            
            # Generate candidate pool: numbers <= 10000 that are smooth w.r.t target_primes
            # Note: The numbers themselves don't have to be smooth, but it helps.
            # Actually, if x is not smooth, x+y might be smooth.
            # But if we restrict x to be smooth, we increase the chance that x+y is smooth?
            # Not necessarily. 3 (smooth) + 5 (smooth) = 8 (smooth).
            # 3 (smooth) + 7 (smooth) = 10 (smooth).
            # 3 (smooth) + 11 (smooth) = 14 (smooth).
            # It seems picking smooth numbers is a good heuristic.
            
            candidates = []
            for num in range(1, 10001):
                if is_smooth(num, target_primes):
                    candidates.append(num)
            
            if len(candidates) < 12:
                continue
            
            # Local Search to find 12 distinct candidates such that all pairwise sums are smooth
            # Cost function: Number of distinct "bad" primes in the sums.
            # Bad prime: p in factors(sum) such that p not in target_primes.
            
            # Initialize with random selection
            random.shuffle(candidates)
            current_arr = candidates[:12]
            
            # Calculate initial cost
            def calc_cost(arr):
                bad_primes_set = set()
                n = len(arr)
                for i in range(n):
                    for j in range(i+1, n):
                        s = arr[i] + arr[j]
                        # Get factors of s
                        fs = factors_cache[s]
                        # Find intersection with complement of target_primes
                        # Or just check if any factor is not in target_primes
                        for p in fs:
                            if p not in target_primes:
                                bad_primes_set.add(p)
                return len(bad_primes_set)
            
            current_cost = calc_cost(current_arr)
            
            if current_cost == 0:
                # Found a valid set!
                best_score = len(target_primes)
                best_arr = current_arr[:]
                return best_arr # We can return immediately as this is optimal for this k
            
            # Simulated Annealing / Hill Climbing
            temp = 100.0
            cooling = 0.95
            min_temp = 0.1
            
            while temp > min_temp and time.time() < deadline:
                # Pick a random index to change
                idx = random.randint(0, 11)
                old_val = current_arr[idx]
                
                # Pick a new value from candidates
                # To speed up, we can pick from a subset of candidates
                new_val = random.choice(candidates)
                
                if new_val == old_val:
                    continue
                    
                # Check distinctness
                # We need to ensure new_val is not in current_arr (except at idx)
                # Since we are picking from candidates, and current_arr is a subset of candidates,
                # we must check if new_val is already in current_arr at a different index.
                is_dup = False
                for j in range(12):
                    if j != idx and current_arr[j] == new_val:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                current_arr[idx] = new_val
                
                # Calculate new cost
                # Optimization: Only recalculate pairs involving idx
                # But for simplicity and given N=12, full recalc is ~66 ops.
                # 66 * (set ops) is fast.
                new_cost = calc_cost(current_arr)
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < pow(2.718, -delta / temp):
                    current_cost = new_cost
                    # Update best for this target set
                    if current_cost == 0:
                        best_score = len(target_primes)
                        best_arr = current_arr[:]
                        return best_arr
                else:
                    current_arr[idx] = old_val
                
                temp *= cooling
        
        # If we didn't return, we didn't find a perfect solution for this k
        # But we might have found a set with few bad primes?
        # The problem asks to minimize the number of DISTINCT primes.
        # If we use target_primes of size k, and the actual set has k + m bad primes,
        # the total distinct primes is k + m.
        # We should track the best actual score found.
        
        # Let's evaluate the best_arr found in this loop (if any)
        # Actually, if we didn't return, it means cost > 0.
        # We should just continue to the next k?
        # Or should we keep the best "almost smooth" set?
        # If k=8 and cost=1, total primes = 9. Score 9/9=1.0.
        # If k=7 and cost=1, total primes = 8. Score 9/8=1.125.
        # So we should evaluate the actual score of the best candidate found.
        
        # For now, let's just let the loop continue to higher k if we didn't find a 0-cost solution.
        # But we should update best_score if we find a set with fewer distinct primes than current best.
        
        # Let's add a general local search at the end if no specific target worked perfectly.
    
    # Fallback: General Local Search to minimize distinct prime count
    # Use the best_arr from above or a random start
    if best_score == 999:
        best_arr = list(range(1, 13))
        best_score = count_distinct_primes(best_arr)
        
    # Perform a general local search to improve best_arr
    random.seed(42)
    current_arr = best_arr[:]
    current_score = best_score
    
    for _ in range(500):
        if time.time() > deadline:
            break
        idx = random.randint(0, 11)
        old_val = current_arr[idx]
        new_val = random.randint(1, 10000)
        if new_val in current_arr and new_val != old_val:
            continue
        current_arr[idx] = new_val
        score = count_distinct_primes(current_arr)
        if score < current_score:
            current_score = score
            if score < best_score:
                best_score = score
                best_arr = current_arr[:]
        else:
            current_arr[idx] = old_val
            
    return best_arr

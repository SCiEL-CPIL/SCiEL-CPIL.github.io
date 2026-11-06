import random
import math
import sys
from itertools import combinations

def construct():
    MAX_VAL = 10000
    MAX_SUM = 2 * MAX_VAL  # 20000
    N = 12

    # Precompute smallest prime factor for all numbers up to MAX_SUM
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def get_prime_factors(n):
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    # Precompute prime factors for all sums up to MAX_SUM
    pf_cache = [set() for _ in range(MAX_SUM + 1)]
    for s in range(2, MAX_SUM + 1):
        pf_cache[s] = get_prime_factors(s)

    def objective(nums):
        prime_set = set()
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                s = nums[i] + nums[j]
                prime_set |= pf_cache[s]
        return len(prime_set)

    def objective_with_set(nums):
        prime_set = set()
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                s = nums[i] + nums[j]
                prime_set |= pf_cache[s]
        return len(prime_set), prime_set

    best = None
    best_score = float('inf')

    # Strategy: Try to find 12 numbers where all pairwise sums have factors from a small prime set P.
    # If we fix a small set of primes P, we can look for numbers such that a_i + a_j is P-smooth (or has only primes from P) for all i != j.
    
    # Candidate prime sets to try.
    # We want the union of prime factors to be small.
    # Let's try sets of size 2, 3, 4, 5.
    
    # If P = {2, 3}, sums must be of form 2^a * 3^b.
    # If P = {2, 3, 5}, sums must be 5-smooth.
    # If P = {2, 3, 5, 7}, sums must be 7-smooth.
    
    # Let's generate candidate prime sets.
    # Primes up to 50: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # We will try to find a set of 12 numbers for a given prime set P.
    # If such a set exists, the score is |P| (or a subset thereof).
    # We iterate over candidate P sets.
    
    # To speed up, we can precompute for a given P, which numbers x in [1, 10000] are "P-good" with respect to a set of already chosen numbers?
    # Actually, the condition is: for all i, j, a_i + a_j is P-smooth.
    # This is a strong constraint.
    
    # Let's try a different approach:
    # Randomly sample numbers, compute the prime set.
    # Then, try to "repair" the set by replacing numbers that introduce "bad" primes.
    
    # A "bad" prime is one that appears in the sum with a high frequency or is the only reason the count is high.
    
    # Let's combine the structured approach with a more aggressive local search.
    
    # Initial candidates from structured patterns
    candidates = []
    
    # Pattern: a_i = 210 * k_i
    # k_i = 1..12
    candidates.append([210 * k for k in range(1, 13)])
    
    # Pattern: a_i = 210 * k_i + 105
    # u_i = 2, 5, 8, ... (2 mod 3)
    us = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
    candidates.append([210 * (u - 1) + 105 for u in us])
    
    # Pattern: a_i = 210 * k_i, k_i even
    candidates.append([210 * k for k in range(2, 25, 2)])
    
    # Pattern: a_i = 210 * k_i, k_i odd
    candidates.append([210 * k for k in range(1, 24, 2)])
    
    # Pattern: a_i = 210 * k_i + 105, u_i even
    us = list(range(2, 25, 2))
    candidates.append([210 * (u - 1) + 105 for u in us])
    
    # Pattern: a_i = 210 * k_i + 105, u_i odd
    us = list(range(1, 24, 2))
    candidates.append([210 * (u - 1) + 105 for u in us])
    
    # Add some random candidates
    random.seed(42)
    for _ in range(20):
        candidates.append(random.sample(range(1, MAX_VAL + 1), 12))
    
    for cand in candidates:
        if all(1 <= x <= MAX_VAL for x in cand) and len(set(cand)) == 12:
            score = objective(cand)
            if score < best_score:
                best_score = score
                best = cand[:]

    # Local Search: Hill Climbing with Restart and "Prime Removal" heuristic
    # We want to minimize the number of distinct primes.
    # If we can identify a prime p that is "expensive" (appears in many sums), we can try to replace numbers to remove p.
    
    # For each restart, we start from the best found so far (or a perturbed version).
    # We try to swap one number at a time.
    # To make it faster, we can precompute the contribution of each number to the prime set.
    
    # Let's implement a smarter local search.
    # State: list of 12 numbers.
    # Move: replace one number with another.
    # Cost: number of distinct primes.
    
    # To evaluate the cost change efficiently:
    # Let S be the current set of primes.
    # Let C_p be the count of sums involving prime p.
    # If we replace a_i with a_i', the sums involving a_i change.
    # The old sums were a_i + a_j for j != i.
    # The new sums are a_i' + a_j for j != i.
    # We need to update S and C_p.
    
    # This is still O(N) per move.
    # With N=12, it's fine.
    
    def local_search(start, max_iters=20000, time_limit=55):
        current = start[:]
        current_score = objective(current)
        best_local = current[:]
        best_local_score = current_score
        
        # Precompute prime counts for current state
        # counts[p] = number of pairs (i,j) such that p divides a_i + a_j
        counts = {}
        for i in range(N):
            for j in range(i + 1, N):
                s = current[i] + current[j]
                for p in pf_cache[s]:
                    counts[p] = counts.get(p, 0) + 1
        
        # We will try to reduce the number of distinct primes.
        # We can also try to reduce the total count of "bad" primes to make future removals easier.
        
        import time
        start_time = time.time()
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Choose a random index to replace
            idx = random.randint(0, N - 1)
            
            # Choose a new value
            # Bias towards values that share factors with existing sums?
            # Or just random.
            # Let's try a mix of random and "local" (near current value).
            if random.random() < 0.5:
                new_val = current[idx] + random.randint(-50, 50)
            else:
                new_val = random.randint(1, MAX_VAL)
            
            if new_val < 1 or new_val > MAX_VAL:
                continue
            if new_val in current and new_val != current[idx]:
                continue
            
            # Calculate delta
            # Remove contributions of old sums
            old_val = current[idx]
            old_sums = []
            for j in range(N):
                if j != idx:
                    s = old_val + current[j]
                    old_sums.append(s)
            
            new_sums = []
            for j in range(N):
                if j != idx:
                    s = new_val + current[j]
                    new_sums.append(s)
            
            # Check if new_val is distinct
            if len(set(current + [new_val]) - set(current)) < 1 and new_val != old_val:
                 # This check is redundant with the earlier one, but safe
                 pass
            
            # Update counts
            # We need to be careful: if a prime p appears in both old and new sums, the net change might be 0.
            # We just recalculate the counts for the affected primes.
            
            # Get the set of primes involved in old and new sums
            affected_primes = set()
            for s in old_sums:
                affected_primes |= pf_cache[s]
            for s in new_sums:
                affected_primes |= pf_cache[s]
            
            # Calculate new counts for these primes
            # We can just recount for these primes from scratch? No, that's O(N^2 * |P|).
            # Better: 
            # delta_count[p] = (count in new_sums) - (count in old_sums)
            
            new_counts = {p: 0 for p in affected_primes}
            for s in new_sums:
                for p in pf_cache[s]:
                    if p in new_counts:
                        new_counts[p] += 1
            
            old_counts = {p: 0 for p in affected_primes}
            for s in old_sums:
                for p in pf_cache[s]:
                    if p in old_counts:
                        old_counts[p] += 1
            
            # Apply changes
            # A prime p is "removed" if old_counts[p] > 0 and new_counts[p] == 0.
            # A prime p is "added" if old_counts[p] == 0 and new_counts[p] > 0.
            
            removed_primes = 0
            added_primes = 0
            
            for p in affected_primes:
                old_c = old_counts.get(p, 0)
                new_c = new_counts.get(p, 0)
                curr_c = counts.get(p, 0)
                
                # The total count of p in the whole set is curr_c - old_c + new_c
                # Wait, old_c is the count in the *removed* sums.
                # So the count in the *remaining* sums is curr_c - old_c.
                # The new total count is (curr_c - old_c) + new_c.
                
                total_new_c = curr_c - old_c + new_c
                
                if total_new_c == 0 and curr_c > 0:
                    removed_primes += 1
                if total_new_c > 0 and curr_c == 0:
                    added_primes += 1
            
            score_delta = added_primes - removed_primes
            
            # We accept if score_delta < 0
            # Or if score_delta == 0 and we want to improve the distribution?
            # Let's stick to strict improvement for now, with some random noise.
            
            if score_delta < 0 or (score_delta == 0 and random.random() < 0.1):
                # Accept
                current[idx] = new_val
                current_score += score_delta
                
                # Update counts
                for p in affected_primes:
                    old_c = old_counts.get(p, 0)
                    new_c = new_counts.get(p, 0)
                    curr_c = counts.get(p, 0)
                    total_new_c = curr_c - old_c + new_c
                    if total_new_c == 0:
                        if p in counts:
                            del counts[p]
                    else:
                        counts[p] = total_new_c
                
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local = current[:]
                    if best_local_score <= 3:
                        return best_local, best_local_score
            else:
                # Reject
                pass
        
        return best_local, best_local_score

    # Run local search with multiple restarts
    import time
    global_start = time.time()
    
    # Restart 0: from best found
    res, score = local_search(best, max_iters=50000, time_limit=50)
    if score < best_score:
        best_score = score
        best = res
    
    # More restarts with perturbation
    for r in range(10):
        if time.time() - global_start > 55:
            break
        
        # Perturb best
        perturbed = best[:]
        for _ in range(3):
            idx = random.randint(0, N - 1)
            perturbed[idx] = random.randint(1, MAX_VAL)
        
        # Ensure distinct
        while len(set(perturbed)) < N:
            perturbed[random.randint(0, N - 1)] = random.randint(1, MAX_VAL)
        
        res, score = local_search(perturbed, max_iters=20000, time_limit=4)
        if score < best_score:
            best_score = score
            best = res
    
    return best

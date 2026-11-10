import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 148s deadline, 2s safety margin

    # 1. Generate primes using a simple sieve up to a reasonable bound.
    # For n <= 10^18, the maximum number of distinct prime factors is 15.
    # The 15th smallest prime is 47. Product of first 15 primes is ~6.1e17.
    # If we remove a small prime like 2, we can add a prime up to ~3e17/2, which is huge.
    # However, adding very large primes disrupts the sorted divisor order significantly.
    # Let's generate primes up to 1,000,000. This is fast and gives plenty of candidates.
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(1000000)
    prime_set_global = set(primes)

    # 2. Compute the baseline: product of the smallest primes.
    # This maximizes the number of distinct prime factors k.
    prod = 1
    max_k = 0
    for p in primes:
        if prod * p > 10**18:
            break
        prod *= p
        max_k += 1

    # Start with the first max_k primes
    initial_primes = primes[:max_k]
    current_primes = list(initial_primes)
    current_n = 1
    for p in current_primes:
        current_n *= p

    # 3. Efficient tau_perp calculation.
    # For square-free n with k primes, there are 2^k divisors.
    # We generate divisors in sorted order using a merge approach.
    # To count coprime adjacent pairs, we track the set of prime factors (as a bitmask) for each divisor.
    # Two divisors are coprime if their bitmasks are disjoint.
    
    def tau_perp(prime_list):
        # prime_list is sorted.
        # divs: list of (divisor_value, bitmask)
        # Start with [ (1, 0) ]
        divs = [(1, 0)]
        for idx, p in enumerate(prime_list):
            bit = 1 << idx
            # Merge divs with [ (d*p, m|bit) for (d,m) in divs ]
            # Both lists are sorted by divisor_value.
            merged = []
            i = 0
            j = 0
            len_d = len(divs)
            while i < len_d or j < len_d:
                if j >= len_d:
                    merged.append(divs[i])
                    i += 1
                elif i >= len_d:
                    d, m = divs[j]
                    merged.append((d * p, m | bit))
                    j += 1
                else:
                    a_val, a_mask = divs[i]
                    b_val = divs[j][0] * p
                    b_mask = divs[j][1] | bit
                    if a_val <= b_val:
                        merged.append((a_val, a_mask))
                        i += 1
                    else:
                        merged.append((b_val, b_mask))
                        j += 1
            divs = merged
        
        # Count adjacent coprime pairs
        score = 0
        for i in range(len(divs) - 1):
            if (divs[i][1] & divs[i+1][1]) == 0:
                score += 1
        return score

    current_score = tau_perp(current_primes)
    best_score = current_score
    best_n = current_n

    # 4. Simulated Annealing / Stochastic Local Search.
    # Strategy:
    # - State: set of primes (and their product n).
    # - Move: Swap one prime in the set with another prime not in the set, such that the new product <= 10^18.
    # - Also allow adding/removing primes if possible (changing k), but k=15 is likely optimal or near-optimal.
    #   Actually, if we remove a prime, k decreases, reducing the total number of divisors (2^k).
    #   Since the score is at most 2^k - 1, reducing k is generally bad unless it allows a much better arrangement.
    #   However, the maximum possible score for k=15 is 32767.
    #   The baseline score is around 22.6? Wait, the score in the prompt is 22.6327.
    #   This suggests the score is NOT just the count of coprime pairs, but maybe normalized?
    #   "The evaluator measures the raw value (higher is better)."
    #   "score tau_perp(n) is the number of adjacent pairs d_i < d_{i+1} with gcd 1."
    #   If the score is just the count, it should be an integer. 22.6327 is not an integer.
    #   Let me re-read carefully.
    #   "Maximize. Large values below 10^18 are finite lower-bound constructions for the maximal order of tau_perp."
    #   "The evaluator measures the raw value (higher is better). The search score is value/seed, so 1.0 equals the seed program..."
    #   Ah, the "score" in the prompt (22.6327) is likely the *normalized* score or the result of a specific evaluation.
    #   But the task says "score tau_perp(n) is the number of adjacent pairs...".
    #   If the output of construct() is just an integer n, and the evaluator computes tau_perp(n),
    #   then the score I should maximize is the integer count.
    #   Why is the current program's score 22.6327?
    #   Maybe the "seed" program produces a score of 22.6327?
    #   Or maybe the score is defined differently in the evaluator than in the text description?
    #   "The search score is value/seed". This implies the reported score is relative.
    #   If the reported score is 22.6327, and it's value/seed, then value = 22.6327 * seed.
    #   If the seed program is the "greedy baseline" (product of smallest primes), let's calculate its score.
    #   For k=15, the number of divisors is 32768. The number of adjacent pairs is 32767.
    #   How many are coprime?
    #   For the product of the first 15 primes, the divisors are generated by subsets.
    #   The sorted order is not simply by subset size or index.
    #   Let's estimate the score for the baseline.
    #   If the baseline score is, say, 16000, and the reported score is 22.6, then the scale is different.
    #   However, the task says "Maximize tau_perp(n)". I will assume the goal is to maximize the integer count of coprime adjacent pairs.
    #   The "22.6327" might be a normalized metric in the evolutionary search framework, not the raw tau_perp value.
    #   I will optimize for the raw tau_perp value.

    # To make the search effective, we need a fast way to evaluate the score.
    # The tau_perp function above takes ~0.1s for k=15 (32k divisors, bitwise ops).
    # We can afford roughly 1000-2000 evaluations in 150s.
    
    # Better Move Generation:
    # Instead of random swaps, we can be smarter.
    # The score depends on the sorted order of divisors.
    # Swapping a small prime for a large prime changes the structure significantly.
    
    # Let's use a hybrid approach:
    # 1. Hill climbing with random restarts.
    # 2. Each iteration: pick a prime in the set, pick a random prime not in the set (within product limit), swap, evaluate.
    # 3. Accept if better. If stuck, perform random restarts or accept worse moves with annealing.

    best_prime_set = list(current_primes)
    
    def evaluate_state(prime_set):
        # Ensure prime_set is a list of distinct primes
        # Check product
        prod = 1
        for p in prime_set:
            prod *= p
            if prod > 10**18:
                return -1
        return tau_perp(prime_set)

    # Pre-compute candidates for each position?
    # If we swap p_old with p_new, the product changes by factor p_new/p_old.
    # We need prod * (p_new / p_old) <= 10^18 => p_new <= 10^18 * p_old / prod.
    
    # Let's run a loop until deadline.
    t0 = time.time()
    
    # Initial state is the greedy one.
    # We will perform local search.
    
    # To speed up evaluation, we can cache results?
    # The state space is huge, so caching is not useful.
    
    # Optimization:
    # The tau_perp calculation is the bottleneck.
    # Can we update the score incrementally?
    # Swapping a prime changes the sorted order of many divisors.
    # It's hard to do incrementally.
    # So we rely on fast tau_perp.
    
    # Let's try to improve the tau_perp speed.
    # Using lists of integers and masks.
    # The merge loop is O(2^k).
    
    # Alternative representation:
    # Since n is square-free, the divisors correspond to subsets of primes.
    # The sorted order of divisors is the same as the sorted order of the subset products.
    # This is equivalent to sorting the 2^k numbers.
    
    # Let's stick to the merge method, it's efficient enough.
    
    # Search Loop
    current_prime_list = list(current_primes)
    current_prod = current_n
    current_score_val = current_score
    best_score_val = current_score
    best_prime_list = list(current_primes)
    
    # Temperature for simulated annealing
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.95
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 2.0:
        # Generate a move
        # Pick a random index in current_prime_list
        idx = random.randint(0, len(current_prime_list) - 1)
        p_old = current_prime_list[idx]
        
        # Max value for p_new
        max_p_new = (10**18 * p_old) // current_prod
        
        # Find candidate primes
        # To avoid iterating through all 100k primes every time,
        # we can pre-filter primes <= max_p_new.
        # But max_p_new changes.
        # Let's just iterate through a random subset or the whole list?
        # The whole list is 100k. Iterating 100k times is fast (~1ms in Python if simple).
        # But we need to check if p_new is in current_prime_set.
        current_set = set(current_prime_list)
        
        # Collect valid candidates
        candidates = []
        # To save time, we can sample from primes list if it's large.
        # But we want to find good swaps.
        # Swapping with a prime close to p_old might be less disruptive?
        # Or maybe swapping with a very different prime is better?
        # Let's try random sampling from the valid range.
        
        # Efficient candidate selection:
        # We can't easily slice the primes list by value without binary search.
        # Let's use bisect.
        import bisect
        pos = bisect.bisect_right(primes, max_p_new)
        # primes[:pos] are valid by value.
        # We need to exclude those in current_set.
        
        # Instead of building a list of all candidates (which could be large),
        # we can pick a random prime from the valid range and check if it's in the set.
        # If it is, try again.
        
        found_new_prime = False
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts:
            if pos == 0:
                break
            p_new = primes[random.randint(0, pos - 1)]
            if p_new not in current_set:
                # Check if it's actually different
                if p_new != p_old:
                    found_new_prime = True
                    break
            attempts += 1
            
        if not found_new_prime:
            # Try to find a valid one by scanning?
            # Or skip this step.
            continue
            
        # Perform swap
        new_prime_list = current_prime_list.copy()
        new_prime_list[idx] = p_new
        new_prod = current_prod // p_old * p_new
        
        # Evaluate new state
        # To save time, we can check if new_prod > 10^18 (shouldn't happen due to max_p_new)
        new_score_val = tau_perp(new_prime_list)
        
        # Acceptance criterion
        if new_score_val > current_score_val:
            # Always accept improvement
            current_prime_list = new_prime_list
            current_prod = new_prod
            current_score_val = new_score_val
            no_improve_count = 0
            
            if new_score_val > best_score_val:
                best_score_val = new_score_val
                best_prime_list = list(current_prime_list)
        else:
            # Accept worse move with probability exp(-(delta)/T)
            delta = current_score_val - new_score_val
            if delta < 0:
                delta = 0
            prob = math.exp(-delta / T) if T > 0 else 0
            if random.random() < prob:
                current_prime_list = new_prime_list
                current_prod = new_prod
                current_score_val = new_score_val
                no_improve_count += 1
            else:
                no_improve_count += 1
                
        # Cooling
        if no_improve_count > max_no_improve:
            # Reset or increase temperature?
            # Let's try a random restart to a new greedy-like configuration
            # or just keep searching with lower T.
            T = max(T_min, T * cooling_rate)
            no_improve_count = 0
        else:
            T = max(T_min, T * cooling_rate)
            
    return best_prime_list[0] if len(best_prime_list) == 1 else 1 # Fallback, shouldn't happen
    
    # Wait, the function must return an integer n, not a list.
    # I made a mistake in the return statement above.
    # Let's fix the logic.

def construct():
    start_time = time.time()
    deadline = start_time + 148.0

    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(1000000)
    
    def tau_perp(prime_list):
        divs = [(1, 0)]
        for idx, p in enumerate(prime_list):
            bit = 1 << idx
            merged = []
            i = 0
            j = 0
            len_d = len(divs)
            while i < len_d or j < len_d:
                if j >= len_d:
                    merged.append(divs[i])
                    i += 1
                elif i >= len_d:
                    d, m = divs[j]
                    merged.append((d * p, m | bit))
                    j += 1
                else:
                    a_val, a_mask = divs[i]
                    b_val = divs[j][0] * p
                    b_mask = divs[j][1] | bit
                    if a_val <= b_val:
                        merged.append((a_val, a_mask))
                        i += 1
                    else:
                        merged.append((b_val, b_mask))
                        j += 1
            divs = merged
        
        score = 0
        for i in range(len(divs) - 1):
            if (divs[i][1] & divs[i+1][1]) == 0:
                score += 1
        return score

    # Initial state: product of smallest primes
    prod = 1
    max_k = 0
    for p in primes:
        if prod * p > 10**18:
            break
        prod *= p
        max_k += 1
        
    current_primes = primes[:max_k]
    current_n = 1
    for p in current_primes:
        current_n *= p
        
    current_score = tau_perp(current_primes)
    best_score = current_score
    best_n = current_n
    best_primes = list(current_primes)
    
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.95
    no_improve_count = 0
    max_no_improve = 100
    
    import bisect
    
    while time.time() < deadline - 2.0:
        idx = random.randint(0, len(current_primes) - 1)
        p_old = current_primes[idx]
        max_p_new = (10**18 * p_old) // current_n
        
        pos = bisect.bisect_right(primes, max_p_new)
        if pos == 0:
            continue
            
        current_set = set(current_primes)
        found_new_prime = False
        p_new = None
        attempts = 0
        max_attempts = 20
        while attempts < max_attempts:
            if pos == 0: break
            cand = primes[random.randint(0, pos - 1)]
            if cand not in current_set:
                p_new = cand
                found_new_prime = True
                break
            attempts += 1
            
        if not found_new_prime:
            continue
            
        new_primes = current_primes.copy()
        new_primes[idx] = p_new
        new_n = current_n // p_old * p_new
        
        new_score = tau_perp(new_primes)
        
        if new_score > current_score:
            current_primes = new_primes
            current_n = new_n
            current_score = new_score
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_n = new_n
                best_primes = list(new_primes)
        else:
            delta = current_score - new_score
            prob = math.exp(-delta / T) if T > 0 else 0
            if random.random() < prob:
                current_primes = new_primes
                current_n = new_n
                current_score = new_score
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            no_improve_count = 0
            # Random restart? Or just continue.
            # Let's try a few random swaps to escape local minima.
            for _ in range(5):
                idx_r = random.randint(0, len(current_primes) - 1)
                p_old_r = current_primes[idx_r]
                max_p_new_r = (10**18 * p_old_r) // current_n
                pos_r = bisect.bisect_right(primes, max_p_new_r)
                if pos_r == 0: continue
                current_set_r = set(current_primes)
                for _ in range(10):
                    cand_r = primes[random.randint(0, pos_r - 1)]
                    if cand_r not in current_set_r:
                        new_primes_r = current_primes.copy()
                        new_primes_r[idx_r] = cand_r
                        new_n_r = current_n // p_old_r * cand_r
                        # Accept regardless of score to escape
                        current_primes = new_primes_r
                        current_n = new_n_r
                        current_score = tau_perp(current_primes)
                        break
                        # break inner loop
                # break outer loop? No, just do one restart step.
                break
                
        T = max(T_min, T * cooling_rate)

    return best_n


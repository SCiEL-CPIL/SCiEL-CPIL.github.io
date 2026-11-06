import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    MAX_VAL = 20000
    
    # 1. Precompute smallest prime factor (SPF)
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Precompute set of prime factors for all numbers up to MAX_VAL
    # This speeds up the "is_smooth" check significantly
    prime_factors = [set() for _ in range(MAX_VAL + 1)]
    for i in range(2, MAX_VAL + 1):
        n = i
        while n > 1:
            p = spf[n]
            prime_factors[i].add(p)
            while n % p == 0:
                n //= p
                
    def get_prime_set(arr):
        """Return the set of distinct prime factors of all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                primes |= prime_factors[s]
        return primes

    def is_smooth_sum(a, b, allowed_primes):
        s = a + b
        return prime_factors[s].issubset(allowed_primes)

    # Candidate pool: Multiples of 6 up to 10000
    # Using multiples of 6 ensures 2 and 3 are always factors of sums.
    # This reduces the search space and guarantees a baseline of 2 primes.
    pool = list(range(6, 10001, 6))
    
    random.seed(42)
    
    best_set = None
    best_score = float('inf')
    
    # Strategies:
    # 1. Try to find a set of 12 where sums only use primes {2, 3}
    # 2. Try to find a set of 12 where sums only use primes {2, 3, 5}
    # 3. Try to find a set of 12 where sums only use primes {2, 3, 5, 7}
    # 4. Local search to minimize actual prime count
    
    allowed_prime_sets = [
        {2, 3},
        {2, 3, 5},
        {2, 3, 5, 7},
        {2, 3, 5, 7, 11}
    ]
    
    def greedy_construct(allowed_primes, num_restarts=20):
        """
        Greedily construct a set of 12 numbers from the pool such that
        all pairwise sums are smooth with respect to allowed_primes.
        """
        for _ in range(num_restarts):
            random.shuffle(pool)
            S = []
            for c in pool:
                if len(S) == 12:
                    return S
                # Check if c can be added to S
                valid = True
                for s in S:
                    if not is_smooth_sum(s, c, allowed_primes):
                        valid = False
                        break
                if valid:
                    S.append(c)
        return None

    # Attempt constrained constructions
    for allowed_primes in allowed_prime_sets:
        if time.time() > deadline - 10: # Leave time for local search
            break
        # Try a few restarts for this constraint
        # The number of restarts can be adjusted based on remaining time
        restarts = 50
        for _ in range(restarts):
            if time.time() > deadline - 10:
                break
            S = greedy_construct(allowed_primes, num_restarts=1)
            if S is not None:
                # Verify
                P = get_prime_set(S)
                if len(P) < best_score:
                    best_score = len(P)
                    best_set = S
                # If we found a solution with the minimum possible primes for this constraint, 
                # we might want to continue to see if we can do better with smaller prime sets?
                # No, we iterate from smallest prime sets to largest.
                # If we found a solution with {2,3}, best_score is 2. 
                # We can stop early if best_score == 2.
                if best_score == 2:
                    return best_set
                # If we found a solution with {2,3,5}, best_score is 3.
                # We continue to see if we can find {2,3} in later iterations? 
                # No, we already tried {2,3} first.
                # So if we are here, {2,3} failed.
                # We accept this as a good candidate and break out to local search?
                # Or keep trying other restarts for this constraint to see if we get a "cleaner" set?
                # The score only depends on the count. So any set with count 3 is equal.
                # We can break out of the restart loop for this constraint.
                break
        
        # If we found a solution, we can try to improve it with local search later.
        # For now, we just record it.
        if best_set is not None and best_score == len(allowed_primes):
            # We found a set that exactly matches the allowed primes count.
            # This is likely optimal for this constraint.
            # We can proceed to the next (larger) constraint only if we failed?
            # No, we want the MINIMUM number of primes.
            # If we found a set with 3 primes, we should try to find one with 2 primes?
            # But we already tried {2,3} (2 primes) and failed (otherwise we would have returned).
            # So 3 is the best we can hope for with this pool.
            # We can break out of the allowed_prime_sets loop.
            break

    # If no constrained solution found, or to polish the result, run local search
    # Start with best_set if found, else random multiples of 6
    if best_set is None:
        best_set = random.sample(pool, 12)
        best_score = len(get_prime_set(best_set))
    
    # Local Search
    # Objective: Minimize len(get_prime_set(S))
    current_set = best_set
    current_score = best_score
    current_primes = get_prime_set(current_set)
    
    # Time budget for local search
    ls_deadline = deadline - 2.0
    
    while time.time() < ls_deadline:
        improved = False
        # Try swapping one element
        i = random.randint(0, 11)
        old_val = current_set[i]
        
        # Pick a new value from pool
        # To avoid bias, pick randomly from pool
        new_val = random.choice(pool)
        
        if new_val == old_val:
            continue
        if new_val in current_set:
            continue
            
        candidate_set = current_set.copy()
        candidate_set[i] = new_val
        
        # Evaluate candidate
        # Optimization: We can calculate the change in prime set
        # Primes removed: those in sums involving old_val that are NOT in sums involving new_val or other sums
        # This is complex. Just re-evaluate.
        candidate_primes = get_prime_set(candidate_set)
        candidate_score = len(candidate_primes)
        
        if candidate_score < current_score:
            current_set = candidate_set
            current_score = candidate_score
            current_primes = candidate_primes
            improved = True
        elif candidate_score == current_score:
            # Accept neutral moves to explore the space
            # Probability of acceptance?
            # If we accept all neutral moves, we might wander.
            # But since we want to find a path to a lower score, wandering is good.
            # However, if we are stuck at 3, and all neighbors are 3, we might loop.
            # Let's accept with probability 0.5 to introduce randomness.
            if random.random() < 0.5:
                current_set = candidate_set
                current_primes = candidate_primes
                improved = True # Mark as step taken, not necessarily improvement
        
        # If we found a 2-prime solution, we can stop
        if current_score == 2:
            break

    return current_set

<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin within 90s limit, leaves time for evaluation overhead if any

    # We want 12 distinct k_i >= 0
    # a_i = 15 + 30 * k_i
    # Max a_i = 10000 => 15 + 30*k_i <= 10000 => 30*k_i <= 9985 => k_i <= 332
    
    # Improved Initial Solution: Use small k_i that are known to produce smooth sums
    # k_i = 0,1,2,3,4,5,6,7,8,9,10,11 gives sums 1+0+1=2, 1+0+2=3, etc.
    # These are all small and likely 5-smooth
    k_vals = list(range(12))
    current_cost = count_new_primes(k_vals)
    best_k = k_vals[:]
    best_cost = current_cost
    
    T = 5.0
    cooling_rate = 0.998  # Slower cooling for more exploration
    
    # Local Search / Simulated Annealing
    while time.time() < deadline:
        if T < 0.01:
            T = 1.0  # Restart temperature if stuck low
        
        # Pick a random index to modify
        idx = random.randint(0, 11)
        old_val = k_vals[idx]
        
        # Generate a new value
        # Bias towards small values where smooth numbers are more likely
        # Larger random walk steps for better exploration
        if random.random() < 0.15:
            # Random restart within [0, 150]
            new_val = random.randint(0, 150)
        else:
            # Random walk with larger steps
            step_size = random.choice([3, 5, 8, 12])
            new_val = max(0, min(332, old_val + random.randint(-step_size, step_size)))
        
        # Check distinctness
        # We need k_vals to be distinct
        if new_val in k_vals and new_val != old_val:
            continue

        # Apply move
        k_vals[idx] = new_val
        
        # Evaluate
        new_cost = count_new_primes(k_vals)
        
        # Acceptance criterion
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_k = k_vals[:]
        else:
            # Simulated Annealing
            delta = new_cost - current_cost
            # delta is integer >= 0
            if delta == 0:
                # Accept moves with same cost to explore
                if random.random() < 0.7:  # Higher acceptance for equal-cost moves
                    current_cost = new_cost
                else:
                    k_vals[idx] = old_val
            else:
                if random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                else:
                    # Reject move
                    k_vals[idx] = old_val
        
        T *= cooling_rate
    
    # Final Polish Phase: More aggressive local search on the best solution
    if time.time() < deadline - 2.0:
        polish_start = time.time()
        polish_deadline = min(deadline - 1.0, polish_start + 8.0)
        
        k_vals = best_k[:]
        current_cost = best_cost
        
        # Try all possible single-element moves from the best solution
        for idx in range(12):
            if time.time() > polish_deadline:
                break
            old_val = k_vals[idx]
            # Try all values in a neighborhood
            for new_val in range(max(0, old_val - 20), min(333, old_val + 21)):
                if new_val == old_val:
                    continue
                if new_val in k_vals:
                    continue
                
                k_vals[idx] = new_val
                new_cost = count_new_primes(k_vals)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_k = k_vals[:]
                    break
                else:
                    k_vals[idx] = old_val
        
        # Try pairwise swaps
        for i in range(12):
            if time.time() > polish_deadline:
                break
            for j in range(i + 1, 12):
                if time.time() > polish_deadline:
                    break
                # Try swapping values
                k_vals[i], k_vals[j] = k_vals[j], k_vals[i]
                new_cost = count_new_primes(k_vals)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_k = k_vals[:]
                else:
                    # Revert swap
                    k_vals[i], k_vals[j] = k_vals[j], k_vals[i]
    
    # Construct final array
    result = [15 + 30 * k for k in best_k]
    
    # Verify distinctness and bounds
    # k are distinct, so result are distinct.
    # Max k is 332, max result 15 + 30*332 = 9975 <= 10000.
    # Min result 15 >= 1.
    
    return result
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin within 90s limit

    # Strategy: Find 12 distinct positive integers <= 10000 such that
    # all pairwise sums have minimal distinct prime factors.
    # 
    # Key insight: If we can make all sums divisible by a small set of primes,
    # and the quotients are also smooth, we minimize the prime set.
    #
    # Approach: 
    # 1. Try candidate prime sets S = {2}, {2,3}, {2,3,5}, {2,3,5,7}, etc.
    # 2. For each S, find residues mod M (M=product of S) such that 
    #    (r_i + r_j) is S-smooth for all i<j
    # 3. Use backtracking to find 12 distinct values with compatible residues
    # 4. Refine with local search

    # Precompute smooth check for small numbers
    def is_S_smooth(n, S):
        """Check if all prime factors of n are in S"""
        if n < 2:
            return True
        temp = n
        for p in S:
            while temp % p == 0:
                temp //= p
        return temp == 1

    # Try different prime sets, starting with smallest
    prime_sets_to_try = [
        {2},
        {2, 3},
        {2, 3, 5},
        {2, 3, 5, 7},
        {2, 3, 5, 7, 11},
        {2, 3, 5, 7, 11, 13},
        {2, 3, 5, 7, 11, 13, 17},
    ]

    best_result = None
    best_cost = float('inf')

    for S in prime_sets_to_try:
        if time.time() > deadline - 5.0:
            break
        
        S_list = sorted(S)
        M = 1
        for p in S_list:
            M *= p
        
        # For each residue class r mod M, determine which other residues r' 
        # make (r + r') S-smooth (i.e., r + r' is divisible by all primes in S,
        # and the quotient is also S-smooth)
        # Actually, we need r_i + r_j to be S-smooth, not just divisible by S.
        
        # Precompute compatibility: comp[r] = set of r' such that (r + r') is S-smooth
        # But (r + r') mod M might not be enough; we need the actual sum to be S-smooth.
        # So we'll work with actual values.
        
        # Alternative: For each residue r mod M, find values a = r + k*M that are "good"
        # A value a is "good" if for most other residues r', there exists a' with residue r'
        # such that a + a' is S-smooth.
        
        # Simpler approach: Use simulated annealing directly on values, but with 
        # a cost function that penalizes primes not in S heavily.
        
        # Let's use a hybrid: start with values that are multiples of M plus offsets,
        # then refine.
        
        # Generate candidate values: for each residue r in [0, M), values r, r+M, r+2M, ...
        # up to 10000
        
        # Use local search with a cost that counts primes outside S
        def cost_with_S(a_vals, S):
            """Count distinct primes not in S dividing any pair sum"""
            extra_primes = set()
            n = len(a_vals)
            for i in range(n):
                for j in range(i + 1, n):
                    s = a_vals[i] + a_vals[j]
                    p_factors = get_prime_factors(s)
                    for p in p_factors:
                        if p not in S:
                            extra_primes.add(p)
            return len(extra_primes)
        
        # Initial solution: use small values
        a_vals = list(range(1, 13))
        current_cost = cost_with_S(a_vals, S)
        best_local = a_vals[:]
        best_local_cost = current_cost
        
        # Simulated annealing
        T = 10.0
        cooling_rate = 0.999
        no_improve = 0
        
        while time.time() < deadline - 10.0 and no_improve < 5000:
            if T < 0.1:
                T = 5.0
                no_improve = 0
            
            idx = random.randint(0, 11)
            old_val = a_vals[idx]
            
            # Generate new value: bias towards values that make sums smooth
            if random.random() < 0.3:
                # Try a value that's a multiple of M or close to one
                base = random.randint(1, 10000 // M) * M
                new_val = max(1, min(10000, base + random.randint(-M//2, M//2)))
            else:
                # Random walk
                step = random.randint(-50, 50)
                new_val = max(1, min(10000, old_val + step))
            
            if new_val == old_val:
                continue
            if new_val in a_vals:
                continue
            
            a_vals[idx] = new_val
            new_cost = cost_with_S(a_vals, S)
            
            if new_cost < current_cost:
                current_cost = new_cost
                no_improve = 0
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local = a_vals[:]
            else:
                delta = new_cost - current_cost
                if delta == 0:
                    if random.random() < 0.5:
                        current_cost = new_cost
                    else:
                        a_vals[idx] = old_val
                else:
                    if random.random() < math.exp(-delta / T):
                        current_cost = new_cost
                    else:
                        a_vals[idx] = old_val
                no_improve += 1
            
            T *= cooling_rate
        
        # Evaluate the best local solution with the true cost
        true_cost = count_distinct_primes(best_local)
        if true_cost < best_cost:
            best_cost = true_cost
            best_result = best_local[:]
        
        # If we found a solution with only primes in S, we're done
        if best_cost == len(S):
            break
    
    # If no good solution found, fall back to a reasonable default
    if best_result is None:
        # Use a simple construction
        best_result = list(range(1, 13))
    
    # Final polish: try to reduce the prime set further
    if time.time() < deadline - 5.0:
        current = best_result[:]
        cur_cost = count_distinct_primes(current)
        
        # Try replacing each element with nearby values
        for _ in range(3):
            improved = False
            for idx in range(12):
                if time.time() > deadline - 3.0:
                    break
                old = current[idx]
                for delta in range(-30, 31):
                    if delta == 0:
                        continue
                    new_val = old + delta
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val in current:
                        continue
                    
                    current[idx] = new_val
                    new_cost = count_distinct_primes(current)
                    if new_cost < cur_cost:
                        cur_cost = new_cost
                        improved = True
                        break
                    else:
                        current[idx] = old
            
            if not improved:
                break
        
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_result = current[:]
    
    return best_result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

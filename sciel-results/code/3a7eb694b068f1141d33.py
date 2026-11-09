import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Generate primes up to 1000
    def get_primes(limit):
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = 0
        return [i for i in range(2, limit + 1) if sieve[i]]

    primes = get_primes(1000)
    prime_set = set(primes)
    
    # Budget is 1.0
    # We use Fraction for exactness in budget check, but float is faster for heuristic.
    # Let's use float for speed, but verify with Fraction at the end.
    
    def sum_reciprocals(prime_list):
        return sum(1.0 / p for p in prime_list)

    # Initial set: primes >= 13
    # Sum of 1/p for p >= 13 is approx 0.94
    current_primes = [p for p in primes if p >= 13]
    current_sum = sum_reciprocals(current_primes)
    
    # Try to add small primes by swapping with large ones
    small_primes_to_add = [p for p in primes if p < 13]
    small_primes_to_add.sort(reverse=True) # 11, 7, 5, 3, 2
    
    # We want to maximize coverage. Small primes are better for small L.
    # Let's try to include as many small primes as possible.
    
    # Sort current_primes descending to easily pop largest
    # We will maintain a list.
    
    # Optimization: 
    # 1. Try to add 11. If budget tight, remove largest p such that 1/11 - 1/p <= slack.
    # 2. Try to add 7.
    # 3. Try to add 5.
    # 4. Try to add 3.
    # 5. Try to add 2.
    
    # Actually, let's just do a general swap loop.
    # Candidate small primes not in set.
    # Candidate large primes in set.
    
    # We'll iterate through small primes in descending order of value (11 first, as it's "smaller" than 7? No, 11 is larger than 7. 
    # But 11 has 1/11 cost. 7 has 1/7 cost. 
    # Usually, we want the smallest primes. 2 is best. 
    # But 2 covers all even numbers. 
    # Let's try to add 2, 3, 5, 7, 11 in that order? 
    # Adding 2 costs 0.5. Very expensive. 
    # Adding 11 costs 0.09. 
    # So we should try to add 11, 7, 5, 3, 2 in that order? 
    # No, 2 is the most useful for coverage density. 
    # But we have a budget of 1. 
    # If we add 2, we have 0.5 left. 
    # Sum of 1/p for p>=3 is approx 1.7. 
    # We can't afford many. 
    # So we must be selective.
    
    # Heuristic: 
    # Start with P = {p >= 13}.
    # Try to swap largest p in P with smallest q not in P (from {2,3,5,7,11}) if budget allows.
    # Repeat until no more swaps possible or time runs out.
    
    # Let's implement a function to find best swap
    def try_optimize_primes(prime_list, time_limit):
        primes_in_set = set(prime_list)
        primes_not_in_set = [p for p in primes if p not in primes_in_set and p < 13]
        # We only care about small primes for adding, as large primes are already in set or useless for small L coverage density.
        # Actually, we might want to swap a medium prime for a small prime.
        # Let's consider all primes not in set that are < 100.
        primes_not_in_set = [p for p in primes if p not in primes_in_set and p < 100]
        primes_not_in_set.sort() # Smallest first? Or largest first?
        # Smallest primes provide highest density.
        
        prime_list.sort(reverse=True) # Largest first for easy removal
        
        while time.time() < time_limit:
            improved = False
            current_sum = sum_reciprocals(prime_list)
            slack = 1.0 - current_sum
            
            # Try to add a small prime q
            # We want to add q and remove p such that 1/q - 1/p <= slack
            # i.e., 1/p >= 1/q - slack
            # To minimize loss of "value", we should remove the largest possible p?
            # Or the p that is least useful?
            # Large p are less useful for small L.
            # So we look for the largest p in prime_list that satisfies 1/p >= 1/q - slack.
            
            for q in primes_not_in_set:
                if q in primes_in_set:
                    continue
                    
                required_inv_p = 1.0 / q - slack
                if required_inv_p <= 0:
                    # We can add q without removing anything?
                    # If slack >= 1/q
                    # Add q
                    prime_list.append(q)
                    primes_in_set.add(q)
                    improved = True
                    break # Restart loop with new set
                
                # Need to remove p such that 1/p >= required_inv_p
                # i.e., p <= 1 / required_inv_p
                max_p_allowed = int(1.0 / required_inv_p)
                
                # Find largest p in prime_list such that p <= max_p_allowed
                # prime_list is sorted descending.
                # We can iterate from end (smallest) to start (largest)?
                # No, we want the LARGEST p that fits.
                # Since list is descending, the largest p is at the beginning.
                # We need p <= max_p_allowed.
                # So we scan from the beginning until we find p > max_p_allowed?
                # No, we want the largest p <= max_p_allowed.
                # Since sorted descending, we can binary search or linear scan.
                # Linear scan from start:
                candidate_p = None
                for p in prime_list:
                    if p <= max_p_allowed:
                        candidate_p = p
                        break # This is the largest such p
                    # If p is too large, keep going? No, list is descending.
                    # If first p is too large, maybe next is smaller and fits?
                    # Yes.
                
                if candidate_p:
                    # Swap
                    prime_list.remove(candidate_p)
                    primes_in_set.remove(candidate_p)
                    prime_list.append(q)
                    primes_in_set.add(q)
                    prime_list.sort(reverse=True)
                    improved = True
                    break # Restart
            
            if not improved:
                break
                
        return prime_list

    # Run prime optimization for 10 seconds
    prime_opt_deadline = start_time + 10.0
    current_primes = try_optimize_primes(current_primes, prime_opt_deadline)
    
    # Verify budget with Fraction
    total_budget = Fraction(0)
    for p in current_primes:
        total_budget += Fraction(1, p)
    if total_budget > 1:
        # Fallback to safe set
        current_primes = [p for p in primes if p >= 13]
    
    # Residue Optimization
    # L max 10^5
    L_MAX = 100000
    covered = [0] * (L_MAX + 1)
    residues = {}
    
    # Initialize residues
    # Greedy init: for each prime, pick residue that covers most uncovered numbers?
    # Or just random.
    # Let's do random init, then hill climb.
    
    def compute_L(covered):
        # Find largest L such that covered[1..L] > 0
        # We can maintain this, but for simplicity, recompute.
        # Optimization: keep track of first_zero
        # But recomputation is O(L).
        # Let's just scan.
        l = 0
        for i in range(1, L_MAX + 1):
            if covered[i] > 0:
                l = i
            else:
                break
        return l

    def update_coverage(p, r, delta):
        # delta is +1 or -1
        for i in range(r, L_MAX + 1, p):
            covered[i] += delta
            # Optimization: if covered[i] becomes 0 and i < current_L, we might need to update L
            # But we recompute L anyway.

    # Initialize
    for p in current_primes:
        r = random.randint(0, p - 1)
        residues[p] = r
        update_coverage(p, r, 1)
    
    current_L = compute_L(covered)
    best_L = current_L
    best_residues = residues.copy()
    
    # Hill Climbing
    # Time limit for this phase: until deadline
    # Use random restarts if stuck
    
    iterations = 0
    last_improvement = 0
    no_improve_count = 0
    MAX_NO_IMPROVE = 2000 # Restart if stuck for this many steps
    
    while time.time() < deadline:
        iterations += 1
        
        # Pick a random prime
        p = random.choice(current_primes)
        old_r = residues[p]
        
        # Pick a new residue
        # To be efficient, we can try all residues? p is small.
        # Or pick a random one.
        # Let's pick a random one different from old_r
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
            
        # Update coverage
        update_coverage(p, old_r, -1)
        update_coverage(p, new_r, 1)
        
        # Compute new L
        new_L = compute_L(covered)
        
        if new_L > current_L:
            current_L = new_L
            residues[p] = new_r
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
            no_improve_count = 0
        else:
            # Revert
            update_coverage(p, new_r, -1)
            update_coverage(p, old_r, 1)
            no_improve_count += 1
            
        # Check for restart
        if no_improve_count > MAX_NO_IMPROVE:
            # Random restart
            for p in current_primes:
                update_coverage(p, residues[p], -1)
            residues = {}
            for p in current_primes:
                r = random.randint(0, p - 1)
                residues[p] = r
                update_coverage(p, r, 1)
            current_L = compute_L(covered)
            no_improve_count = 0
            
    # Final result
    # Use best_residues
    # Verify budget again? Yes.
    total_budget = Fraction(0)
    result = []
    for p in current_primes:
        total_budget += Fraction(1, p)
        r = best_residues.get(p, 0)
        result.append([p, r])
        
    if total_budget > 1:
        # Should not happen, but fallback
        return [[p, (i + 1) % p] for i, p in enumerate([p for p in primes if p >= 13])]
        
    return result

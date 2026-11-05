import time
import random
from fractions import Fraction

def _sieve(n):
    is_prime = [True] * (n + 1)
    if n >= 0: is_prime[0] = False
    if n >= 1: is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit (or 150s if evaluator allows)
    
    primes = _sieve(1000)
    
    # We want to maximize L.
    # Initial heuristic: Use small primes.
    # Sum of reciprocals of primes:
    # 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23 + 1/29 + 1/31
    # 0.5 + 0.333 + 0.2 + 0.142 + 0.0909 + 0.0769 + 0.0588 + 0.0526 + 0.0434 + 0.0344 + 0.0322
    # Sum approx: 0.5+0.33=0.83, +0.2=1.03. So we can't take all up to 5.
    # Max set of smallest primes with sum <= 1:
    # {2, 3, 5, 7, 11, 13, 17, 19} -> Sum = 0.5+0.333+0.2+0.142+0.0909+0.0769+0.0588+0.0526 = 1.45? No.
    # Let's compute exactly.
    
    # Strategy:
    # 1. Start with a set of small primes that fits the budget.
    # 2. Set residues to cover 1..W efficiently.
    # 3. Local Search:
    #    a. Optimize residues for fixed primes (Greedy max gain).
    #    b. Try swapping primes (Remove p, add q) if it improves L.
    #    c. Try adding a prime if budget allows.
    
    # Window size for coverage tracking.
    # If we use primes up to 30, LCM is huge. But we only care about prefix 1..L.
    # Let's set W = 20000 to allow room for growth.
    # If we exceed W, we can expand or just cap L.
    # Actually, if we cover 1..W, L >= W.
    # Let's use W = 10000.
    
    W = 10000
    covered = [False] * (W + 1)
    
    # Initial set of primes
    # Try to find a good initial set.
    # Greedy add smallest primes that fit budget.
    initial_primes = []
    current_sum = Fraction(0, 1)
    
    # Sort primes by value (ascending)
    for p in primes:
        if current_sum + Fraction(1, p) <= 1:
            initial_primes.append(p)
            current_sum += Fraction(1, p)
        else:
            # If we can't add p, maybe we can swap a larger prime for a smaller one?
            # But we are adding in ascending order, so all are smaller than p.
            # So if p doesn't fit, no larger prime will fit either?
            # Wait. 1/p decreases as p increases.
            # If current_sum + 1/p > 1, then current_sum + 1/q > 1 for all q > p.
            # So we stop adding.
            break
            
    # It's possible we didn't use the budget well.
    # Example: {2, 3} sum = 5/6. Remaining 1/6.
    # Next prime 5: 1/5 = 0.2 > 1/6.
    # So we stop at {2, 3}.
    # But maybe {5, 7, 11, 13, ...} is better?
    # Sum 1/5+1/7+1/11+1/13+1/17+1/19+1/23+1/29+1/31
    # 0.2+0.142+0.0909+0.0769+0.0588+0.0526+0.0434+0.0344+0.0322 ~ 0.73.
    # We can add more.
    
    # Let's try a few different initial sets.
    # Set 1: Smallest primes (Greedy)
    # Set 2: Random sample of small/medium primes
    # Set 3: Primes around 10-50
    
    def evaluate_and_optimize(prime_set, time_limit):
        # prime_set: list of primes
        # Returns (L, classes)
        
        # Check budget
        s = Fraction(0, 1)
        for p in prime_set:
            s += Fraction(1, p)
        if s > 1:
            return 0, []
            
        # Initialize covered
        cov = [False] * (W + 1)
        
        # Initialize residues
        # For each prime, pick residue that covers most numbers in 1..W
        # This is a greedy assignment.
        residues = {}
        for p in prime_set:
            best_r = 0
            best_count = 0
            # Count covered numbers for each residue
            # Note: cov is initially all False, so we just count total numbers x in 1..W with x%p==r
            # But as we assign primes, we should consider interaction?
            # Simple greedy: assign one by one, updating cov.
            # Order: Smallest primes first? Largest?
            # Smallest primes have more "density".
            # Let's sort by p ascending.
            
            # Actually, let's just do a local search on residues.
            # Initialize randomly or 0.
            residues[p] = 0
            
        # Local Search Loop
        best_L = 0
        best_residues = residues.copy()
        
        # Helper to update coverage
        def update_coverage(res_map):
            c = [False] * (W + 1)
            for p, r in res_map.items():
                # Mark x where x % p == r
                start = r
                if start == 0:
                    start = p
                for x in range(start, W + 1, p):
                    c[x] = True
            return c
            
        # Helper to calculate L
        def calc_L(c):
            L = 0
            for i in range(1, W + 1):
                if c[i]:
                    L = i
                else:
                    break
            return L
            
        # Initial coverage
        cov = update_coverage(residues)
        cur_L = calc_L(cov)
        best_L = cur_L
        best_residues = residues.copy()
        
        # Optimization Phase 1: Greedy Residue Optimization
        # Iterate until no improvement
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Sort primes by size?
            for p in sorted(prime_set):
                # Try all residues for p
                best_r_for_p = residues[p]
                best_gain = 0
                # Current coverage with p's current residue
                # We need to know how many NEW numbers are covered by changing r.
                # This requires knowing coverage from OTHER primes.
                
                # Simplified:
                # For each candidate r, calculate how many x in 1..W are:
                # 1. Not covered by other primes
                # 2. Covered by (p, r)
                
                # To do this efficiently, we can maintain an array 'covered_by_others'
                # But that's expensive to recompute for each p.
                
                # Alternative:
                # Just compute the full coverage for each candidate r?
                # O(P * P * W) is too slow.
                
                # Let's use a heuristic:
                # Count uncovered numbers in 1..W for each residue of p.
                # This assumes other primes' coverage is fixed.
                
                # Create a set of numbers covered by other primes
                # This is O(W) per prime.
                # P ~ 10-20. W = 10000.
                # 20 * 10000 = 200k ops per pass.
                # Passes: P * (number of residue changes).
                # This is feasible.
                
                other_cov = [False] * (W + 1)
                for q in prime_set:
                    if q == p:
                        continue
                    r_q = residues[q]
                    start = r_q
                    if start == 0: start = q
                    for x in range(start, W + 1, q):
                        other_cov[x] = True
                        
                # Now find best r for p
                best_r = residues[p]
                best_new_cov = 0
                
                # Count uncovered numbers for each r
                # We can iterate x in 1..W.
                # If not other_cov[x], then x is candidate.
                # Increment count for x % p.
                counts = [0] * p
                for x in range(1, W + 1):
                    if not other_cov[x]:
                        counts[x % p] += 1
                        
                for r in range(p):
                    if counts[r] > best_new_cov:
                        best_new_cov = counts[r]
                        best_r = r
                        
                if best_r != residues[p]:
                    residues[p] = best_r
                    improved = True
                    # Update global coverage?
                    # We'll recompute at the end of the pass or when checking L.
                    # To be safe, recompute cov and L periodically.
                    
        # Recompute final L for this residue set
        cov = update_coverage(residues)
        cur_L = calc_L(cov)
        if cur_L > best_L:
            best_L = cur_L
            best_residues = residues.copy()
            
        # Optimization Phase 2: Prime Swaps
        # Try to swap a prime p in the set with a prime q not in the set.
        # Condition: 1/p + 1/q' <= 1 - (sum - 1/p) ... i.e. new sum <= 1.
        # Actually, sum_new = sum_old - 1/p + 1/q.
        # We need sum_new <= 1.
        
        # Generate candidate primes for swap
        # Primes not in set.
        # We want q such that 1/q is small enough to fit, or q is small to improve coverage.
        # Usually, smaller q is better for coverage density, but costs more budget.
        
        # Let's try swapping the largest prime in the set with a smaller prime not in the set?
        # Or just random swaps.
        
        # For time efficiency, let's just try a few specific swaps.
        # 1. Remove largest p, add smallest available q that fits budget.
        # 2. Remove smallest p, add largest available q that fits budget (unlikely to help).
        
        # Let's implement a simple local search for swaps.
        # We will run this for a few seconds.
        
        swap_deadline = time.time() + 20 # 20 seconds for swaps
        
        while time.time() < swap_deadline:
            # Pick a random prime from the set to remove
            if not prime_set: break
            p_remove = random.choice(prime_set)
            
            # Find candidates to add
            # Budget remaining after removing p_remove:
            # sum_rest = sum_old - 1/p_remove
            # We need sum_rest + 1/q <= 1
            # 1/q <= 1 - sum_rest
            
            sum_rest = Fraction(0, 1)
            for p in prime_set:
                if p != p_remove:
                    sum_rest += Fraction(1, p)
            
            max_inv_q = 1 - sum_rest
            
            candidates = []
            for q in primes:
                if q in prime_set:
                    continue
                if Fraction(1, q) <= max_inv_q:
                    candidates.append(q)
                    
            if not candidates:
                continue
                
            # Pick a random candidate or the smallest?
            # Smallest q gives highest density.
            # Let's try the smallest candidate first.
            q_add = min(candidates)
            
            # Create new set
            new_set = [p for p in prime_set if p != p_remove]
            new_set.append(q_add)
            
            # Evaluate new set
            # Reuse residues? 
            # The residues for common primes might still be good.
            # Initialize residues for new_set.
            new_residues = {}
            for p in new_set:
                if p in best_residues:
                    new_residues[p] = best_residues[p]
                else:
                    new_residues[p] = 0 # Random or 0
            
            # Quick evaluation: Optimize residues for this new set briefly
            # Use a faster version of the residue optimization
            # Just one pass of greedy residue update
            
            other_cov_cache = {} # Not needed for one pass
            
            # One pass residue optimization
            for p in sorted(new_set):
                # Compute other_cov
                other_cov = [False] * (W + 1)
                for q in new_set:
                    if q == p: continue
                    r_q = new_residues[q]
                    start = r_q
                    if start == 0: start = q
                    for x in range(start, W + 1, q):
                        other_cov[x] = True
                        
                counts = [0] * p
                for x in range(1, W + 1):
                    if not other_cov[x]:
                        counts[x % p] += 1
                        
                best_r = 0
                max_c = 0
                for r in range(p):
                    if counts[r] > max_c:
                        max_c = counts[r]
                        best_r = r
                new_residues[p] = best_r
                
            # Calculate L
            cov = update_coverage(new_residues)
            new_L = calc_L(cov)
            
            if new_L > best_L:
                best_L = new_L
                best_residues = new_residues
                prime_set = new_set # Update the set for next iteration
                
        return best_L, [[p, best_residues[p]] for p in prime_set]

    # Run multiple trials with different initial sets
    
    best_overall_L = 0
    best_overall_classes = []
    
    # Trial 1: Greedy smallest primes
    trial_primes = list(initial_primes)
    L, classes = evaluate_and_optimize(trial_primes, deadline - 10)
    if L > best_overall_L:
        best_overall_L = L
        best_overall_classes = classes
        
    if time.time() > deadline - 10:
        return best_overall_classes
        
    # Trial 2: Primes in range [10, 50]
    trial_primes_2 = [p for p in primes if 10 <= p <= 50]
    # Filter by budget
    filtered_2 = []
    s = Fraction(0, 1)
    for p in trial_primes_2:
        if s + Fraction(1, p) <= 1:
            filtered_2.append(p)
            s += Fraction(1, p)
            
    if len(filtered_2) > 0:
        L, classes = evaluate_and_optimize(filtered_2, deadline - 5)
        if L > best_overall_L:
            best_overall_L = L
            best_overall_classes = classes
            
    if time.time() > deadline - 5:
        return best_overall_classes
        
    # Trial 3: Random mix
    random.seed(42)
    trial_primes_3 = random.sample(primes, 15)
    # Filter by budget - sort by 1/p descending? No, just check sum.
    # To maximize chance of fitting, pick small ones?
    # Let's just take the first 15 that fit.
    filtered_3 = []
    s = Fraction(0, 1)
    for p in sorted(trial_primes_3):
        if s + Fraction(1, p) <= 1:
            filtered_3.append(p)
            s += Fraction(1, p)
            
    if len(filtered_3) > 0:
        L, classes = evaluate_and_optimize(filtered_3, deadline - 1)
        if L > best_overall_L:
            best_overall_L = L
            best_overall_classes = classes
            
    return best_overall_classes

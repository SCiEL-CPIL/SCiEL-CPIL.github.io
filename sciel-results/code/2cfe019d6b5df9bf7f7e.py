import time
import random
from fractions import Fraction
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Generate primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    prime_set = set(primes)
    MAX_L = 1000
    
    # Precompute masks for all primes and residues
    # mask[p][r] is a bitset of integers n in 1..1000 such that n % p == r
    masks = {}
    for p in primes:
        masks[p] = [0] * p
        for r in range(p):
            mask = 0
            # n ranges from 1 to 1000
            # First n >= 1 such that n % p == r
            if r == 0:
                start = p
            else:
                start = r
            n = start
            while n <= MAX_L:
                mask |= (1 << (n - 1))
                n += p
            masks[p][r] = mask
            
    full_mask = (1 << MAX_L) - 1
    
    # Helper to calculate L from a cover bitset
    def calc_L(cover):
        # Find the first 0 bit from the left (MSB side of the relevant range)
        # We want the largest L such that bits 0..L-1 are set.
        # If bit i is 0, L <= i.
        # We can iterate from 1 to MAX_L.
        # Optimization: if cover == full_mask, return MAX_L.
        if (cover & full_mask) == full_mask:
            return MAX_L
        
        # Find first uncovered position
        # ~cover has 1s where cover has 0s.
        # We want the lowest set bit in (~cover & full_mask).
        # If no bits set in that range, it's fully covered.
        uncovered = (~cover) & full_mask
        if uncovered == 0:
            return MAX_L
        
        # Get index of lowest set bit
        # Python 3.11+ has int.bit_length, but we need index.
        # (x & -x) isolates the lowest set bit.
        # Then log2.
        low_bit = uncovered & -uncovered
        # Index is log2(low_bit)
        # Since MAX_L is 1000, we can just loop or use bit_length-1
        idx = low_bit.bit_length() - 1
        return idx
        
    # Helper to calculate total cost
    def calc_cost(selected_primes):
        # selected_primes is a list of unique primes
        c = Fraction(0)
        for p in selected_primes:
            c += Fraction(1, p)
        return c

    # Helper to evaluate a solution: returns (L, cost, is_valid)
    # Solution: list of (p, r)
    def evaluate_solution(classes):
        if not classes:
            return 0, Fraction(0), True
        
        # Check distinct primes
        seen = set()
        for p, r in classes:
            if p in seen:
                return 0, Fraction(1), False # Invalid
            seen.add(p)
            if r < 0 or r >= p:
                return 0, Fraction(1), False
            if p > 1000:
                return 0, Fraction(1), False
                
        # Calculate cover
        cover = 0
        for p, r in classes:
            cover |= masks[p][r]
            
        L = calc_L(cover)
        
        # Calculate cost
        cost = calc_cost(seen)
        
        if cost > 1:
            return 0, cost, False
            
        return L, cost, True

    # --- Phase 1: Greedy Initialization ---
    # We want a good starting point.
    # Greedy strategy: Pick (p, r) that maximizes gain * p (value/cost)
    # But we must respect the budget.
    
    current_classes = []
    used_primes = set()
    current_cover = 0
    current_cost = Fraction(0)
    
    # To make greedy fast, we'll do a few passes.
    # Since we have 115s, we can afford a slightly slower init if it leads to better SA.
    
    # Sort primes by size? Small primes are "expensive" but cover dense areas.
    # Large primes are "cheap" but cover sparse areas.
    # Actually, let's just run the greedy loop until we can't add more or budget is exhausted.
    
    # Optimization for Greedy:
    # Precompute list of all (p, r, mask)
    # But checking all 84k residues every step is slow in Python.
    # Let's limit the greedy init to be quick, then use SA to refine.
    
    # A better greedy: 
    # 1. Include small primes (2, 3, 5, 7) if they fit? 
    #    Cost(2)=0.5, Cost(3)=0.33. Sum=0.83.
    #    With 2 and 3, we cover 1..L quite well?
    #    2 covers evens. 3 covers 1,4,7... or 2,5,8...
    #    Let's just run a fast greedy.
    
    # To speed up greedy, we only check the "best" residue for each prime in each step.
    # We scan primes, find best r for that p, then pick best p.
    
    while time.time() < start_time + 5.0: # Limit greedy to 5s
        best_gain_ratio = -1
        best_p = None
        best_r = None
        best_mask = None
        best_gain = 0
        
        for p in primes:
            if p in used_primes:
                continue
            if current_cost + Fraction(1, p) > 1:
                continue
            
            prime_best_gain = 0
            prime_best_r = -1
            prime_best_mask = 0
            
            for r in range(p):
                m = masks[p][r]
                new_bits = m & ~current_cover
                if new_bits:
                    gain = new_bits.bit_count()
                    if gain > prime_best_gain:
                        prime_best_gain = gain
                        prime_best_r = r
                        prime_best_mask = m
                        
            if prime_best_gain > 0:
                ratio = prime_best_gain * p
                if ratio > best_gain_ratio:
                    best_gain_ratio = ratio
                    best_p = p
                    best_r = prime_best_r
                    best_mask = prime_best_mask
                    best_gain = prime_best_gain
                    
        if best_p is None:
            break
            
        used_primes.add(best_p)
        current_cost += Fraction(1, best_p)
        current_classes.append((best_p, best_r))
        current_cover |= best_mask
        
        if (current_cover & full_mask) == full_mask:
            break
            
    best_L, best_cost, is_valid = evaluate_solution(current_classes)
    best_solution = [list(x) for x in current_classes]
    
    # --- Phase 2: Simulated Annealing ---
    # State: list of (p, r)
    # Moves:
    # 1. Add a new prime (if budget allows)
    # 2. Remove a prime
    # 3. Change residue of an existing prime
    # 4. Swap a prime for another (if budget allows)
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.99
    
    T = T_start
    random.seed(42) # For reproducibility in debugging, but maybe random is better?
    
    current_L = best_L
    current_classes_list = [list(x) for x in best_solution]
    current_cover = 0
    for p, r in current_classes_list:
        current_cover |= masks[p][r]
    current_cost = calc_cost([p for p, r in current_classes_list])
    
    # Precompute available primes for quick access
    all_primes = primes
    
    while time.time() < deadline and T > T_end:
        # Generate a neighbor
        move_type = random.randint(0, 3)
        new_classes = [list(x) for x in current_classes_list]
        new_cover = current_cover
        new_cost = current_cost
        new_used = set(p for p, r in new_classes)
        
        if move_type == 0: # Add
            # Find affordable primes not in new_used
            affordable = [p for p in all_primes if p not in new_used and new_cost + Fraction(1, p) <= 1]
            if affordable:
                p = random.choice(affordable)
                r = random.randint(0, p - 1)
                new_classes.append([p, r])
                new_used.add(p)
                new_cover |= masks[p][r]
                new_cost += Fraction(1, p)
            else:
                continue
                
        elif move_type == 1: # Remove
            if new_classes:
                idx = random.randint(0, len(new_classes) - 1)
                p, r = new_classes.pop(idx)
                new_used.discard(p)
                # Recalculate cover is hard if we just remove. 
                # We need to rebuild cover or maintain counts.
                # Rebuilding cover for 1000 bits is fast.
                new_cover = 0
                for pp, rr in new_classes:
                    new_cover |= masks[pp][rr]
                new_cost = calc_cost(new_used)
            else:
                continue

        elif move_type == 2: # Change Residue
            if new_classes:
                idx = random.randint(0, len(new_classes) - 1)
                p, old_r = new_classes[idx]
                new_r = random.randint(0, p - 1)
                if new_r != old_r:
                    new_classes[idx][1] = new_r
                    # Update cover
                    # Remove old contribution? No, just rebuild.
                    new_cover = 0
                    for pp, rr in new_classes:
                        new_cover |= masks[pp][rr]
                    # Cost unchanged
            else:
                continue

        elif move_type == 3: # Swap Prime
            if new_classes:
                idx = random.randint(0, len(new_classes) - 1)
                old_p, r = new_classes[idx]
                # Find a replacement prime q such that cost(q) <= cost(old_p) + (1 - current_cost)
                # Actually, we just need new_cost <= 1.
                # Let's pick a random prime q not in new_used.
                candidates = [p for p in all_primes if p not in new_used]
                if candidates:
                    q = random.choice(candidates)
                    # Check if we can afford q instead of old_p
                    # New cost = current_cost - 1/old_p + 1/q
                    if new_cost - Fraction(1, old_p) + Fraction(1, q) <= 1:
                        new_classes[idx][0] = q
                        # Residue r might be invalid if r >= q? 
                        # If r >= q, we must change r.
                        new_r = r
                        if new_r >= q:
                            new_r = random.randint(0, q - 1)
                        new_classes[idx][1] = new_r
                        new_used.discard(old_p)
                        new_used.add(q)
                        # Rebuild cover
                        new_cover = 0
                        for pp, rr in new_classes:
                            new_cover |= masks[pp][rr]
                        new_cost = calc_cost(new_used)
                    else:
                        continue
            else:
                continue
                
        # Evaluate new state
        new_L = calc_L(new_cover)
        
        # Acceptance criterion
        if new_L > current_L:
            current_L = new_L
            current_classes_list = new_classes
            current_cover = new_cover
            current_cost = new_cost
            # Update best
            if current_L > best_L:
                best_L = current_L
                best_solution = [list(x) for x in current_classes_list]
        else:
            # Delta L = new_L - current_L (negative or 0)
            delta_L = new_L - current_L
            # We want to accept worse solutions to escape local optima.
            # Probability = exp(delta_L / T)
            # But delta_L is integer.
            # If delta_L == 0, we might still accept to explore? 
            # Usually we only accept if better or with prob exp(-cost/T).
            # Here "cost" is loss in L.
            if delta_L < 0:
                prob = 2.71828 ** (delta_L / T) # e^(delta/T)
                if random.random() < prob:
                    current_L = new_L
                    current_classes_list = new_classes
                    current_cover = new_cover
                    current_cost = new_cost
            else:
                # delta_L == 0, accept with small probability to randomize
                if random.random() < 0.1:
                    current_L = new_L
                    current_classes_list = new_classes
                    current_cover = new_cover
                    current_cost = new_cost
                    
        T *= cooling_rate
        
    # Final Verification
    # Ensure the best solution is valid
    final_L, final_cost, valid = evaluate_solution(best_solution)
    if not valid:
        # Fallback to a simple valid solution if something went wrong
        # e.g. just prime 2, r=1
        best_solution = [[2, 1]]
        final_L, final_cost, valid = evaluate_solution(best_solution)
        
    return best_solution

# EVOLVE-BLOCK-END

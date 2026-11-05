"""Rewritten program: Simulated Annealing for global residue optimization."""
# EVOLVE-BLOCK-START
import math
import random

def construct():
    # Generate primes <= 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(p ** 0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    
    n_primes = len(primes)  # 168
    L_max = 5000  # Search horizon
    
    # Initialize residues using a simple greedy to get a good start
    residues = [0] * n_primes
    coverage = [0] * (L_max + 1)
    
    # Phase 1: Greedy Initialization
    # Assign small primes first to build a base layer of coverage
    # We want to cover early positions densely.
    # For each prime, pick residue that maximizes coverage of positions with cov < 2 in [1, 500]
    
    for i, p in enumerate(primes):
        best_r = 0
        best_gain = -1
        
        for r in range(p):
            gain = 0
            # Check positions in [1, 500]
            x = r if r > 0 else p
            if r == 0: x = p
            while x <= 500:
                if coverage[x] < 2:
                    gain += 1
                x += p
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        residues[i] = best_r
        # Update coverage
        x = best_r if best_r > 0 else p
        if best_r == 0: x = p
        while x <= L_max:
            coverage[x] += 1
            x += p
            
    # Helper to calculate deficit sum
    def calc_deficit(coverage):
        s = 0
        for c in coverage[1:]:
            if c < 2:
                s += (2 - c)
        return s
        
    # Helper to find L
    def find_L(coverage):
        L = 0
        for x in range(1, L_max + 1):
            if coverage[x] >= 2:
                L = x
            else:
                break
        return L

    current_L = find_L(coverage)
    best_L = current_L
    best_residues = residues[:]
    best_coverage = coverage[:]
    
    # Simulated Annealing
    # We need to be careful with time. 120s limit.
    # Estimate steps: 50,000 to 100,000 steps.
    
    T = 10.0
    T_min = 0.01
    alpha = 0.9995 # cooling rate
    
    # Precompute affected positions lists? No, too memory heavy for all primes.
    # Just iterate on the fly.
    
    steps = 0
    import time
    start_time = time.time()
    
    while time.time() - start_time < 110: # Leave 10s buffer
        if T < T_min:
            break
            
        steps += 1
        
        # Pick random prime
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        r_old = residues[i]
        
        # Pick new residue
        r_new = random.randint(0, p - 1)
        
        if r_old == r_new:
            continue
            
        # Calculate delta E
        # Affected positions are those in (r_old mod p) and (r_new mod p) sequences
        # We need to update coverage temporarily or calculate delta directly
        
        # To be safe and simple, we will update coverage, calc delta, and revert if needed.
        # Cost: O(L_max / p)
        
        # Remove old
        x_old = r_old if r_old > 0 else p
        if r_old == 0: x_old = p
        old_positions = []
        while x_old <= L_max:
            old_positions.append(x_old)
            x_old += p
            
        # Add new
        x_new = r_new if r_new > 0 else p
        if r_new == 0: x_new = p
        new_positions = []
        while x_new <= L_max:
            new_positions.append(x_new)
            x_new += p
            
        # Calculate deficit change
        # We need to know the coverage before change for all affected spots.
        # Affected spots = set(old_positions) U set(new_positions)
        
        # Optimization: Only iterate over the union.
        # Since p is small, lists are long. For p=2, len=2500.
        # Creating a set of 5000 ints is fast.
        
        affected = set(old_positions)
        affected.update(new_positions)
        
        delta_E = 0
        for x in affected:
            c_old = coverage[x]
            
            # Determine new coverage
            # c_new = c_old - (1 if x in old_positions else 0) + (1 if x in new_positions else 0)
            # Using sets for membership is O(1)
            
            in_old = x in old_positions # This is O(1) if we used a set, but old_positions is list.
            # Better: use a temporary dict or just logic.
            
            # Actually, let's just update coverage, calculate, and revert.
            # It's simpler and less prone to bug.
            
            # Apply change
            # Subtract old
            for x in old_positions:
                coverage[x] -= 1
            # Add new
            for x in new_positions:
                coverage[x] += 1
            
            # Calculate new deficit for affected only?
            # No, we need total deficit change.
            # E_new - E_old = sum_{x in affected} (f(c_new[x]) - f(c_old[x]))
            
            # We lost c_old[x] for old positions, gained for new.
            # Let's just calculate the sum over affected.
            
            # We need c_old[x] values. We just modified coverage.
            # So c_old[x] is NOT in coverage anymore.
            
            # This approach (modify then calc) requires storing old values.
            
            # Let's go back to:
            # 1. Store old coverage for affected spots.
            # 2. Modify coverage.
            # 3. Calc new deficit for affected spots.
            # 4. Compare with old deficit for affected spots.
            # 5. Revert if rejected.
            
            # This is getting complex. Let's use the direct delta formula.
            
            # f(c) = max(0, 2-c)
            # If c >= 2, f(c) = 0.
            # If c == 1, f(c) = 1.
            # If c == 0, f(c) = 2.
            
            # Change for a position x:
            # If x in old only: c decreases by 1.
            # If x in new only: c increases by 1.
            # If x in both: no change.
            # If x in neither: no change.
            
            # We can compute delta for each x in affected.
            
            # We need c_old[x].
            # Let's fetch c_old[x] before modifying.
            
            # Re-doing the logic cleanly:
            
            # 1. Identify affected set.
            # 2. For each x in affected:
            #    c0 = coverage[x]
            #    c1 = c0
            #    if x in old_set: c1 -= 1
            #    if x in new_set: c1 += 1
            #    delta_E += f(c1) - f(c0)
            
            # To make membership O(1), convert lists to sets.
            
            break # Break out of the inner loop to restart the step logic cleanly
            
        # --- Corrected Step Logic ---
        
        # 1. Build sets for O(1) lookup
        old_set = set(old_positions)
        new_set = set(new_positions)
        affected_set = old_set.union(new_set)
        
        delta_E = 0
        for x in affected_set:
            c0 = coverage[x]
            c1 = c0
            if x in old_set:
                c1 -= 1
            if x in new_set:
                c1 += 1
            
            f0 = 0 if c0 >= 2 else (2 - c0)
            f1 = 0 if c1 >= 2 else (2 - c1)
            delta_E += (f1 - f0)
            
        # 2. Decide accept/reject
        accept = False
        if delta_E < 0:
            accept = True
        else:
            # Probability of accepting worse solution
            if delta_E == 0:
                accept = True # Neutral move, always accept to explore
            else:
                # exp(-delta_E / T)
                # delta_E is integer, T is float
                # If T is small, exp is small
                if random.random() < math.exp(-delta_E / T):
                    accept = True
                    
        if accept:
            # Commit change
            for x in old_positions:
                coverage[x] -= 1
            for x in new_positions:
                coverage[x] += 1
            residues[i] = r_new
            
            # Check new L
            # Only check if we improved deficit or if we might have improved L
            # Checking L every step is O(L_max). 100k steps * 5000 = 500M ops. Too slow.
            # We should check L periodically or only when deficit improves significantly.
            
            # Strategy: Check L every 100 steps or if deficit dropped by > 10.
            if steps % 100 == 0:
                new_L = find_L(coverage)
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues[:]
                    best_coverage = coverage[:]
        else:
            # Revert (do nothing, as we didn't modify coverage)
            pass
            
        # Cool down
        T *= alpha
        
    # Final check
    new_L = find_L(coverage)
    if new_L > best_L:
        best_L = new_L
        best_residues = residues[:]
        
    return best_residues
# EVOLVE-BLOCK-END

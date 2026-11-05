import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    all_primes_set = set(primes)
    
    # We aim to cover up to L_max. 
    # The seed covers ~1600. Inspiration 1 covers ~1800. 
    # Let's aim for a range that allows optimization.
    L_max = 5000
    
    covered = np.zeros(L_max + 1, dtype=bool)
    indices = np.arange(1, L_max + 1)
    
    selected_classes = []
    used_primes = set()
    current_sum = Fraction(0, 1)
    
    # --- Phase 1: Gap-Filling Greedy ---
    # Iterate x from 1. If x is not covered, pick the best prime to cover it.
    
    for x in range(1, L_max + 1):
        if time.time() > deadline - 20:  # Leave time for local search
            break
            
        if covered[x]:
            continue
            
        best_p = None
        best_r = None
        best_gain = -1
        
        # Iterate through unused primes
        # Optimization: Only consider primes that are "affordable"
        # We can pre-filter primes by cost, but cost is 1/p.
        # Smaller p = higher cost.
        
        for p in primes:
            if p in used_primes:
                continue
            
            cost = Fraction(1, p)
            if current_sum + cost > Fraction(1, 1):
                continue
                
            r = x % p
            
            # Calculate gain: number of uncovered y in [1, L_max] such that y % p == r
            # mask = (indices % p) == r
            # gain = np.sum(~covered[1:L_max+1] & mask)
            
            # Vectorized check for speed
            # Note: indices is 1..L_max. 
            # y % p == r is equivalent to (y - r) % p == 0.
            
            # Efficient counting:
            # The numbers covered are r, r+p, r+2p...
            # We need to count how many of these are currently uncovered.
            
            # Using numpy boolean mask is fast enough for L_max=5000
            mask = (indices % p) == r
            unc_mask = ~covered[1:L_max+1]
            gain = np.sum(unc_mask & mask)
            
            if gain > best_gain:
                best_gain = int(gain)
                best_p = p
                best_r = r
        
        if best_p is None:
            # No prime can cover x without exceeding budget
            break
            
        # Add the class
        used_primes.add(best_p)
        current_sum += Fraction(1, best_p)
        selected_classes.append([best_p, best_r])
        
        # Update covered
        mask = (indices % best_p) == best_r
        covered[1:L_max+1][mask] = True
        
    # Calculate current L
    def get_L(covered_arr):
        # Find first False in 1..L_max
        false_indices = np.where(~covered_arr[1:L_max+1])[0]
        if len(false_indices) == 0:
            return L_max
        return false_indices[0]
        
    current_L = get_L(covered)
    
    # --- Phase 2: Local Search ---
    # Try to improve L by:
    # 1. Changing residues of existing primes
    # 2. Swapping primes
    
    improved = True
    max_iterations = 50
    iteration = 0
    
    while improved and iteration < max_iterations and time.time() < deadline - 5:
        improved = False
        iteration += 1
        
        # 1. Residue Optimization
        for i in range(len(selected_classes)):
            if time.time() > deadline - 5:
                break
                
            p, r = selected_classes[i]
            
            # Identify which numbers are covered ONLY by this class
            # This is expensive to compute exactly every time.
            # Approximation: Check if changing r to r' uncovers any number in [1, current_L]
            
            best_r_new = r
            best_L_new = current_L
            
            # Try all residues
            for r_new in range(p):
                if r_new == r:
                    continue
                    
                # Simulate change
                # Remove old coverage, add new coverage?
                # Faster: Check if new residue covers all "critical" points
                # Critical points are those in [1, current_L] covered by (p, r) 
                # that are NOT covered by any other class.
                
                # To avoid O(L * N_classes) per check, we can just compute the new L
                # by updating the covered array.
                
                # Save state
                old_covered = covered.copy()
                
                # Remove old prime's contribution? 
                # This is hard because we don't track counts.
                # Instead, we can just recompute coverage from scratch? 
                # No, too slow.
                
                # Alternative: 
                # We know which numbers are covered by (p, r).
                # Let's find if any of them are covered by others.
                
                # Actually, let's just try the swap and see if L improves.
                # We need to update `covered`.
                # This requires knowing if removing (p, r) uncovers anything.
                
                # Let's use a simpler heuristic: 
                # Only try residues that cover x = current_L + 1 (the next gap).
                # If current_L + 1 is not covered by (p, r_new), it won't help extend L.
                
                if (current_L + 1) % p != r_new:
                    continue
                
                # Check if this residue uncovers any number in [1, current_L]
                # that was covered ONLY by (p, r).
                
                # Find numbers in [1, current_L] covered by (p, r)
                # y % p == r
                # Check if they are covered by other classes.
                
                safe = True
                # Iterate through numbers covered by old class in [1, current_L]
                # start = r if r != 0 else p
                start = r if r != 0 else p
                for y in range(start, current_L + 1, p):
                    if not covered[y]: # Should not happen if covered is correct
                        continue
                    # Check if y is covered by any other class
                    is_covered_by_other = False
                    for j in range(len(selected_classes)):
                        if j == i:
                            continue
                        p_j, r_j = selected_classes[j]
                        if y % p_j == r_j:
                            is_covered_by_other = True
                            break
                    if not is_covered_by_other:
                        safe = False
                        break
                
                if not safe:
                    continue
                
                # If safe, calculate new L
                # Apply new residue
                # We need to update `covered` array.
                # This is tricky without counts.
                # Let's just assume it's safe and try to see if L extends.
                # We can simulate by creating a temporary covered array?
                # Too slow.
                
                # Let's just accept the change if it covers current_L+1 and is safe.
                # We will update the class and re-evaluate L globally.
                
                # Actually, let's just do the swap and check L.
                # To update `covered` correctly, we need to know what was lost.
                # Since we checked "safe", nothing in [1, current_L] is lost.
                # So we just add the new coverage.
                
                # Add new coverage
                mask_new = (indices % p) == r_new
                covered[1:L_max+1][mask_new] = True
                
                # Calculate new L
                new_L = get_L(covered)
                
                if new_L > best_L_new:
                    best_L_new = new_L
                    best_r_new = r_new
                else:
                    # Revert
                    covered = old_covered
            
            if best_r_new != r:
                selected_classes[i] = [p, best_r_new]
                current_L = best_L_new
                improved = True
        
        # 2. Prime Swapping
        # Try to replace a used prime with an unused prime
        unused_primes = [p for p in primes if p not in used_primes]
        
        for i in range(len(selected_classes)):
            if time.time() > deadline - 5:
                break
                
            p_old, r_old = selected_classes[i]
            cost_old = Fraction(1, p_old)
            
            for p_new in unused_primes:
                if time.time() > deadline - 5:
                    break
                    
                cost_new = Fraction(1, p_new)
                
                if current_sum - cost_old + cost_new > Fraction(1, 1):
                    continue
                
                # Try to find best residue for p_new that covers current_L + 1
                if (current_L + 1) % p_new == 0:
                    r_new = 0
                else:
                    r_new = (current_L + 1) % p_new
                
                # Check if swap is safe (doesn't uncover [1, current_L])
                # This requires checking if any number in [1, current_L] covered by (p_old, r_old)
                # is NOT covered by others.
                
                safe = True
                start = r_old if r_old != 0 else p_old
                for y in range(start, current_L + 1, p_old):
                    if not covered[y]:
                        continue
                    is_covered_by_other = False
                    for j in range(len(selected_classes)):
                        if j == i:
                            continue
                        p_j, r_j = selected_classes[j]
                        if y % p_j == r_j:
                            is_covered_by_other = True
                            break
                    if not is_covered_by_other:
                        safe = False
                        break
                
                if not safe:
                    continue
                
                # Perform swap
                old_covered = covered.copy()
                
                # Remove old
                # We can't easily remove. 
                # But if safe, we know [1, current_L] is covered by others.
                # So we just add new coverage.
                
                mask_new = (indices % p_new) == r_new
                covered[1:L_max+1][mask_new] = True
                
                new_L = get_L(covered)
                
                if new_L > current_L:
                    # Accept swap
                    selected_classes[i] = [p_new, r_new]
                    used_primes.remove(p_old)
                    used_primes.add(p_new)
                    current_sum = current_sum - cost_old + cost_new
                    current_L = new_L
                    improved = True
                else:
                    covered = old_covered
    
    # Final check
    # The score is L.
    # We return the classes.
    
    return selected_classes

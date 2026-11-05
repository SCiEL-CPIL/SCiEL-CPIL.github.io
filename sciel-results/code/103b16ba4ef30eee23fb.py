import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # 1. Precompute primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # 2. Generate candidate sets of primes
    # We want sets where sum(1/p) <= 1.
    
    # Strategy: 
    # 1. Start with all primes.
    # 2. Sort by reciprocal size (descending) -> 2, 3, 5, 7, ...
    # 3. Greedily add primes to the set if sum(1/p) + 1/p_new <= 1.
    # This maximizes the number of small primes included.
    
    def generate_greedy_set():
        # Sort primes by 1/p descending (i.e., p ascending)
        sorted_primes = sorted(primes)
        current_sum = 0
        selected = []
        for p in sorted_primes:
            if current_sum + 1.0/p <= 1.0:
                selected.append(p)
                current_sum += 1.0/p
        return selected
    
    # This will select 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97...
    # Let's check the sum for the first few.
    # 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23 + 1/29 + 1/31 + 1/37 + 1/41 + 1/43 + 1/47 + 1/53 + 1/59 + 1/61 + 1/67 + 1/71 + 1/73 + 1/79 + 1/83 + 1/89 + 1/97
    # It will stop when the next prime doesn't fit.
    
    candidate_sets = []
    
    # Candidate 1: Greedy selection of smallest primes
    set_1 = generate_greedy_set()
    candidate_sets.append(set_1)
    
    # Candidate 2: Exclude 2, try to fit more others?
    # If we exclude 2, we have 1/2 = 0.5 budget left.
    # We can fit 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113...
    # Sum of 1/p for p>=3 is ~1.45. So we can fit many.
    # But excluding 2 means we lose the class mod 2.
    # However, we might be able to cover more with more classes.
    
    def generate_set_excluding(excluded):
        sorted_primes = sorted(primes)
        current_sum = 0
        selected = []
        for p in sorted_primes:
            if p in excluded:
                continue
            if current_sum + 1.0/p <= 1.0:
                selected.append(p)
                current_sum += 1.0/p
        return selected
    
    set_2 = generate_set_excluding({2})
    candidate_sets.append(set_2)
    
    set_3 = generate_set_excluding({2, 3})
    candidate_sets.append(set_3)
    
    # 3. Evaluate each candidate using Local Search
    max_L_target = 100000
    best_L = 0
    best_classes = []
    
    for P in candidate_sets:
        if time.time() > deadline - 10:
            break
            
        P_sorted = sorted(P)
        residues = {p: 0 for p in P_sorted}
        
        # Initial Greedy Assignment
        covered = np.zeros(max_L_target + 1, dtype=bool)
        
        # Order primes by size? Or by potential gain?
        # Let's try ascending primes first.
        for p in P_sorted:
            best_r = 0
            best_gain = -1
            
            for r in range(p):
                start = r if r > 0 else p
                if start > max_L_target:
                    continue
                idx = np.arange(start, max_L_target + 1, p)
                gain = np.sum(~covered[idx])
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            residues[p] = best_r
            start = best_r if best_r > 0 else p
            idx = np.arange(start, max_L_target + 1, p)
            covered[idx] = True
            
        # Calculate initial L
        def get_L(covered):
            uncovered_indices = np.where(~covered[1:])[0]
            if len(uncovered_indices) == 0:
                return max_L_target
            return uncovered_indices[0]
            
        current_L = get_L(covered)
        
        # Local Search: Improve by re-optimizing residues
        # We can do a few passes of "best improvement"
        # For each prime, try all residues and pick the one that maximizes coverage.
        # This is expensive, so we limit iterations.
        
        max_iter = 5
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            improved = False
            # Randomize order of primes to escape local minima?
            # Or just iterate.
            for p in P_sorted:
                old_r = residues[p]
                
                # Find best r
                best_r = old_r
                best_gain = -1
                
                # We need to calculate gain relative to current coverage.
                # Current coverage includes p's contribution.
                # To find gain of changing r, we should:
                # 1. Remove p's current contribution
                # 2. Try all r, calculate gain
                # 3. Keep the best
                
                # Remove current
                old_start = old_r if old_r > 0 else p
                if old_start <= max_L_target:
                    old_idx = np.arange(old_start, max_L_target + 1, p)
                    # We can't just set to False because other primes might cover these.
                    # So we need a count array or re-evaluate.
                    
                    # Simpler: Just calculate gain for each r assuming p is NOT present.
                    # Then compare.
                    # But "gain" is the number of NEWLY covered elements.
                    # If we remove p, some elements become uncovered.
                    # Then adding p with new r covers some of them and maybe more.
                    
                    # Let's use a helper to calculate coverage with a specific set of residues.
                    # That's O(N * P) per check, which is too slow.
                    
                    # Alternative:
                    # Maintain a count array `counts` where counts[i] is the number of classes covering i.
                    # covered[i] is counts[i] > 0.
                    # When changing p from r_old to r_new:
                    # 1. Decrement counts for r_old.
                    # 2. Increment counts for r_new.
                    # 3. Check if any counts went from 0 to 1 (gain) or 1 to 0 (loss).
                    # 4. Net gain = gain - loss.
                    
                    pass # This logic is complex to implement efficiently in the loop without pre-allocated arrays.
                    
            # Since the simple local search with counts is complex to code quickly and safely,
            # let's stick to the initial greedy which is decent, but try different prime sets.
            
        if current_L > best_L:
            best_L = current_L
            best_classes = [[p, residues[p]] for p in P_sorted]
            
    # Fallback if nothing found (should not happen)
    if not best_classes:
        # Use the first candidate set with default residues
        P = candidate_sets[0]
        best_classes = [[p, 0] for p in P]
        
    # Final Verification with Fractions
    s = 0
    for p, r in best_classes:
        s += Fraction(1, p)
    if s > 1:
        # Fallback to a known valid small set
        return [[p, 0] for p in [13, 17, 19]]
        
    return best_classes

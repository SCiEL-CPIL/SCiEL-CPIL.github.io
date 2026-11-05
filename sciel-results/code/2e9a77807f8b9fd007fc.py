import time
from fractions import Fraction
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # Define a few candidate sets of primes
    # Strategy: Include small primes for density, then add larger primes to fill budget
    # The sum of reciprocals of primes 2,3,5,7,11,13,17,19,23 is approx 0.5 + 0.33 + 0.2 + 0.14 + 0.09 + 0.077 + 0.058 + 0.052 + 0.043 = 1.49 (too high)
    # Let's check sums:
    # 2,3,5,7,11,13: 0.5+0.333+0.2+0.142+0.0909+0.0769 = 1.34 (too high)
    # 2,3,5,7,11: 0.5+0.333+0.2+0.142+0.0909 = 1.26 (too high)
    # 2,3,5,7: 0.5+0.333+0.2+0.142 = 1.17 (too high)
    # 2,3,5: 0.5+0.333+0.2 = 1.03 (too high)
    # 2,3: 0.833
    # 2,5: 0.7
    # 3,5: 0.533
    # 2,3,7: 0.976
    # 2,3,11: 0.924
    # 2,3,13: 0.909
    # 2,5,7: 0.842
    # 2,5,11: 0.790
    # 3,5,7: 0.675
    
    # We need sum <= 1.
    # Good sets:
    # Set A: [2, 3, 7, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79]
    # Let's just generate sets by starting with small primes and adding largest possible primes? No, smallest possible primes give better coverage.
    
    candidate_sets = []
    
    # Base sets of small primes that fit in budget
    base_sets = [
        [2, 3, 7], # 0.5+0.333+0.142 = 0.975
        [2, 3, 11], # 0.5+0.333+0.0909 = 0.924
        [2, 3, 13], # 0.5+0.333+0.0769 = 0.909
        [2, 5, 7], # 0.5+0.2+0.142 = 0.842
        [2, 5, 11], # 0.5+0.2+0.0909 = 0.791
        [2, 5, 13], # 0.5+0.2+0.0769 = 0.777
        [3, 5, 7], # 0.333+0.2+0.142 = 0.675
        [3, 5, 11], # 0.333+0.2+0.0909 = 0.624
        [3, 5, 13], # 0.333+0.2+0.0769 = 0.610
        [2, 3, 7, 17], # 0.975 + 0.058 = 1.033 (Too high)
        [2, 3, 7, 19], # 0.975 + 0.052 = 1.027 (Too high)
        [2, 3, 11, 17], # 0.924 + 0.058 = 0.982
        [2, 3, 11, 19], # 0.924 + 0.052 = 0.976
        [2, 3, 13, 17], # 0.909 + 0.058 = 0.967
        [2, 3, 13, 19], # 0.909 + 0.052 = 0.961
        [2, 5, 7, 11], # 0.842 + 0.0909 = 0.933
        [2, 5, 7, 13], # 0.842 + 0.0769 = 0.919
        [2, 5, 11, 13], # 0.791 + 0.0769 = 0.868
        [3, 5, 7, 11], # 0.675 + 0.0909 = 0.766
        [3, 5, 7, 13], # 0.675 + 0.0769 = 0.752
        [3, 5, 11, 13], # 0.624 + 0.0769 = 0.701
    ]
    
    # Filter valid base sets
    valid_base_sets = []
    for bs in base_sets:
        s = sum(1.0/p for p in bs)
        if s <= 1.0:
            valid_base_sets.append(bs)
            
    # Expand each base set by adding smallest possible primes that fit in remaining budget
    # This maximizes density
    for bs in valid_base_sets:
        current_set = list(bs)
        current_sum = sum(1.0/p for p in bs)
        # Add primes in ascending order if they fit
        for p in primes:
            if p in current_set:
                continue
            if current_sum + 1.0/p <= 1.0:
                current_set.append(p)
                current_sum += 1.0/p
        
        # Avoid duplicates
        if current_set not in candidate_sets:
            candidate_sets.append(current_set)
            
    # Also try a set with just [2, 3] and fill with smallest primes
    # And [2, 5] and fill
    # The above logic should cover this if we include [2,3] and [2,5] in base_sets?
    # [2,3] sum=0.833. Next prime 5: 0.833+0.2=1.033 (No). Next 7: 0.833+0.142=0.975 (Yes).
    # So [2,3] expands to [2,3,7, ...]. This is already in base_sets as [2,3,7].
    
    # Add a few random-ish sets? No, deterministic is better for reproducibility and speed.
    
    best_L = 0
    best_classes = []
    MAX_L = 100000
    
    def evaluate_set(P):
        P = sorted(P)
        n_primes = len(P)
        
        # Arrays for coverage
        # cover_count[i] = number of classes covering i
        cover_count = [0] * (MAX_L + 1)
        residues = {}
        
        # Greedy Phase
        # We want to cover 1..L
        # Start with no residues assigned.
        # We iterate n from 1 to MAX_L.
        # If cover_count[n] == 0, we need to assign a residue to some prime p to cover n.
        # Which p? One that maximizes the next gap.
        
        # To make this fast, we can precompute the "next gap" for each possible (p, r) pair?
        # That's expensive.
        
        # Alternative: Simple Greedy
        # Try to cover n with the prime p that is "best".
        # A good heuristic: choose p that minimizes the next uncovered number.
        
        # Let's use a simpler heuristic first:
        # For the first gap n, try all unassigned primes p.
        # Set r = n % p.
        # Calculate the next gap if we use this (p, r).
        # Pick the one with the largest next gap.
        
        # To speed up "calculate next gap", we can just scan forward.
        # Since MAX_L is 100000, and we might have ~20-30 primes, this is manageable.
        
        assigned = set()
        
        n = 1
        while n <= MAX_L:
            if time.time() > deadline - 5:
                break
                
            if cover_count[n] > 0:
                n += 1
                continue
                
            # n is a gap
            best_p = None
            best_next_gap = 0
            best_r = None
            
            for p in P:
                if p in assigned:
                    continue
                    
                r = n % p
                
                # Estimate next gap
                # Check k = n, n+1, ... until we find one not covered by existing classes AND not covered by (p, r)
                # Note: existing classes are fixed. (p, r) is new.
                
                next_gap = n + 1
                # Optimization: if p is large, the step is large, so gaps are likely close.
                # If p is small, step is small, gaps might be further.
                
                # We can break early if next_gap exceeds current best_next_gap + some margin?
                # No, we need the exact next gap to compare.
                
                # But we can stop checking if next_gap > best_next_gap + 1000? 
                # Actually, we just want the max.
                
                # Let's just scan up to best_next_gap + 100 or so? 
                # If we find a gap beyond best_next_gap, we update.
                # If we don't find a gap in the window [n, best_next_gap], it's a candidate.
                
                # Simpler: Just scan until we find a gap.
                # This is O(GapSize).
                
                k = n
                while k <= MAX_L:
                    # Check if k is covered by existing
                    if cover_count[k] == 0:
                        # Check if k is covered by new (p, r)
                        if k % p == r:
                            k += 1
                            continue
                        else:
                            next_gap = k
                            break
                    else:
                        k += 1
                        
                if k > MAX_L:
                    next_gap = MAX_L + 1
                    
                if next_gap > best_next_gap:
                    best_next_gap = next_gap
                    best_p = p
                    best_r = r
                    
            if best_p is None:
                break
                
            # Assign best_p
            residues[best_p] = best_r
            assigned.add(best_p)
            
            # Update cover_count
            for k in range(best_r if best_r > 0 else best_p, MAX_L + 1, best_p):
                cover_count[k] += 1
                
            n += 1
            
        # Calculate L
        L = 0
        for i in range(1, MAX_L + 1):
            if cover_count[i] > 0:
                L = i
            else:
                break
                
        # Local Search
        # Try to improve L by shifting residues of assigned primes.
        # G = L + 1
        G = L + 1
        
        if G <= MAX_L:
            # Iterate local search
            # We will try to cover G by changing residue of some p in assigned.
            # New r_p = G % p.
            # Check if this breaks any k < G.
            # k is broken if cover_count[k] == 1 and k % p == old_r_p.
            
            improved = True
            iterations = 0
            max_iterations = 500 # Limit iterations
            
            while improved and iterations < max_iterations and time.time() < deadline - 10:
                improved = False
                iterations += 1
                
                for p in list(assigned):
                    if time.time() > deadline - 10:
                        break
                        
                    old_r = residues[p]
                    new_r = G % p
                    
                    if old_r == new_r:
                        continue
                        
                    # Check validity
                    valid = True
                    # Check k in old_r class, k < G
                    # k = old_r, old_r+p, ... < G
                    # If cover_count[k] == 1, then k is only covered by p.
                    # If we change p's residue, k becomes uncovered.
                    
                    start_k = old_r if old_r > 0 else p
                    for k in range(start_k, G, p):
                        if cover_count[k] == 1:
                            valid = False
                            break
                            
                    if valid:
                        # Apply change
                        # Update cover_count
                        # Decrement for old_r class
                        for k in range(start_k, MAX_L + 1, p):
                            cover_count[k] -= 1
                            
                        # Increment for new_r class
                        start_k_new = new_r if new_r > 0 else p
                        for k in range(start_k_new, MAX_L + 1, p):
                            cover_count[k] += 1
                            
                        residues[p] = new_r
                        
                        # Update G
                        # G is now covered. Find next gap.
                        G += 1
                        while G <= MAX_L and cover_count[G] > 0:
                            G += 1
                            
                        improved = True
                        break # Restart loop with new G
                        
        # Final L calculation
        L = 0
        for i in range(1, MAX_L + 1):
            if cover_count[i] > 0:
                L = i
            else:
                break
                
        return L, [[p, residues[p]] for p in P if p in residues]

    for P in candidate_sets:
        if time.time() > deadline - 15:
            break
            
        L, classes = evaluate_set(P)
        if L > best_L:
            best_L = L
            best_classes = classes
            
    # Verify sum
    s = 0
    for p, r in best_classes:
        s += Fraction(1, p)
    if s > 1:
        # Fallback
        return []
        
    return best_classes

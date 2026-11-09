import time
import random
import numpy as np
from fractions import Fraction

MAX_L = 100000
DEADLINE = 110.0 # Leave 10s for cleanup/return

def construct():
    start_time = time.time()
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, p in enumerate(is_prime) if p]
    
    primes = sieve(1000)
    prime_set = set(primes)
    
    # Initial solution: Greedy
    # Start with small primes, assign residues to maximize early coverage
    # The seed program uses i-th prime from 13.
    # Let's try a greedy that adds primes one by one, choosing the residue that maximizes L.
    
    current_classes = []
    current_sum_frac = Fraction(0, 1)
    current_sum_float = 0.0
    
    # We need to track coverage to calculate L
    # We will use a numpy array for speed
    covered = np.zeros(MAX_L + 1, dtype=bool)
    
    def calc_L(covered_arr):
        # Find first False index >= 1
        # ~covered_arr gives True where not covered
        # We want the first True in ~covered_arr[1:]
        # If all covered, return MAX_L
        if np.all(covered_arr[1:]):
            return MAX_L
        idx = np.argmax(~covered_arr[1:])
        return idx + 1
        
    # Helper to update covered array for a prime p and residue r
    # add=True means add the class, add=False means remove
    def update_coverage(p, r, add, cov_arr):
        # Indices: r, r+p, r+2p ... <= MAX_L
        # Note: r can be 0. Indices are 1-based in our logic?
        # Task: 1..L.
        # If r=0, it covers p, 2p, ...
        # If r=1, it covers 1, 1+p, ...
        # So indices are r + k*p.
        # We need indices >= 1.
        start = r
        if start < 1:
            # r=0 means 0, p, 2p... but 0 is not in 1..L
            # So first valid is p
            start = p
        # Actually, if r=0, the numbers are 0, p, 2p...
        # We care about 1..MAX_L.
        # So indices are max(1, r) + k*p?
        # No, if r=0, indices are p, 2p...
        # If r=1, indices are 1, 1+p...
        # General: k*p + r.
        # We want k*p + r >= 1.
        
        indices = np.arange(start, MAX_L + 1, p)
        # Wait, if r=0, start should be p? 
        # 0 is not covered. p is covered.
        # np.arange(0, MAX_L+1, p) gives 0, p, 2p...
        # We should exclude 0.
        if r == 0:
             indices = indices[1:] # Remove 0
        cov_arr[indices] = add
             
    # Greedy Initialization
    # Try to build a set of classes.
    # We have budget 1.
    # Small primes are "cheap" in terms of count but "expensive" in terms of 1/p?
    # 1/2 = 0.5, 1/3 = 0.33.
    # We want to cover 1..L.
    
    # Let's start with the seed-like approach but optimize residues.
    # Seed: primes from 13.
    # Let's try a greedy that picks the best prime/residue pair at each step.
    # This is expensive: O(P * R * L).
    
    # Alternative: Start with a known good set (e.g. from literature or seed) and improve.
    # Seed set:
    # primes = [13, 17, 19, ...]
    # Let's just use the seed's primes and optimize residues via local search.
    
    seed_primes = [p for p in primes if p >= 13]
    # Take enough primes to stay under budget 1
    # Sum 1/p for p>=13.
    # 1/13+1/17+1/19+...
    # Let's pick primes until sum > 1
    initial_primes = []
    s = Fraction(0,1)
    for p in seed_primes:
        if s + Fraction(1, p) <= Fraction(1, 1):
            initial_primes.append(p)
            s += Fraction(1, p)
        else:
            break
            
    # Initialize covered
    for p in initial_primes:
        # Try to find best residue for this p given current coverage
        best_r = 0
        best_l = -1
        # We need to test all r
        # This is O(P * P * L) which is too slow for init if P is large.
        # Let's just use r=1 for all initially, or random.
        # Or use the seed's logic: (i+1) % p
        pass
        
    # Let's just use the seed's exact configuration as start
    current_classes = []
    current_sum_frac = Fraction(0,1)
    for i, p in enumerate(initial_primes):
        r = (i + 1) % p
        current_classes.append([p, r])
        current_sum_frac += Fraction(1, p)
        
    # Build initial covered array
    covered = np.zeros(MAX_L + 1, dtype=bool)
    for p, r in current_classes:
        update_coverage(p, r, True, covered)
        
    current_L = calc_L(covered)
    
    # Local Search
    # We will try to improve L.
    # Moves:
    # 1. Change residue of an existing prime.
    # 2. Swap a prime with another prime (if budget allows).
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < DEADLINE and no_improve_count < max_no_improve:
        # Pick a random move
        move_type = random.choice(['residue', 'swap'])
        
        if move_type == 'residue':
            # Pick a random class
            idx = random.randint(0, len(current_classes) - 1)
            p, old_r = current_classes[idx]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
                
            # Check if changing residue improves L
            # We need to simulate the change
            # Create a copy of covered? Or modify and revert?
            # Modifying and reverting is faster.
            
            # Remove old
            update_coverage(p, old_r, False, covered)
            # Add new
            update_coverage(p, new_r, True, covered)
            
            new_L = calc_L(covered)
            
            if new_L > current_L:
                current_L = new_L
                current_classes[idx][1] = new_r
                no_improve_count = 0
            else:
                # Revert
                update_coverage(p, new_r, False, covered)
                update_coverage(p, old_r, True, covered)
                no_improve_count += 1
                
        else: # swap
            # Pick a prime to remove and a prime to add
            if len(current_classes) == 0:
                continue
            idx_remove = random.randint(0, len(current_classes) - 1)
            p_remove, r_remove = current_classes[idx_remove]
            
            # Find a prime to add
            # Candidate primes: not in current set
            current_prime_set = set(c[0] for c in current_classes)
            candidates = [p for p in primes if p not in current_prime_set]
            if not candidates:
                continue
                
            p_add = random.choice(candidates)
            
            # Check budget
            new_sum = current_sum_frac - Fraction(1, p_remove) + Fraction(1, p_add)
            if new_sum > Fraction(1, 1):
                continue
                
            # Simulate swap
            # Remove p_remove, r_remove
            update_coverage(p_remove, r_remove, False, covered)
            
            # Add p_add, r_add (try best residue for p_add?)
            # For simplicity, try a few random residues for p_add
            best_r_add = -1
            best_l_add = -1
            
            # Try 5 random residues
            for _ in range(5):
                r_try = random.randint(0, p_add - 1)
                update_coverage(p_add, r_try, True, covered)
                l_try = calc_L(covered)
                update_coverage(p_add, r_try, False, covered)
                
                if l_try > best_l_add:
                    best_l_add = l_try
                    best_r_add = r_try
                    
            if best_l_add > current_L:
                # Commit the swap
                current_classes.pop(idx_remove)
                current_classes.append([p_add, best_r_add])
                current_sum_frac = new_sum
                update_coverage(p_add, best_r_add, True, covered)
                current_L = best_l_add
                no_improve_count = 0
            else:
                no_improve_count += 1
                
    return current_classes

# Test
# print(construct())

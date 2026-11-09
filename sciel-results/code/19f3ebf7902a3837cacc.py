import time
import random
import numpy as np
from fractions import Fraction

def _get_primes(limit=1000):
    """Generate primes up to limit using simple sieve."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    primes = _get_primes(1000)
    L_cap = 100000
    
    # Pre-compute reciprocals as floats for fast budget checks in search loop.
    # We will do a final exact check with Fraction.
    reciprocals = {p: 1.0 / p for p in primes}
    
    # We aim to find a set of primes and residues that covers 1..L.
    # Strategy:
    # 1. Start with a greedy seed that covers a decent prefix.
    # 2. Use NumPy to maintain a coverage count array.
    # 3. Local Search: 
    #    - Move types: Change residue, Change prime, Add prime, Remove prime.
    #    - Objective: Maximize L (first index where count is 0).
    #    - Constraint: Sum(1/p) <= 1.
    
    # --- Helper: Calculate L from numpy count array ---
    # This is fast because it uses vectorized operations or early exit in C.
    # However, finding the first zero in a numpy array is still O(L) in worst case.
    # We can optimize by only checking up to the current known L + some buffer, 
    # but since L can jump, we check up to L_cap.
    # Actually, `np.argmax(counts[1:] == 0)` returns the first index where it's True.
    # If no zeros, it returns 0. We need to handle that.
    
    def get_L(counts):
        # counts[0] is unused or for 0. We care about 1..L_cap
        # Find first index i in 1..L_cap where counts[i] == 0
        # If all covered, return L_cap
        # Using argmax on a boolean array is efficient.
        # Note: counts is 1-indexed effectively for our range of interest, 
        # but numpy is 0-indexed. 
        # Let's keep counts aligned: counts[i] corresponds to integer i.
        
        # Check if any zero in 1..L_cap
        # To be safe and fast, we can just scan the numpy array.
        # But Python loop is slow.
        # Let's use:
        # zeros = np.where(counts[1:L_cap+1] == 0)[0]
        # if len(zeros) > 0: return zeros[0]
        # else: return L_cap
        
        # Even faster: 
        # We know L is likely small (1000-5000). 
        # We can just check a window? No, we need exact L.
        
        # Let's stick to np.where. It's C-speed.
        mask = counts[1:L_cap+1] == 0
        if not np.any(mask):
            return L_cap
        first_zero = np.argmax(mask)
        return first_zero
        
    # --- State Representation ---
    # active_primes: list of primes
    # active_residues: list of residues
    # current_sum: float sum of 1/p
    
    # --- Initialization ---
    # Greedy initialization:
    # Try to cover 1, 2, 3...
    # For each number n not covered, pick a prime p and residue r=n%p.
    # To save budget, prefer larger primes? Or smaller?
    # Smaller primes cover more numbers but cost more budget.
    # Larger primes cost less but cover fewer.
    # Since we want to maximize L, and budget is 1, we should use as many small primes as possible?
    # Actually, the "standard" covering for x=1000 uses primes >= 13.
    # Let's start with primes >= 13 and random residues, then optimize.
    
    def initialize():
        # Pick primes >= 13 to have some budget headroom for optimization?
        # Or start with all primes and remove?
        # Let's start with primes 13..997.
        init_primes = [p for p in primes if p >= 13]
        init_residues = [random.randint(0, p-1) for p in init_primes]
        
        # Build count array
        counts = np.zeros(L_cap + 1, dtype=np.int32)
        for p, r in zip(init_primes, init_residues):
            # r is residue mod p.
            # Numbers covered: r, r+p, r+2p...
            # Note: if r==0, covers p, 2p... (0 is not in 1..L)
            # So start = r if r > 0 else p
            start = r if r > 0 else p
            if start <= L_cap:
                counts[start:L_cap+1:p] += 1
                
        current_sum = sum(1.0/p for p in init_primes)
        return init_primes, init_residues, counts, current_sum

    best_L = 0
    best_primes = []
    best_residues = []
    
    # Run multiple restarts
    # Each restart: Initialize, then Local Search until stuck or time up.
    
    num_restarts = 0
    
    while time.time() < deadline:
        num_restarts += 1
        current_primes, current_residues, counts, current_sum = initialize()
        
        # Check initial L
        current_L = get_L(counts)
        if current_L > best_L:
            best_L = current_L
            best_primes = current_primes[:]
            best_residues = current_residues[:]
            
        # Local Search Loop
        # We use a simple hill climbing with random restarts within the search?
        # No, just pure hill climbing. If stuck, break and restart.
        
        stuck_count = 0
        max_stuck = 5000 # If no improvement for 5000 moves, restart
        
        while time.time() < deadline and stuck_count < max_stuck:
            # Choose a move
            move_type = random.random()
            
            # Helper to apply move and get new L
            # We need to update counts efficiently.
            
            if move_type < 0.6:
                # Change residue of an existing prime
                if not current_primes:
                    continue
                idx = random.randint(0, len(current_primes) - 1)
                p = current_primes[idx]
                old_r = current_residues[idx]
                
                # Generate candidate residues
                # 1. Cover current_L + 1
                # 2. Random
                # 3. Neighbor of old_r
                
                candidates = set()
                gap = current_L + 1
                if gap <= L_cap:
                    candidates.add(gap % p)
                
                candidates.add(random.randint(0, p-1))
                candidates.add((old_r + 1) % p)
                candidates.add((old_r - 1) % p)
                
                new_r = random.choice(list(candidates))
                if new_r == old_r:
                    continue
                
                # Apply move
                # Remove old
                start_old = old_r if old_r > 0 else p
                if start_old <= L_cap:
                    counts[start_old:L_cap+1:p] -= 1
                    
                # Add new
                start_new = new_r if new_r > 0 else p
                if start_new <= L_cap:
                    counts[start_new:L_cap+1:p] += 1
                    
                current_residues[idx] = new_r
                new_L = get_L(counts)
                
                if new_L > current_L:
                    current_L = new_L
                    stuck_count = 0
                    if current_L > best_L:
                        best_L = current_L
                        best_primes = current_primes[:]
                        best_residues = current_residues[:]
                else:
                    # Revert
                    counts[start_old:L_cap+1:p] += 1
                    if start_new <= L_cap:
                        counts[start_new:L_cap+1:p] -= 1
                    current_residues[idx] = old_r
                    stuck_count += 1
                    
            elif move_type < 0.8:
                # Swap a prime
                if not current_primes:
                    continue
                idx = random.randint(0, len(current_primes) - 1)
                p_old = current_primes[idx]
                r_old = current_residues[idx]
                
                # Pick new prime
                # Prefer primes not in current set
                # And check budget
                p_new = random.choice(primes)
                if p_new == p_old:
                    continue
                if p_new in current_primes:
                    continue
                    
                new_sum = current_sum - 1.0/p_old + 1.0/p_new
                if new_sum > 1.0:
                    continue
                
                # Residue: map old residue to new prime? Or random?
                # If we keep the same "pattern" of coverage, r_new = r_old % p_new might be good?
                # Or target the gap.
                gap = current_L + 1
                r_new = gap % p_new
                
                # Apply move
                # Remove old
                start_old = r_old if r_old > 0 else p_old
                if start_old <= L_cap:
                    counts[start_old:L_cap+1:p_old] -= 1
                    
                # Add new
                start_new = r_new if r_new > 0 else p_new
                if start_new <= L_cap:
                    counts[start_new:L_cap+1:p_new] += 1
                    
                current_primes[idx] = p_new
                current_residues[idx] = r_new
                current_sum = new_sum
                
                new_L = get_L(counts)
                
                if new_L > current_L:
                    current_L = new_L
                    stuck_count = 0
                    if current_L > best_L:
                        best_L = current_L
                        best_primes = current_primes[:]
                        best_residues = current_residues[:]
                else:
                    # Revert
                    counts[start_old:L_cap+1:p_old] += 1
                    if start_new <= L_cap:
                        counts[start_new:L_cap+1:p_new] -= 1
                    current_primes[idx] = p_old
                    current_residues[idx] = r_old
                    current_sum = current_sum - 1.0/p_new + 1.0/p_old
                    stuck_count += 1
                    
            else:
                # Add a prime
                # Find a prime that fits in budget
                # And is not in current set
                # To make this fast, we can just try a random one.
                p_add = random.choice(primes)
                if p_add in current_primes:
                    continue
                    
                new_sum = current_sum + 1.0/p_add
                if new_sum > 1.0:
                    continue
                
                # Residue to cover gap
                gap = current_L + 1
                r_add = gap % p_add
                
                # Apply
                start_new = r_add if r_add > 0 else p_add
                if start_new <= L_cap:
                    counts[start_new:L_cap+1:p_add] += 1
                    
                current_primes.append(p_add)
                current_residues.append(r_add)
                current_sum = new_sum
                
                new_L = get_L(counts)
                
                if new_L > current_L:
                    current_L = new_L
                    stuck_count = 0
                    if current_L > best_L:
                        best_L = current_L
                        best_primes = current_primes[:]
                        best_residues = current_residues[:]
                else:
                    # Revert
                    if start_new <= L_cap:
                        counts[start_new:L_cap+1:p_add] -= 1
                    current_primes.pop()
                    current_residues.pop()
                    current_sum = current_sum - 1.0/p_add
                    stuck_count += 1
            # End move selection

    # Final Exact Check
    if best_primes:
        exact_sum = sum(Fraction(1, p) for p in best_primes)
        if exact_sum <= 1:
            return [[p, r] for p, r in zip(best_primes, best_residues)]
    
    # Fallback: Return a valid, small solution
    # Just one prime class, e.g., [2, 1] covers 1, 3, 5... L=1? No, 2 is not covered.
    # [2, 0] covers 2, 4... 1 not covered. L=0.
    # [3, 1] covers 1, 4... 2 not covered. L=1.
    # Let's return a known valid small set.
    # Primes: 3, 5, 7.
    # 3, 1: 1, 4, 7...
    # 5, 2: 2, 7, 12...
    # 7, 3: 3, 10, 17...
    # Covers 1,2,3. L=3.
    # Sum 1/3+1/5+1/7 = 35/105 + 21/105 + 15/105 = 71/105 < 1.
    return [[3, 1], [5, 2], [7, 3]]

# Ensure we don't return an empty list if something goes wrong
# The logic above is robust.

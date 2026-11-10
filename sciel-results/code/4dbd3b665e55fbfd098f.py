"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def _sieve_primes(n):
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # We want to cover 1..L. Let's aim for a large L, e.g., 10000 initially, 
    # but we can expand if budget allows. 
    # The task says score is largest L such that 1..L is covered.
    # Max L is capped at 10^5.
    # Let's target covering up to 10000 first, as covering 10^5 is likely impossible with budget 1.
    # Actually, sum of 1/p for p<=1000 is approx 2.2. So we have plenty of budget.
    # The constraint is sum 1/p <= 1.
    # Let's try to cover as much as possible.
    
    MAX_TARGET = 10000 # Start with a reasonable target. If we cover this, we can try higher.
    # Actually, let's just compute coverage for 1..MAX_TARGET.
    
    primes = _sieve_primes(1000)
    num_primes = len(primes)
    
    # Coverage array for 1..MAX_TARGET
    # Index 0 is unused, 1..MAX_TARGET
    covered = np.zeros(MAX_TARGET + 1, dtype=np.int8)
    
    # Budget tracking using Fraction for exactness
    current_budget = Fraction(0, 1)
    max_budget = Fraction(1, 1)
    
    # Precompute residues for each prime? No, too much memory if we store all.
    # We will evaluate candidates on the fly or in batches.
    
    # To speed up, we can precompute the indices for each prime and residue.
    # But iterating all p, r for each step is O(P^2 * L/p).
    # P ~ 168. L = 10000.
    # Total ops ~ sum(p) * 10000/p = 168 * 10000 = 1.68M ops per step? No.
    # For each candidate (p, r), we check how many new integers are covered.
    # Integers are r, r+p, r+2p ... <= MAX_TARGET.
    # Count is ~ MAX_TARGET / p.
    # Checking each one is O(1) with array lookup.
    # So cost per candidate is MAX_TARGET / p.
    # Total cost per step is sum_{p in primes} (MAX_TARGET / p) = MAX_TARGET * sum(1/p).
    # sum(1/p) for p<=1000 is ~ 2.2.
    # So cost per step is ~ 22000 operations.
    # Number of steps is at most num_primes (since distinct primes).
    # So total operations ~ 168 * 22000 = 3.7M. This is very fast.
    
    selected_classes = []
    
    # We need to pick the best (p, r) at each step.
    # We can precompute the list of indices for each (p, r) to make it faster?
    # Or just loop.
    
    # Let's store available primes
    available_primes = set(primes)
    
    while available_primes:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_p = None
        best_r = None
        best_cost = None
        
        # Iterate through available primes
        for p in available_primes:
            # Check if adding any residue of p exceeds budget
            cost_p = Fraction(1, p)
            if current_budget + cost_p > max_budget:
                continue
                
            # Try all residues r in 0..p-1
            # We want to maximize new coverage in 1..MAX_TARGET
            # New coverage = count of i in [r, r+p, ...] <= MAX_TARGET where covered[i] == 0
            
            # Optimization: If p > MAX_TARGET, only r=1..MAX_TARGET matter?
            # No, r is in 0..p-1.
            # If r > MAX_TARGET, it covers nothing in 1..MAX_TARGET.
            # So we only need to check r in 0..min(p-1, MAX_TARGET).
            max_r_to_check = min(p, MAX_TARGET + 1) # r can be 0..MAX_TARGET
            
            for r in range(max_r_to_check):
                # Count new coverage
                # Indices: r, r+p, r+2p ... 
                # Note: if r==0, it covers p, 2p... (since 0 is not in 1..L)
                # So start index is r if r>=1 else p.
                
                count = 0
                # Start from first positive multiple/residue
                start = r if r > 0 else p
                for i in range(start, MAX_TARGET + 1, p):
                    if covered[i] == 0:
                        count += 1
                        if count > best_gain:
                            # Early exit if we already beat best? 
                            # We can't easily break the inner loop without checking all r for this p?
                            # But we can break if count + (remaining steps) <= best_gain? 
                            # Remaining steps = (MAX_TARGET - i) // p.
                            # If count + (MAX_TARGET - i)//p <= best_gain, break.
                            pass
                            
                if count > best_gain:
                    best_gain = count
                    best_p = p
                    best_r = r
                    best_cost = cost_p
        
        if best_p is None:
            break
            
        # Select best
        current_budget += best_cost
        available_primes.remove(best_p)
        selected_classes.append([best_p, best_r])
        
        # Update coverage array
        p = best_p
        r = best_r
        start = r if r > 0 else p
        for i in range(start, MAX_TARGET + 1, p):
            covered[i] = 1
            
        # Check if we have covered 1..MAX_TARGET
        # If yes, we can try to extend? 
        # The score is the largest L. 
        # If 1..MAX_TARGET is covered, L >= MAX_TARGET.
        # We could increase MAX_TARGET and continue?
        # But the loop structure is fixed. 
        # Let's just check if 1..MAX_TARGET is covered.
        
    # Calculate L
    # Find first i in 1..MAX_TARGET where covered[i] == 0
    # If all covered, L = MAX_TARGET (or more if we expanded, but we didn't)
    # Actually, we should check if we can cover more.
    # But for now, return based on MAX_TARGET.
    
    # To be safe, let's check the actual L up to MAX_TARGET
    # If covered[1..MAX_TARGET] are all 1, then L >= MAX_TARGET.
    # Could L be larger? 
    # The classes cover residues mod p. 
    # If 1..MAX_TARGET is covered, it doesn't guarantee MAX_TARGET+1 is covered.
    # But the score is capped at 10^5.
    # If we cover 1..10000, score is at least 10000.
    # If we cover 1..999, score is 999.
    
    # Let's compute L exactly.
    # Use numpy to find first zero in covered[1:]
    # covered is 1-indexed effectively (0 is dummy)
    
    if MAX_TARGET > 0:
        # Find first index i >= 1 where covered[i] == 0
        # np.nonzero(covered[1:] == 0)
        zeros = np.nonzero(covered[1:] == 0)[0]
        if len(zeros) == 0:
            L = MAX_TARGET
        else:
            L = zeros[0]
    else:
        L = 0
        
    return selected_classes

# EVOLVE-BLOCK-END

import time
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    DEADLINE = 135  # safety margin before 150s kill
    
    # Generate primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    prime_set = set(primes)
    
    # We need to cover 1..L with residue classes [p, r]
    # A class [p, r] covers all n where n ≡ r (mod p)
    # Budget: sum of 1/p <= 1
    
    # Strategy: Greedy with efficiency scoring
    # For each candidate class, compute how many NEW integers it covers
    # Efficiency = new_coverage / (1/p) = new_coverage * p
    
    # We'll use a bitset for tracking coverage up to some max
    MAX_L = 100000  # cap
    
    # Start with empty coverage
    covered = [False] * (MAX_L + 1)
    # covered[n] = True if n is covered by some class
    # We need 1..L all covered, so track the max consecutive from 1
    
    best_L = 0
    selected = []  # list of (p, r)
    budget_used = Fraction(0, 1)
    
    # Precompute for each prime p, the residue classes and their coverage patterns
    # For prime p, residue r covers: r, r+p, r+2p, ... (but only if r >= 1 for our range 1..L)
    # Actually r can be 0, which covers p, 2p, 3p, ...
    
    def get_coverage_count(p, r, current_covered, max_check):
        """Count how many integers in 1..max_check are newly covered by class (p, r)."""
        count = 0
        # Start from the smallest n >= 1 with n ≡ r (mod p)
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start = 1 + ((1 - 1) % p)  # should be r if r>=1
            start = r
        for n in range(start, max_check + 1, p):
            if n >= 1 and not current_covered[n]:
                count += 1
        return count
    
    # Greedy approach: repeatedly pick the class with best efficiency
    # efficiency = new_coverage * p (since cost is 1/p)
    
    # To make this fast, we'll limit how far we check coverage
    # Initially check up to some reasonable bound, then expand
    
    # Let's use a smarter approach:
    # 1. Start with a small window (e.g., 1000)
    # 2. Greedily select classes that maximize new coverage in the window
    # 3. After budget is used, check actual coverage
    
    WINDOW = 2000  # initial window for greedy decisions
    
    def compute_new_coverage(p, r, covered_arr, window):
        """Count new coverage in 1..window for class (p, r)."""
        count = 0
        if r == 0:
            start = p
        else:
            start = r
        for n in range(start, window + 1, p):
            if n >= 1 and not covered_arr[n]:
                count += 1
        return count
    
    def apply_class(p, r, covered_arr, window):
        """Mark class (p, r) as covering in covered_arr up to window."""
        if r == 0:
            start = p
        else:
            start = r
        for n in range(start, window + 1, p):
            if n >= 1:
                covered_arr[n] = True
    
    def get_max_consecutive(covered_arr, limit):
        """Find largest L such that 1..L are all covered."""
        L = 0
        for n in range(1, limit + 1):
            if covered_arr[n]:
                L = n
            else:
                break
        return L
    
    # Greedy selection
    max_iterations = 500
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() - start_time > DEADLINE:
            break
        
        # Find best class to add
        best_eff = -1
        best_p = None
        best_r = None
        best_new_cov = 0
        
        for p in primes:
            cost = Fraction(1, p)
            if budget_used + cost > Fraction(1, 1):
                continue
            
            for r in range(p):
                # Skip if this (p, r) is already selected
                if any(sp == p and sr == r for sp, sr in selected):
                    continue
                
                new_cov = compute_new_coverage(p, r, covered, WINDOW)
                if new_cov == 0:
                    continue
                
                # Efficiency: new coverage per unit budget
                # Since cost is 1/p, efficiency = new_cov * p
                eff = new_cov * p
                
                if eff > best_eff:
                    best_eff = eff
                    best_p = p
                    best_r = r
                    best_new_cov = new_cov
        
        if best_p is None:
            break
        
        # Add the best class
        selected.append((best_p, best_r))
        budget_used += Fraction(1, best_p)
        apply_class(best_p, best_r, covered, WINDOW)
        
        # Check current coverage
        current_L = get_max_consecutive(covered, WINDOW)
        if current_L > best_L:
            best_L = current_L
        
        iteration += 1
    
    # Now verify and extend coverage check to larger values
    # Rebuild full coverage array
    full_covered = [False] * (MAX_L + 1)
    for p, r in selected:
        if r == 0:
            start = p
        else:
            start = r
        for n in range(start, MAX_L + 1, p):
            if n >= 1:
                full_covered[n] = True
    
    # Find actual max L
    final_L = 0
    for n in range(1, MAX_L + 1):
        if full_covered[n]:
            final_L = n
        else:
            break
    
    if final_L > best_L:
        best_L = final_L
    
    # Try to improve: if we have budget left, try adding more classes
    # Also try local search: swap classes
    
    # Check if we can do better with additional greedy passes on larger window
    if time.time() - start_time < DEADLINE - 10 and final_L < 999:
        # Try a second pass with larger window
        WINDOW2 = min(5000, final_L + 2000)
        covered2 = [False] * (WINDOW2 + 1)
        for p, r in selected:
            if r == 0:
                start = p
            else:
                start = r
            for n in range(start, WINDOW2 + 1, p):
                if n >= 1:
                    covered2[n] = True
        
        # Try adding more classes
        for _ in range(200):
            if time.time() - start_time > DEADLINE:
                break
            
            best_eff = -1
            best_p = None
            best_r = None
            
            for p in primes:
                cost = Fraction(1, p)
                if budget_used + cost > Fraction(1, 1):
                    continue
                
                for r in range(p):
                    if any(sp == p and sr == r for sp, sr in selected):
                        continue
                    
                    new_cov = compute_new_coverage(p, r, covered2, WINDOW2)
                    if new_cov == 0:
                        continue
                    
                    eff = new_cov * p
                    
                    if eff > best_eff:
                        best_eff = eff
                        best_p = p
                        best_r = r
            
            if best_p is None:
                break
            
            selected.append((best_p, best_r))
            budget_used += Fraction(1, best_p)
            apply_class(best_p, best_r, covered2, WINDOW2)
        
        # Re-verify
        full_covered = [False] * (MAX_L + 1)
        for p, r in selected:
            if r == 0:
                start = p
            else:
                start = r
            for n in range(start, MAX_L + 1, p):
                if n >= 1:
                    full_covered[n] = True
        
        final_L2 = 0
        for n in range(1, MAX_L + 1):
            if full_covered[n]:
                final_L2 = n
            else:
                break
        
        if final_L2 > best_L:
            best_L = final_L2
    
    return [[p, r] for p, r in selected]

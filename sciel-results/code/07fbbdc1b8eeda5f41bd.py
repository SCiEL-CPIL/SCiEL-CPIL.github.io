"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from fractions import Fraction
import numpy as np

# Precompute primes up to 1000
def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

_PRIMES = _primes_upto(1000)
_L_LIMIT = 20000  # Search up to this for L

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # State
    classes = []
    used_primes = set()
    budget_remaining = Fraction(1, 1)
    
    # Helper to find first gap and L
    def get_L(covered):
        # covered is a numpy bool array of size _L_LIMIT + 1
        # Find first index i >= 1 where covered[i] is False
        # If all 1.._L_LIMIT are True, return _L_LIMIT
        gap_indices = np.nonzero(~covered[1:])[0]
        if len(gap_indices) == 0:
            return _L_LIMIT
        else:
            return gap_indices[0]

    # 1. Greedy Initialization
    current_covered = np.zeros(_L_LIMIT + 1, dtype=bool)
    current_L = 0
    
    while time.time() < deadline - 10:
        if current_L >= _L_LIMIT:
            break
            
        # Find first uncovered n
        n = current_L + 1
        # Verify n is actually uncovered (it should be, by definition of L)
        if current_covered[n]:
            # Update L
            current_L = get_L(current_covered)
            continue
            
        # Find best prime to cover n
        best_p = None
        best_r = None
        best_score = -1
        
        for p in _PRIMES:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if budget_remaining < cost:
                continue
                
            r = n % p
            
            # Estimate gain
            start = r if r != 0 else p
            if start > _L_LIMIT:
                continue
                
            mask = current_covered[start : _L_LIMIT + 1 : p]
            gain = np.sum(~mask)
            
            # Score: gain * p
            score = gain * p
            
            if score > best_score:
                best_score = score
                best_p = p
                best_r = r
                
        if best_p is None:
            break
            
        # Apply best
        used_primes.add(best_p)
        budget_remaining -= Fraction(1, best_p)
        classes.append([best_p, best_r])
        
        # Update coverage
        start = best_r if best_r != 0 else best_p
        if start <= _L_LIMIT:
            current_covered[start : _L_LIMIT + 1 : best_p] = True
            
        # Update current_L
        current_L = get_L(current_covered)
        
        if current_L < n:
            break

    # 2. Local Search: Targeted Repair
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Find current L and first gap U
        current_L = get_L(current_covered)
        if current_L >= _L_LIMIT:
            break
            
        U = current_L + 1
        
        best_new_classes = None
        best_new_budget = budget_remaining
        best_new_L = current_L
        best_new_covered = None
        
        # Option A: Add new prime
        for p in _PRIMES:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if budget_remaining < cost:
                continue
                
            r = U % p
            new_classes = classes + [[p, r]]
            
            # Compute new coverage
            new_covered = current_covered.copy()
            start = r if r != 0 else p
            if start <= _L_LIMIT:
                new_covered[start : _L_LIMIT + 1 : p] = True
                
            new_L = get_L(new_covered)
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_new_classes = new_classes
                best_new_budget = budget_remaining - cost
                best_new_covered = new_covered
                improved = True
                
        # Option B: Change existing prime
        for idx, (p, r) in enumerate(classes):
            r_new = U % p
            if r_new == r:
                continue
                
            new_classes = [c[:] for c in classes]
            new_classes[idx] = [p, r_new]
            
            # Compute new coverage
            new_covered = np.zeros(_L_LIMIT + 1, dtype=bool)
            for pp, rr in new_classes:
                start = rr if rr != 0 else pp
                if start <= _L_LIMIT:
                    new_covered[start : _L_LIMIT + 1 : pp] = True
                    
            # Check prefix 1..U-1
            if not np.all(new_covered[1:U]):
                continue
                
            new_L = get_L(new_covered)
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_new_classes = new_classes
                best_new_budget = budget_remaining
                best_new_covered = new_covered
                improved = True
        
        if improved:
            classes = best_new_classes
            budget_remaining = best_new_budget
            current_covered = best_new_covered

    # Final check
    final_budget = sum(Fraction(1, p) for p, r in classes)
    if final_budget > Fraction(1, 1):
        return []
        
    return classes
# EVOLVE-BLOCK-END

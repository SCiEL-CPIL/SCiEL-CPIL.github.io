import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s budget, leave 10s margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute fractions for speed
    frac_values = [Fraction(1, d) for d in denominators]
    
    best_deficit = None
    best_indices = None
    best_sum = None
    
    # We search for subsets of size k
    # Based on insights, optimal is likely small (3-6).
    # We will search k=1 to 7.
    max_k = 7
    
    for k in range(1, max_k + 1):
        # If we have a very small deficit, we might prune, but let's just run.
        # Pruning: if current best deficit is D, we only need sum > 1 - D.
        # For k=1, max sum is 1/2. Deficit 1/2.
        # For k=2, max sum 1/2+1/3 = 5/6. Deficit 1/6.
        # For k=3, max sum 1/2+1/3+1/4 = 13/12 > 1. So we can't just take largest.
        
        for combo in combinations(range(n), k):
            # Calculate sum
            s = sum(frac_values[i] for i in combo)
            
            if s < 1:
                deficit = 1 - s
                if best_deficit is None or deficit < best_deficit:
                    best_deficit = deficit
                    best_indices = list(combo)
                    best_sum = s
            
            # Time check
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
            
    # If no solution found (should not happen, e.g. k=1, d=2 -> 1/2 < 1), return fallback
    if best_indices is None:
        return [0] # Index 0 is denominator 2, sum 1/2 < 1
        
    return best_indices

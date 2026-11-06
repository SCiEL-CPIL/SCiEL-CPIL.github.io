"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute suffix sums of reciprocals (as floats for speed)
    suffix_sums = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + 1.0 / denominators[i]
    
    # Initialize with Greedy Solution
    best_subset = []
    best_sum = Fraction(0, 1)
    current_sum = Fraction(0, 1)
    current_float = 0.0
    for i, d in enumerate(denominators):
        if current_sum + Fraction(1, d) < 1:
            best_subset.append(i)
            current_sum += Fraction(1, d)
            current_float += 1.0 / d
    best_sum = current_sum
    # best_subset is now the greedy set
    
    # DFS to improve
    def dfs(index, current_sum_frac, current_sum_float, subset):
        nonlocal best_sum, best_subset
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning: If max possible sum <= best_sum, prune
        # We are maximizing sum.
        if current_sum_float + suffix_sums[index] <= float(best_sum) - 1e-12:
            return
            
        # Base case: No more denominators
        if index == n:
            if current_sum_frac < 1 and current_sum_frac > best_sum:
                best_sum = current_sum_frac
                best_subset = list(subset)
            return
            
        # Option 1: Exclude denominators[index]
        # We try exclude first? Or include first?
        # Including large fractions first might hit the bound faster.
        # But we already have a good best_sum.
        # Let's try Include first to find better sums quickly?
        # Actually, if we include, sum increases. If we exclude, sum stays.
        # If we want to maximize, including is "greedy".
        # But we need to explore both.
        
        d = denominators[index]
        
        # Option 2: Include denominators[index]
        new_sum_float = current_sum_float + 1.0 / d
        if new_sum_float < 1.0:
            new_sum_frac = current_sum_frac + Fraction(1, d)
            if new_sum_frac < 1:
                subset.append(index)
                dfs(index + 1, new_sum_frac, new_sum_float, subset)
                subset.pop()
        
        # Option 1: Exclude denominators[index]
        dfs(index + 1, current_sum_frac, current_sum_float, subset)

    # Start DFS from index 0
    # Note: The subset passed to DFS should be empty initially, 
    # but we are comparing against the global best which is the greedy one.
    # The DFS will find subsets that are potentially better.
    # We don't need to start with the greedy subset in the DFS stack, 
    # because DFS explores all subsets. The global best just holds the record.
    dfs(0, Fraction(0, 1), 0.0, [])
    
    if not best_subset:
        # Fallback
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected
        
    return best_subset
# EVOLVE-BLOCK-END

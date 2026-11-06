import time
from fractions import Fraction

def construct():
    # Denominators in descending order: 48, 47, ..., 2
    # Reciprocals in ascending order: 1/48, 1/47, ..., 1/2
    denoms = list(range(48, 1, -1))
    recips = [Fraction(1, d) for d in denoms]
    n = len(recips)
    
    # Precompute suffix sums for pruning
    # suffix_sum[i] is the sum of recips[i] to recips[n-1]
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + recips[i]
        
    best_sum = Fraction(0)
    best_indices = []
    
    # Deadline guard: 50 seconds
    start_time = time.time()
    deadline = start_time + 50
    
    def dfs(idx, current_sum, selected):
        nonlocal best_sum, best_indices
        
        # Time check
        if time.time() > deadline:
            return
            
        # Prune if even adding all remaining can't beat best_sum
        if current_sum + suffix_sum[idx] <= best_sum:
            return
            
        # Update best if current is better
        if current_sum > best_sum:
            best_sum = current_sum
            best_indices = selected.copy()
            
        # Try adding each subsequent reciprocal
        for j in range(idx, n):
            new_sum = current_sum + recips[j]
            if new_sum >= 1:
                # Since recips are sorted in increasing order, 
                # all subsequent recips are larger, so sum will only increase.
                break
            selected.append(denoms[j])
            dfs(j + 1, new_sum, selected)
            selected.pop()
            
    dfs(0, Fraction(0), [])
    
    # Convert selected denominators to zero-based indices of [2, 3, ..., 48]
    # Index i in [2, 3, ..., 48] corresponds to denominator i + 2
    # So denominator d corresponds to index d - 2
    result_indices = [d - 2 for d in best_indices]
    
    # Fallback if no solution found (should not happen)
    if not result_indices:
        return [46] # Index for denominator 48
        
    return result_indices

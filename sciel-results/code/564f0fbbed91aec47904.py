"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s
    
    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)  # 47
    recip = [Fraction(1, d) for d in denominators]
    
    # Precompute suffix sums: suffix[i] = sum of recip[i:]
    suffix = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + recip[i]
    
    # Greedy initial solution: take items in descending order of value (i.e., ascending denominator)
    best_sum = Fraction(0)
    best_indices = []
    cur_sum = Fraction(0)
    for i in range(n):
        if cur_sum + recip[i] < 1:
            cur_sum += recip[i]
            best_indices.append(i)
    best_sum = cur_sum
    
    # Try to improve with local search: try swapping one selected item for one unselected
    improved = True
    while improved:
        improved = False
        sel_set = set(best_indices)
        for i in best_indices:
            for j in range(n):
                if j not in sel_set:
                    new_sum = best_sum - recip[i] + recip[j]
                    if 0 < new_sum < 1 and new_sum > best_sum:
                        best_sum = new_sum
                        best_indices = [j if k == i else k for k in best_indices]
                        sel_set = set(best_indices)
                        improved = True
                        break
            if improved:
                break
    
    best_sum_val = best_sum
    
    # Branch and bound with time check
    def bb(idx, cur_sum, cur_indices):
        nonlocal best_sum, best_sum_val, best_indices
        
        if time.time() > deadline:
            return
        
        # Prune: can't beat best
        if cur_sum + suffix[idx] <= best_sum_val:
            return
        
        if idx == n:
            if cur_sum < 1 and cur_sum > best_sum_val:
                best_sum_val = cur_sum
                best_sum = cur_sum
                best_indices = list(cur_indices)
            return
        
        # Option 1: include item idx (try first to find good solutions early)
        new_sum = cur_sum + recip[idx]
        if new_sum < 1:
            cur_indices.append(idx)
            bb(idx + 1, new_sum, cur_indices)
            cur_indices.pop()
        
        # Option 2: exclude item idx
        bb(idx + 1, cur_sum, cur_indices)
    
    # Start branch and bound
    bb(0, Fraction(0), [])
    
    return best_indices

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
from math import gcd

DENOMINATORS = list(range(2, 49))
N = len(DENOMINATORS)
FRAC_DENOMS = [Fraction(1, d) for d in DENOMINATORS]
ONE = Fraction(1, 1)
ZERO = Fraction(0, 1)

# Precompute float values for speed in B&B
FLOAT_DENOMS = [1.0 / d for d in DENOMINATORS]

def _branch_and_bound():
    """
    Branch and Bound to find the subset with maximum sum < 1.
    Returns (best_sum_float, best_indices).
    """
    # Items are already sorted by value descending (index 0 is 1/2)
    
    # Precompute suffix sums of float values
    # suffix_sum[i] = sum of FLOAT_DENOMS[i:]
    suffix_sum = [0.0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + FLOAT_DENOMS[i]
    
    best_sum = 0.0
    best_indices = []
    current_indices = []
    
    # Optimization: Track nodes visited to avoid infinite loops or excessive time?
    # For N=47, B&B is usually fast.
    
    def dfs(idx, current_sum):
        nonlocal best_sum, best_indices
        
        # Pruning 1: If current sum is already >= 1, we cannot add more valid items
        # (and this state itself is invalid if we consider the set so far).
        # However, we only recurse if new_sum < 1.0. So current_sum < 1.0 always.
        # But if current_sum >= 1.0, it's an error state, prune.
        if current_sum >= 1.0:
            return
            
        # Pruning 2: Upper bound
        # Max possible sum = current_sum + sum of all remaining items
        if current_sum + suffix_sum[idx] <= best_sum:
            return
            
        # If we are at the end, update best
        if idx == N:
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = list(current_indices)
            return
        
        val = FLOAT_DENOMS[idx]
        
        # Option 1: Include item idx
        new_sum = current_sum + val
        if new_sum < 1.0:
            current_indices.append(idx)
            dfs(idx + 1, new_sum)
            current_indices.pop()
            
        # Option 2: Exclude item idx
        dfs(idx + 1, current_sum)

    dfs(0, 0.0)
    
    return best_sum, best_indices

def _local_search_refine(start_indices, deadline):
    """
    Refine a set of indices using 1-swap and 2-swap local search with exact Fractions.
    """
    best_indices = list(start_indices)
    if not best_indices:
        best_indices = [0]
        
    # Ensure validity
    exact_sum = sum(FRAC_DENOMS[i] for i in best_indices)
    if exact_sum >= ONE:
        # Remove largest elements until valid
        sorted_idx = sorted(best_indices, key=lambda i: -FLOAT_DENOMS[i])
        for i in sorted_idx:
            exact_sum -= FRAC_DENOMS[i]
            best_indices.remove(i)
            if exact_sum < ONE:
                break
        if exact_sum >= ONE:
            best_indices = [0]
            exact_sum = FRAC_DENOMS[0]

    best_deficit = ONE - exact_sum
    in_set = set(best_indices)
    
    # 1-swap
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in best_indices[:]:
            if time.time() > deadline:
                break
            sum_without = exact_sum - FRAC_DENOMS[i]
            for j in range(N):
                if j in in_set:
                    continue
                new_sum = sum_without + FRAC_DENOMS[j]
                if new_sum < ONE:
                    new_def = ONE - new_sum
                    if new_def < best_deficit:
                        best_indices.remove(i)
                        best_indices.append(j)
                        in_set.remove(i)
                        in_set.add(j)
                        exact_sum = new_sum
                        best_deficit = new_def
                        improved = True
                        break
            if improved:
                break
                
    # 2-swap
    if time.time() < deadline - 2:
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            k = len(best_indices)
            if k < 2: break
            for a in range(k):
                if time.time() > deadline:
                    break
                for b in range(a+1, k):
                    if time.time() > deadline:
                        break
                    i, j = best_indices[a], best_indices[b]
                    sum_without_two = exact_sum - FRAC_DENOMS[i] - FRAC_DENOMS[j]
                    for x in range(N):
                        if x in in_set:
                            continue
                        f_x = FRAC_DENOMS[x]
                        sum_after_x = sum_without_two + f_x
                        if sum_after_x >= ONE:
                            continue
                        for y in range(x+1, N):
                            if y in in_set:
                                continue
                            f_y = FRAC_DENOMS[y]
                            new_sum = sum_after_x + f_y
                            if new_sum < ONE:
                                new_def = ONE - new_sum
                                if new_def < best_deficit:
                                    best_indices.remove(i)
                                    best_indices.remove(j)
                                    best_indices.append(x)
                                    best_indices.append(y)
                                    in_set.remove(i)
                                    in_set.remove(j)
                                    in_set.add(x)
                                    in_set.add(y)
                                    exact_sum = new_sum
                                    best_deficit = new_def
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
                    
    return best_indices

def construct():
    deadline = time.time() + 85 # Safety margin
    
    # Run Branch and Bound
    best_float_sum, best_indices = _branch_and_bound()
    
    # Refine with exact arithmetic
    final_indices = _local_search_refine(best_indices, deadline)
    
    return final_indices
# EVOLVE-BLOCK-END

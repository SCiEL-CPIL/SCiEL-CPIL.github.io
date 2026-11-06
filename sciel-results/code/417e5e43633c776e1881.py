"""Combined program: Branch-and-Bound DFS with local search refinement."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
from fractions import Fraction

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    # Denominators 2 to 48 inclusive
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Compute LCM of 2..48
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(2, 49))
    
    # Precompute integer weights
    # index i corresponds to denominator i+2
    # weight[i] = L // (i+2)
    weights = [L // d for d in denominators]
    
    # Precompute suffix sums of weights for the fractional knapsack bound
    # suffix_sum[i] = sum(weights[i:])
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
        
    # Best solution tracking
    best_sum_weight = 0
    best_indices = []
    
    def update_best(sum_w, indices):
        nonlocal best_sum_weight, best_indices
        if sum_w > best_sum_weight:
            best_sum_weight = sum_w
            best_indices = list(indices)
            
    # DFS with Branch and Bound
    # We want to maximize sum(weights[i]) subject to sum(weights[i]) < L
    # This is equivalent to minimizing L - sum(weights[i])
    
    # Heuristic: "Take First"
    # At each step, try taking the current item first, then skipping it.
    # This helps find large sums early, raising the best_sum_weight threshold
    # and improving pruning.
    
    def dfs(idx, current_sum, current_indices):
        nonlocal best_sum_weight, best_indices
        
        if time.time() - start_time > timeout:
            return
            
        # Pruning: If even taking all remaining items doesn't beat the best, prune
        # Upper bound = current_sum + suffix_sum[idx]
        # If current_sum + suffix_sum[idx] <= best_sum_weight, prune
        if current_sum + suffix_sum[idx] <= best_sum_weight:
            return
            
        # If we've considered all items
        if idx == n:
            if current_sum > best_sum_weight and current_sum < L:
                update_best(current_sum, current_indices)
            return
            
        # Option 1: Take item idx
        # Only valid if current_sum + weights[idx] < L
        if current_sum + weights[idx] < L:
            dfs(idx + 1, current_sum + weights[idx], current_indices + [idx])
            
            # If we timed out, return immediately
            if time.time() - start_time > timeout:
                return
                
        # Option 2: Skip item idx
        dfs(idx + 1, current_sum, current_indices)
        
    # Start DFS
    dfs(0, 0, [])
    
    if not best_indices:
        return [0] # Fallback: 1/2
        
    # Convert best_indices from DFS (which are 0-based indices into denominators [2..48])
    # These are already the correct indices for the problem
    best_set = set(best_indices)
    
    # Now do local search refinement using exact fractions
    # to ensure we're not missing any improvements
    inv = [Fraction(1, d) for d in denominators]
    
    total = sum(inv[i] for i in best_set)
    selected = set(best_set)
    
    # Local search: try single moves and swaps
    def try_update(sel, tot):
        nonlocal selected, total
        if tot < 1 and tot > sum(inv[i] for i in selected):
            selected = set(sel)
            total = tot
            return True
        return False
    
    # Single insertion
    for i in range(n):
        if i not in selected:
            new_total = total + inv[i]
            if new_total < 1:
                if try_update(selected | {i}, new_total):
                    pass
    
    # Single swap
    sel_list = list(selected)
    for i in sel_list:
        for j in range(n):
            if j not in selected:
                new_total = total - inv[i] + inv[j]
                if new_total < 1:
                    if try_update((selected - {i}) | {j}, new_total):
                        pass
    
    return list(selected)
# EVOLVE-BLOCK-END

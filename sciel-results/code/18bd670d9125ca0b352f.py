"""Rewritten program: exact subset-sum search over unit fractions 1/2..1/48
to find the subset with reciprocal sum strictly below 1 and minimal deficit."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    # Denominators 2..48, indices 0..46
    denoms = list(range(2, 49))
    n = len(denoms)
    recip = [Fraction(1, d) for d in denoms]

    # Sort by value descending (i.e., by denominator ascending) for better pruning
    # Keep track of original indices
    order = list(range(n))
    order.sort(key=lambda i: -recip[i].numerator / recip[i].denominator)  # descending value
    # Actually for unit fractions, 1/d is larger when d is smaller, so sort by d ascending
    order = list(range(n))
    # order[i] = original index of the i-th largest reciprocal (smallest denominator first)
    # Since recip[i] = 1/(i+2), larger reciprocal means smaller i
    # So order is just 0, 1, 2, ..., 46 (already in descending value order)
    
    # Precompute suffix sums for upper bound pruning
    # suffix_sum[i] = sum of recip[order[i]] + recip[order[i+1]] + ... + recip[order[n-1]]
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + recip[order[i]]
    
    # Total sum of all reciprocals
    total_all = suffix_sum[0]
    
    # We want to find subset S such that sum_{i in S} recip[i] < 1, minimizing 1 - sum
    # Equivalently, maximize sum_{i in S} recip[i] subject to sum < 1
    
    # Greedy upper bound: the maximum possible sum < 1
    # We'll use branch and bound
    
    best_sum = Fraction(0)  # We'll track the best (largest) sum < 1
    best_indices = []
    
    deadline = time.time() + 85  # 85 seconds, leave 5s margin
    
    # Use DFS with branch and bound
    # State: (pos, current_sum, selected_indices)
    # pos: next position in order to consider
    # We prune if:
    #   1. current_sum + suffix_sum[pos] <= current_best_sum (can't improve)
    #   2. current_sum >= 1 (invalid, and adding more only makes it worse)
    
    def dfs(pos, current_sum, selected):
        nonlocal best_sum, best_indices
        if time.time() > deadline:
            return
        
        # Pruning: even if we take all remaining, can't beat best
        if current_sum + suffix_sum[pos] <= best_sum:
            return
        
        # If current_sum >= 1, this branch is invalid
        if current_sum >= 1:
            return
        
        # If we've considered all elements
        if pos == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = list(selected)
            return
        
        # Option 1: take element at position pos
        new_sum = current_sum + recip[order[pos]]
        if new_sum < 1:
            selected.append(order[pos])
            dfs(pos + 1, new_sum, selected)
            selected.pop()
        
        # Option 2: skip element at position pos
        # Pruning: if current_sum + suffix_sum[pos+1] <= best_sum, skip
        if current_sum + suffix_sum[pos + 1] > best_sum:
            dfs(pos + 1, current_sum, selected)
    
    # Initialize with a greedy solution to get a good lower bound
    # Greedy: add elements in order of largest reciprocal first while sum < 1
    greedy_sum = Fraction(0)
    greedy_selected = []
    for i in range(n):
        if greedy_sum + recip[order[i]] < 1:
            greedy_sum += recip[order[i]]
            greedy_selected.append(order[i])
    
    if greedy_sum > best_sum:
        best_sum = greedy_sum
        best_indices = list(greedy_selected)
    
    # Now run the full search
    dfs(0, Fraction(0), [])
    
    # The best_indices contains original indices
    # But we need to return zero-based indices into denominators [2,3,...,48]
    # which is exactly what we have (index 0 -> denom 2, index 1 -> denom 3, etc.)
    
    if not best_indices:
        # Fallback: return a single element
        return [0]
    
    return best_indices
# EVOLVE-BLOCK-END

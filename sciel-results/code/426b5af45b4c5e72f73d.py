"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, n + 1):
        if n % i == 0:
            divisors.append(i)
    divisors.sort(reverse=True)
    divisors = [d for d in divisors if d <= target]
    
    # Try combinations for small k first
    for k in range(1, 10):
        if time.time() > deadline:
            break
        for combo in combinations(divisors, k):
            if sum(combo) == target:
                return list(combo)
    
    # Use DFS with pruning for larger k
    # Precompute suffix sums for pruning
    m = len(divisors)
    suffix_sum = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]
    
    best = None
    
    def dfs(idx, remaining, selected):
        nonlocal best
        if time.time() > deadline:
            return
        if remaining == 0:
            if best is None or len(selected) < len(best):
                best = list(selected)
            return
        if remaining < 0:
            return
        if idx >= m:
            return
        # Pruning: if even taking all remaining divisors can't reach target, return
        if suffix_sum[idx] < remaining:
            return
        # Pruning: if we already have a solution and can't improve, return
        if best is not None and len(selected) >= len(best):
            return
        
        # Option 1: take divisors[idx]
        if divisors[idx] <= remaining:
            selected.append(divisors[idx])
            dfs(idx + 1, remaining - divisors[idx], selected)
            selected.pop()
        
        # Option 2: skip divisors[idx]
        dfs(idx + 1, remaining, selected)
    
    dfs(0, target, [])
    
    if best is not None:
        return best
    
    # Greedy fallback
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
    return chosen

# EVOLVE-BLOCK-END

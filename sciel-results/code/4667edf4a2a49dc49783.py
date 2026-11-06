"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _get_divisors(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs, reverse=True)

def construct():
    deadline = time.time() + 50  # safety margin before 90s kill
    n = 5040
    target = 4979
    divisors = _get_divisors(n)
    # Filter divisors <= target
    divisors = [d for d in divisors if d <= target]
    m = len(divisors)
    
    # BFS by number of divisors used (level = count)
    # State: sum -> list of divisor indices used
    # Start with sum=0, empty set
    # We want to reach sum=target with minimum count
    
    # Use BFS: at each level, we add one more divisor
    # visited: sum -> (count, indices) to avoid reprocessing
    from collections import defaultdict, deque
    
    # visited[sum] = minimum count to achieve that sum, and the path
    best_count_for_sum = {}  # sum -> count
    best_path_for_sum = {}   # sum -> tuple of indices
    
    # BFS queue: (sum, count, indices_tuple)
    # But indices_tuple can be long. Instead, store parent pointers.
    
    # Let's use a different approach: iterative deepening DFS
    # Try k = 1, 2, 3, ...
    # For each k, check if any combination of k divisors sums to target
    
    # Precompute divisors as list
    divs = divisors  # sorted descending
    
    # For efficiency, use recursive search with pruning
    def dfs(start_idx, remaining, count_left, current_set):
        if count_left == 0:
            return current_set if remaining == 0 else None
        if remaining == 0:
            return current_set  # early termination, but we need exactly count_left==0 for min
        if remaining < 0:
            return None
        # Pruning: even if we take all remaining largest, can we reach?
        # Max possible sum with count_left more divisors from start_idx onwards
        # But this is complex. Let's just do it with time check.
        
        for i in range(start_idx, len(divs)):
            d = divs[i]
            if d > remaining:
                continue
            # Check time
            if time.time() > deadline:
                return None
            result = dfs(i + 1, remaining - d, count_left - 1, current_set + [d])
            if result is not None:
                return result
        return None
    
    for k in range(1, m + 1):
        if time.time() > deadline:
            break
        result = dfs(0, target, k, [])
        if result is not None:
            return result
    
    # Fallback: greedy (shouldn't reach here)
    remainder = target
    chosen = []
    for d in divs:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
    return chosen
# EVOLVE-BLOCK-END

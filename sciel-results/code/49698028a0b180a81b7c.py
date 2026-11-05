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
    # Only keep divisors that could be part of a sum to target
    divisors = [d for d in divisors if d <= target]
    
    # Use iterative deepening DFS with pruning
    # For each k from 1 upward, search for a combination of k distinct divisors summing to target
    # Sort divisors in descending order to try large values first
    
    # Precompute suffix sums for pruning: suffix_sum[i] = sum of divisors[i:]
    # This helps prune when remaining sum is too large or too small
    suffix_sums = [0] * (len(divisors) + 1)
    for i in range(len(divisors) - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + divisors[i]
    
    def dfs(start, remaining, k, path):
        """Try to find k divisors from divisors[start:] that sum to 'remaining'"""
        if time.time() > deadline:
            return None
        if k == 0:
            if remaining == 0:
                return list(path)
            return None
        if remaining < 0:
            return None
        if start >= len(divisors):
            return None
        # Pruning: even taking all remaining divisors from start, can we reach 'remaining'?
        # The max we can get from k divisors starting at index 'start' is the sum of the k largest
        # available from start onwards
        if start + k > len(divisors):
            return None
        # Max possible sum: sum of k largest divisors from start
        max_possible = suffix_sums[start] - suffix_sums[start + k] if start + k <= len(divisors) else suffix_sums[start]
        if max_possible < remaining:
            return None
        # Min possible sum: sum of k smallest divisors from start
        min_possible = suffix_sums[len(divisors) - k] - suffix_sums[len(divisors)] if k <= len(divisors) else 0
        # Actually, min possible from divisors[start:] taking k items is sum of k smallest in divisors[start:]
        # Since divisors is sorted descending, the k smallest from start are divisors[-k:] but only if start <= len-k
        # Let's just use: if remaining < k * divisors[-1] (smallest divisor overall) but that's not tight enough
        # Better: the k smallest divisors available from index start are divisors[len(divisors)-k:] if start <= len(divisors)-k
        # For simplicity, let's use a looser bound
        if remaining < k:  # each divisor is at least 1
            return None
        
        for i in range(start, len(divisors) - k + 1):
            d = divisors[i]
            if d > remaining:
                continue
            path.append(d)
            result = dfs(i + 1, remaining - d, k - 1, path)
            if result is not None:
                return result
            path.pop()
            if time.time() > deadline:
                return None
        return None
    
    best_result = None
    for k in range(1, 20):
        if time.time() > deadline:
            break
        # Check feasibility: sum of k largest divisors must be >= target
        if sum(divisors[:k]) < target:
            continue
        # Check feasibility: sum of k smallest divisors must be <= target
        if sum(divisors[-k:]) > target:
            break
        result = dfs(0, target, k, [])
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback
    return divisors[:1]

# EVOLVE-BLOCK-END

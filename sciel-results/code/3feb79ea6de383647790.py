"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

# Precompute optimal solutions at module load time
def _precompute_solutions():
    """Precompute minimum-term antichain representations for targets 1-128."""
    # Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)
    
    smooth.sort()
    n_smooth = len(smooth)
    
    # Precompute conflict bitmask
    conflict_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and (smooth[i] % smooth[j] == 0 or smooth[j] % smooth[i] == 0):
                conflict_mask[i] |= (1 << j)
    
    # For each target, find the minimum number of terms needed
    # We'll use BFS/DFS with increasing k until we find a solution
    solutions = {}
    
    for target in range(1, 129):
        # Try k = 1, 2, 3, ... until we find a valid antichain
        found = False
        for k in range(1, 11):  # Max 10 terms should be enough
            # Use DFS to find antichain of size k summing to target
            result = _find_antichain_size_k(target, k, smooth, conflict_mask, n_smooth)
            if result is not None:
                solutions[target] = [smooth[i] for i in result]
                found = True
                break
        
        if not found:
            # Fallback: use any valid representation
            solutions[target] = _find_any(target, smooth, n_smooth)
    
    return solutions

def _find_antichain_size_k(target, k, smooth, conflict_mask, n_smooth):
    """Find antichain of exactly size k summing to target using DFS with pruning."""
    if k == 0:
        return () if target == 0 else None
    if target < 0 or n_smooth < k:
        return None
    
    # Pruning: minimum possible sum with k elements is sum of k smallest
    # Maximum possible sum is sum of k largest
    if k > 0:
        min_possible = sum(smooth[:k])
        max_possible = sum(smooth[-k:])
        if target < min_possible or target > max_possible:
            return None
    
    def dfs(remaining, start_idx, chosen_mask, chosen_indices, depth):
        if depth == k:
            return chosen_indices if remaining == 0 else None
        if start_idx >= n_smooth:
            return None
        
        # Pruning: not enough elements left
        if n_smooth - start_idx < (k - depth):
            return None
        
        # Pruning: remaining must be at least sum of (k-depth) smallest from start_idx
        if remaining < sum(smooth[start_idx:start_idx + (k - depth)]):
            return None
        
        # Pruning: remaining must be at most sum of (k-depth) largest
        if remaining > sum(smooth[n_smooth - (k - depth):n_smooth]):
            return None
        
        for i in range(start_idx, n_smooth):
            val = smooth[i]
            if val > remaining:
                break
            
            # Check conflict
            if chosen_mask & conflict_mask[i]:
                continue
            
            new_mask = chosen_mask | (1 << i)
            result = dfs(remaining - val, i + 1, new_mask, chosen_indices + (i,), depth + 1)
            if result is not None:
                return result
        
        return None
    
    return dfs(target, 0, 0, (), 0)

def _find_any(target, smooth, n_smooth):
    """Find any valid antichain representation using DFS."""
    def dfs(remaining, start_idx, chosen_mask, chosen_indices):
        if remaining == 0:
            return chosen_indices
        if start_idx >= n_smooth or remaining < 0:
            return None
        
        val = smooth[start_idx]
        if val <= remaining:
            # Check conflict
            has_conflict = False
            for i in chosen_indices:
                if smooth[i] % val == 0 or val % smooth[i] == 0:
                    has_conflict = True
                    break
            if not has_conflict:
                result = dfs(remaining - val, start_idx + 1, chosen_mask | (1 << start_idx), chosen_indices + (start_idx,))
                if result is not None:
                    return result
        
        return dfs(remaining, start_idx + 1, chosen_mask, chosen_indices)
    
    result = dfs(target, 0, 0, ())
    if result is not None:
        return [smooth[i] for i in result]
    return []

# Precompute all solutions
_PRECOMPUTED_SOLUTIONS = _precompute_solutions()

def construct():
    """Return 128 rows, each containing a valid antichain representation of its target."""
    return [_PRECOMPUTED_SOLUTIONS[n] for n in range(1, 129)]

# EVOLVE-BLOCK-END

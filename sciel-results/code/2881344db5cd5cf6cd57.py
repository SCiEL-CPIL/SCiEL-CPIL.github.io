import time
from fractions import Fraction

def construct():
    """
    Finds the subset of denominators [2, 3, ..., 48] that minimizes the deficit 
    1 - sum(1/d) subject to sum(1/d) < 1.
    
    Uses a sophisticated branch-and-bound with float arithmetic for speed,
    exact rational verification, and multiple search strategies.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocal values as floats and exact fractions
    reciprocals_float = [1.0 / d for d in denominators]
    reciprocals_frac = [Fraction(1, d) for d in denominators]
    
    # Precompute suffix sums for pruning (float)
    suffix_sums = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + reciprocals_float[i]
    
    # Global best state
    best_sum_float = -1.0
    best_indices = []
    best_sum_frac = None
    
    # Check if a candidate is strictly below 1 using exact arithmetic
    def verify_strictly_below(indices):
        if not indices:
            return Fraction(0, 1)
        s = Fraction(0, 1)
        for idx in indices:
            s += reciprocals_frac[idx]
        return s if s < 1 else None
    
    # Try to improve with exact verification
    def try_improve(indices):
        nonlocal best_sum_float, best_indices, best_sum_frac
        if not indices:
            return
        s = verify_strictly_below(indices)
        if s is not None:
            s_float = float(s)
            if s_float > best_sum_float:
                best_sum_float = s_float
                best_indices = indices[:]
                best_sum_frac = s
    
    # Strategy 1: Greedy initialization - start with small denominators, add while < 1
    greedy_indices = []
    greedy_sum = 0.0
    for i in range(n):
        new_sum = greedy_sum + reciprocals_float[i]
        if new_sum < 1.0:
            greedy_indices.append(i)
            greedy_sum = new_sum
    try_improve(greedy_indices)
    
    # Strategy 2: Branch and bound
    def dfs(i, cur_sum_float, indices):
        nonlocal best_sum_float, best_indices, best_sum_frac
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning 1: If current sum is already >= 1, we can't use this path
        if cur_sum_float >= 1.0:
            return
            
        # Pruning 2: Even if we add all remaining items, can we beat the best?
        if cur_sum_float + suffix_sums[i] <= best_sum_float + 1e-15:
            return
            
        # If we've considered all items, check if this is the best
        if i == n:
            if cur_sum_float > best_sum_float:
                try_improve(indices)
            return
        
        # Option 1: Exclude item i
        dfs(i + 1, cur_sum_float, indices)
        
        # Option 2: Include item i
        new_sum = cur_sum_float + reciprocals_float[i]
        if new_sum < 1.0:
             dfs(i + 1, new_sum, indices + [i])
    
    # Run B&B
    dfs(0, 0.0, [])
    
    # Strategy 3: Local search - try to swap or add/remove items to improve
    if best_indices and time.time() < deadline - 5.0:
        current = best_indices[:]
        current_set = set(current)
        current_sum_frac = best_sum_frac
        
        # Try adding any single item not in current
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if i in current_set:
                    continue
                new_set = current_set | {i}
                new_sum = current_sum_frac + reciprocals_frac[i]
                if new_sum < 1:
                    # Check if this improves
                    if float(new_sum) > float(current_sum_frac):
                        current = list(new_set)
                        current_set = new_set
                        current_sum_frac = new_sum
                        improved = True
                        if time.time() > deadline - 2.0:
                            break
            
            # Try removing one item and adding a different one
            if not improved and time.time() < deadline - 2.0:
                for remove_idx in current:
                    for add_idx in range(n):
                        if add_idx in current_set:
                            continue
                        new_set = (current_set - {remove_idx}) | {add_idx}
                        new_sum = current_sum_frac - reciprocals_frac[remove_idx] + reciprocals_frac[add_idx]
                        if new_sum < 1 and float(new_sum) > float(current_sum_frac):
                            current = list(new_set)
                            current_set = new_set
                            current_sum_frac = new_sum
                            improved = True
                            if time.time() > deadline - 2.0:
                                break
                    if time.time() > deadline - 2.0:
                        break
            
            # Try removing one item
            if not improved and time.time() < deadline - 2.0:
                for remove_idx in current:
                    new_set = current_set - {remove_idx}
                    new_sum = current_sum_frac - reciprocals_frac[remove_idx]
                    # Removing always makes sum smaller, so this won't improve unless
                    # we can add something else. Skip pure removals.
            
            # Try adding two items at once
            if not improved and time.time() < deadline - 2.0:
                for i in range(n):
                    if i in current_set:
                        continue
                    for j in range(i+1, n):
                        if j in current_set:
                            continue
                        new_sum = current_sum_frac + reciprocals_frac[i] + reciprocals_frac[j]
                        if new_sum < 1 and float(new_sum) > float(current_sum_frac):
                            current = current + [i, j]
                            current_set = current_set | {i, j}
                            current_sum_frac = new_sum
                            improved = True
                            if time.time() > deadline - 2.0:
                                break
                    if time.time() > deadline - 2.0:
                        break
        
        try_improve(current)
    
    # Fallback if no solution found
    if not best_indices:
        return [0]
        
    return best_indices

if __name__ == "__main__":
    res = construct()
    pass

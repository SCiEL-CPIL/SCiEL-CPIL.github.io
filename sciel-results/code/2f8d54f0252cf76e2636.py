"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

def _can_form_target(values, target):
    """Check if any non-empty subset of values sums to target using meet-in-the-middle."""
    n = len(values)
    if n == 0:
        return False
    if target < 0:
        return False
    
    # Quick checks
    if target in values:
        return True
    if sum(values) < target:
        return False
    
    # Meet in the middle
    mid = n // 2
    left = values[:mid]
    right = values[mid:]
    
    # Generate all subset sums for left half
    left_sums = set()
    for i in range(len(left) + 1):
        for combo in combinations(range(len(left)), i):
            s = sum(left[j] for j in combo)
            left_sums.add(s)
    
    # Check right half subsets
    for i in range(len(right) + 1):
        for combo in combinations(range(len(right)), i):
            s = sum(right[j] for j in combo)
            if (target - s) in left_sums:
                # Ensure not both empty (i.e., at least one element used)
                if i > 0 or (target - s) != 0 or i == 0 and (target - s) != 0:
                    # More precisely: we need at least one element total
                    # If i == 0, then left sum must be target and target != 0 (already checked)
                    # If i > 0, we have at least one right element
                    return True
    return False

def _can_form_target_fast(values, target):
    """Faster subset sum check using bitset-like approach for small value ranges."""
    n = len(values)
    if n == 0 or target < 0:
        return False
    if target in values:
        return True
    if sum(values) < target:
        return False
    
    # For small n, use meet-in-the-middle
    if n <= 20:
        return _can_form_target(values, target)
    
    # For larger n, use DFS with pruning
    values_sorted = sorted(values, reverse=True)
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + values_sorted[i]
    
    # DFS with memoization
    memo = {}
    
    def dfs(idx, remaining):
        if remaining == 0:
            return True
        if idx == n or remaining < 0:
            return False
        if suffix_sums[idx] < remaining:
            return False
        key = (idx, remaining)
        if key in memo:
            return memo[key]
        
        # Try taking values_sorted[idx]
        if dfs(idx + 1, remaining - values_sorted[idx]):
            memo[key] = True
            return True
        # Try skipping
        if dfs(idx + 1, remaining):
            memo[key] = True
            return True
        
        memo[key] = False
        return False
    
    return dfs(0, target)

def _is_reciprocal_subsum_free(selected):
    """Verify that the set is reciprocal-subsum-free."""
    sel_set = set(selected)
    for x in selected:
        others = [y for y in selected if y != x]
        if not others:
            continue
        L = _lcm_list(others)
        L2 = _lcm(L, x)
        target = L2 // x
        values = [L2 // y for y in others]
        if _can_form_target_fast(values, target):
            return False
    return True

def _try_add_element(selected, x, deadline):
    """Try to add element x to selected, checking if it's safe."""
    if x in selected:
        return selected, True
    
    others = selected
    if not others:
        return selected + [x], True
    
    L = _lcm_list(others)
    L2 = _lcm(L, x)
    target = L2 // x
    values = [L2 // y for y in others]
    
    if _can_form_target_fast(values, target):
        return selected, False
    
    return selected + [x], True

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    # Greedy: try elements from 24 down to 1
    # When adding element x, check if 1/x can be formed by subset of reciprocals of already-selected elements
    # Already-selected elements are all > x, so their reciprocals are all < 1/x
    
    selected = []  # list of selected elements (will be in descending order)
    
    for x in range(24, 0, -1):
        if time.time() > deadline:
            break
        
        if not selected:
            selected.append(x)
            continue
        
        # Compute LCM of all elements in selected
        L = _lcm_list(selected)
        L2 = _lcm(L, x)
        target = L2 // x
        values = [L2 // y for y in selected]
        
        if not _can_form_target_fast(values, target):
            selected.append(x)
    
    # Local search: try to add back excluded elements
    if time.time() < deadline - 5:
        excluded = [x for x in range(1, 25) if x not in selected]
        # Try adding each excluded element
        for x in excluded:
            if time.time() > deadline - 5:
                break
            new_selected, success = _try_add_element(selected, x, deadline)
            if success:
                selected = new_selected
    
    # Try 1-out-k-in local search
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            sel_set = set(selected)
            excluded = [x for x in range(1, 25) if x not in sel_set]
            
            for x in excluded:
                if time.time() > deadline - 10:
                    break
                # Try removing one element and adding x
                for y in selected:
                    if time.time() > deadline - 10:
                        break
                    candidate = [z for z in selected if z != y] + [x]
                    if len(candidate) > len(selected) or (len(candidate) == len(selected) and _is_reciprocal_subsum_free(candidate)):
                        # Check if this is actually an improvement or same size but we prefer it
                        if len(candidate) > len(selected):
                            selected = candidate
                            improved = True
                            break
                        elif _is_reciprocal_subsum_free(candidate):
                            # Same size, keep the first valid one found
                            pass
                if improved:
                    break
    
    # Return as a list (order doesn't matter for the contract)
    if not selected:
        return [24]
    return list(selected)

# EVOLVE-BLOCK-END

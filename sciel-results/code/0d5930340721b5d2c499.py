"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from math import gcd

# Increase recursion limit for B&B
sys.setrecursionlimit(10000)

def _lcm(a, b):
    return a * b // gcd(a, b)

def _subset_sum_check(target, values):
    """
    Check if target can be formed by a subset of values.
    values should be sorted descending for pruning.
    target and values are integers (scaled by LCM).
    """
    if target == 0:
        return True
    if not values:
        return False
    if sum(values) < target:
        return False
    
    # Sort descending to prune faster
    vals = sorted(values, reverse=True)
    
    # Recursive helper
    def _dfs(idx, remaining):
        if remaining == 0:
            return True
        if idx >= len(vals):
            return False
        if remaining < 0:
            return False
        if sum(vals[idx:]) < remaining:
            return False
            
        # Option 1: Include vals[idx]
        if _dfs(idx + 1, remaining - vals[idx]):
            return True
        # Option 2: Exclude vals[idx]
        if _dfs(idx + 1, remaining):
            return True
        return False
        
    return _dfs(0, target)

# Precompute LCM of 1..24
L = 1
for i in range(1, 25):
    L = _lcm(L, i)

# Precompute scaled reciprocals: val[i] = L // i
vals = [0] * 25
for i in range(1, 25):
    vals[i] = L // i

def _is_incrementally_valid(s, x):
    """
    Check if adding x to valid set s keeps it valid.
    s is a set of ints. x is an int not in s.
    """
    # 1. Check if 1/x is sum of subset of s
    target_x = vals[x]
    others_x = [vals[y] for y in s]
    if _subset_sum_check(target_x, others_x):
        return False
        
    # 2. Check if adding x creates conflict for existing y in s
    # Conflict if 1/y = sum(subset of (s \ {y}) U {x})
    # This implies 1/y - 1/x = sum(subset of s \ {y, x})
    # This is only possible if 1/y > 1/x => y < x
    for y in s:
        if y < x:
            target_y_minus_x = vals[y] - vals[x]
            if target_y_minus_x <= 0:
                continue
            others_y_minus_x = [vals[z] for z in s if z != y and z != x]
            if _subset_sum_check(target_y_minus_x, others_y_minus_x):
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_size = 0
    best_set = set()
    
    # Elements ordered 24 down to 1
    # Larger numbers are "safer" to include, so trying them first might find large valid sets quickly.
    # However, for B&B, the order affects pruning.
    # Let's try 24, 23, ..., 1.
    elements = list(range(24, 0, -1))
    
    current_set = set()
    
    def solve(idx):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        # Prune: if current size + remaining elements <= best size, no improvement possible
        remaining_count = len(elements) - idx
        if len(current_set) + remaining_count <= best_size:
            return
            
        if idx == len(elements):
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
            
        x = elements[idx]
        
        # Option 1: Include x
        if _is_incrementally_valid(current_set, x):
            current_set.add(x)
            solve(idx + 1)
            current_set.remove(x)
            
        # Option 2: Exclude x
        solve(idx + 1)
        
    solve(0)
    
    if not best_set:
        # Fallback if B&B failed or found nothing (should not happen, {24} is valid)
        best_set = {24}
        
    return sorted(best_set)

# EVOLVE-BLOCK-END

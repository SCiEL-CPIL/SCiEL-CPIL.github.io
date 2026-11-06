import time
import math
from itertools import combinations

# Precompute LCM of 1..24
def _lcm(a, b):
    return a * b // math.gcd(a, b)

_LCM_1_24 = 1
for i in range(1, 25):
    _LCM_1_24 = _lcm(_LCM_1_24, i)

# Weights w_v = LCM // v
_W = [_LCM_1_24 // i for i in range(1, 25)]

DEADLINE = 85.0
_START = None

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE

def _subset_sum_possible_int(vals, target):
    """True iff some subset of `vals` (ints) sums exactly to `target`."""
    n = len(vals)
    if n == 0:
        return target == 0
    if n <= 1:
        return any(v == target for v in vals)
    
    mid = n // 2
    left, right = vals[:mid], vals[mid:]

    L = {0}
    for v in left:
        L = L | {x + v for x in L}
    R = {0}
    for v in right:
        R = R | {x + v for x in R}

    if len(L) <= len(R):
        for l in L:
            if (target - l) in R:
                return True
    else:
        for r in R:
            if (target - r) in L:
                return True
    return False

def is_valid_int(S):
    """S is a list of indices 0-23 (representing 1-24)."""
    if not S:
        return False
    # S contains indices
    for i, a in enumerate(S):
        target = _W[a]
        # others are all indices in S except a
        others = [_W[v] for v in S if v != a]
        if _subset_sum_possible_int(others, target):
            return False
    return True

def construct():
    global _START
    _START = time.time()
    
    best = list(range(12, 24)) # 13..24 (indices 12..23)
    best_size = len(best)
    
    # B&B Search
    # We will search for a valid set.
    # State: current_set (list of indices), next_index (int)
    # We iterate through indices 0..23.
    # At each step, we decide to include or exclude index i.
    # To maximize size, we try to include first.
    
    # Heuristic ordering: 
    # Elements with smaller values (larger reciprocals) are more likely to be sums of others?
    # Actually, 1/1 = 1. Sum of any other reciprocals is < 1 (since max sum of 1/2..1/24 is < 1).
    # So 1 is NEVER a sum of others. 1 is always safe to include?
    # Wait. 1/1 = 1. Others are 1/2, 1/3... Sum of 1/2+1/3+1/6 = 1. 
    # So if 2,3,6 are in S, then 1 cannot be in S.
    # So 1 is not always safe.
    
    # Let's use a simple recursive B&B.
    # To speed up, we can precompute "conflicts" or just rely on the fast check.
    
    def dfs(current_set, start_idx):
        nonlocal best, best_size
        
        if _deadline_hit():
            return
            
        # Pruning: even if we add all remaining, can we beat best?
        remaining = 24 - start_idx
        if len(current_set) + remaining <= best_size:
            return
            
        # If we've considered all elements
        if start_idx == 24:
            if len(current_set) > best_size:
                best = current_set[:]
                best_size = len(current_set)
            return
            
        # Option 1: Exclude start_idx
        dfs(current_set, start_idx + 1)
        
        # Option 2: Include start_idx
        # Check if adding start_idx to current_set is valid
        # We need to check:
        # 1. Is w[start_idx] a sum of subset of current_set?
        # 2. For each a in current_set, is w[a] a sum of subset of (current_set \ {a}) U {start_idx}?
        
        # Optimization: 
        # If current_set is empty, adding start_idx is always valid.
        if not current_set:
            dfs(current_set + [start_idx], start_idx + 1)
            return # We already handled exclude above? No, we called exclude first.
            # Wait, standard B&B:
            # 1. Try include
            # 2. Try exclude
            # Order matters for finding large sets quickly.
            # Let's try Include first.
            
        # Re-structure DFS:
        # 1. Try Include
        # 2. Try Exclude
        
    # Let's rewrite the DFS logic properly.
    
    def search(current_set, start_idx):
        nonlocal best, best_size
        
        if _deadline_hit():
            return
            
        # Prune
        if len(current_set) + (24 - start_idx) <= best_size:
            return
            
        if start_idx == 24:
            if len(current_set) > best_size:
                best = current_set[:]
                best_size = len(current_set)
            return
        
        # Try to include start_idx
        # Check validity
        valid = True
        new_elem = start_idx
        w_new = _W[new_elem]
        
        # Check 1: w_new is not sum of subset of current_set
        if current_set:
            others_w = [_W[v] for v in current_set]
            if _subset_sum_possible_int(others_w, w_new):
                valid = False
        else:
            # Empty set, sum is 0, w_new > 0, so valid
            pass
            
        # Check 2: For each a in current_set, w[a] is not sum of subset of (current_set \ {a}) U {new_elem}
        if valid and current_set:
            for a in current_set:
                w_a = _W[a]
                # Subset of (current_set \ {a}) U {new_elem}
                # This is equivalent to: 
                # Is w_a - w_new a sum of subset of (current_set \ {a})? (if we pick new_elem)
                # OR is w_a a sum of subset of (current_set \ {a})? (if we don't pick new_elem)
                # But we are ADDING new_elem. The condition is that NO subset of the NEW set (excluding a) sums to w_a.
                # The new set of others is (current_set \ {a}) U {new_elem}.
                # So we need to check if w_a is a sum of a subset of (current_set \ {a}) U {new_elem}.
                
                # Let S' = current_set \ {a}
                # We need to check if w_a is in Sum(S' U {w_new})
                # This is true if:
                # (a) w_a is in Sum(S')  [subset doesn't use new_elem]
                # (b) w_a - w_new is in Sum(S') [subset uses new_elem]
                
                # Note: If (a) is true, then current_set was ALREADY invalid?
                # No, current_set is valid. So w_a is NOT a sum of subset of current_set \ {a}.
                # Wait. current_set is valid means: for each a in current_set, w_a is not sum of subset of (current_set \ {a}).
                # So (a) is ALWAYS FALSE for a valid current_set.
                # Therefore, we only need to check (b):
                # Is (w_a - w_new) a sum of a subset of (current_set \ {a})?
                
                if w_a < w_new:
                    # w_a - w_new is negative. Sum of positive ints cannot be negative.
                    # So (b) is false.
                    continue
                    
                target_diff = w_a - w_new
                others_ex_a = [_W[v] for v in current_set if v != a]
                if _subset_sum_possible_int(others_ex_a, target_diff):
                    valid = False
                    break
        
        if valid:
            search(current_set + [new_elem], start_idx + 1)
            
        # Try to exclude start_idx
        search(current_set, start_idx + 1)

    # Start search
    # Initial best is 13..24 (indices 12..23)
    # We can start with empty set and let B&B find the max.
    # But starting with a good seed helps.
    # Let's just run B&B from scratch, it should find 19+ quickly if ordering is good.
    # To ensure we find a large set fast, we can order indices by "difficulty" or just 0..23.
    # Including smaller numbers (larger weights) might be harder?
    # Let's try 0..23.
    
    search([], 0)
    
    # Convert indices back to values 1..24
    result = [i + 1 for i in best]
    
    if not result:
        # Fallback
        result = list(range(13, 25))
        
    return result

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

# Precompute LCM of 1..24
_LCM_24 = 1
for i in range(1, 25):
    _LCM_24 = _LCM_24 * i // math.gcd(_LCM_24, i)

# Precompute scaled reciprocals: SCALED[i] = _LCM_24 // i
_SCALED = [0] * 25
for i in range(1, 25):
    _SCALED[i] = _LCM_24 // i

def _is_safe_int(x, others):
    """
    Return True if SCALED[x] is NOT a subset sum of {SCALED[y] for y in others}.
    others is a list of integers.
    """
    if not others:
        return True
    
    target = _SCALED[x]
    n = len(others)
    
    # Meet-in-the-middle for subset sum
    mid = n // 2
    left = others[:mid]
    right = others[mid:]
    
    # Generate all subset sums for left
    left_sums = {0}
    for y in left:
        val = _SCALED[y]
        new_sums = set()
        for s in left_sums:
            ns = s + val
            if ns < target:
                new_sums.add(ns)
            elif ns == target:
                return False
        left_sums.update(new_sums)
    
    # Check if any left sum + right sum == target
    # We generate right sums and check against left_sums
    right_sums = {0}
    for y in right:
        val = _SCALED[y]
        new_sums = set()
        for s in right_sums:
            ns = s + val
            if ns < target:
                new_sums.add(ns)
            elif ns == target:
                return False
        right_sums.update(new_sums)
    
    for rs in right_sums:
        if rs > target:
            continue
        if (target - rs) in left_sums:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # Strategy 1: Greedy construction from largest to smallest.
    selected = set()
    for i in range(24, 0, -1):
        if time.time() > deadline:
            break
        if _is_safe_int(i, list(selected)):
            selected.add(i)
    
    best_set = set(selected)
    
    # Strategy 2: Local search to add excluded elements.
    # We iterate until no more elements can be added or time runs out.
    # Adding an element x requires checking if x is safe w.r.t current set.
    # Note: Adding x might make some existing y unsafe?
    # If y < x, 1/y > 1/x.
    # 1/y could be 1/x + sum(subset of S \ {y, x}).
    # So we must check all y < x in S.
    # However, _is_safe_int(x, list(best_set)) only checks x.
    # We need a function to check if adding x to S is valid.
    
    def _can_add(x, S):
        # 1. Check x
        if not _is_safe_int(x, list(S)):
            return False
        # 2. Check y in S where y < x
        for y in S:
            if y >= x:
                continue
            # Check if 1/y - 1/x is a subset sum of S \ {y, x}
            # 1/y - 1/x = (x-y)/(xy)
            # Scaled: _SCALED[y] - _SCALED[x]
            # We need to check if this difference is a subset sum of reciprocals of S \ {y, x}.
            # This is equivalent to: is (SCALED[y] - SCALED[x]) a subset sum of {SCALED[z] for z in S \ {y, x}}?
            # Note: SCALED[y] > SCALED[x] since y < x.
            target_diff = _SCALED[y] - _SCALED[x]
            others = [z for z in S if z != y and z != x]
            
            # We can reuse _is_safe_int logic but with a custom target.
            # _is_safe_int checks if SCALED[x] is a subset sum.
            # Here we check if target_diff is a subset sum.
            # Let's write a helper or adapt.
            
            # For simplicity and correctness, let's just check if the new set S U {x} is valid.
            # This is O(N * 2^(N/2)).
            # With N=24, it's fast enough with integer arithmetic.
            # But we can optimize:
            # We only need to check if target_diff is in the subset sums of others.
            
            if not others:
                if target_diff != 0: # target_diff > 0 always here
                    continue # Safe, no subset sum can match positive diff if others empty (sum=0)
                else:
                    return False # target_diff 0 means 1/y=1/x, impossible for distinct y,x
            
            # Meet in the middle for others
            n = len(others)
            mid = n // 2
            left = others[:mid]
            right = others[mid:]
            
            left_sums = {0}
            for z in left:
                val = _SCALED[z]
                new_sums = set()
                for s in left_sums:
                    ns = s + val
                    if ns < target_diff:
                        new_sums.add(ns)
                    elif ns == target_diff:
                        return False
                left_sums.update(new_sums)
            
            right_sums = {0}
            for z in right:
                val = _SCALED[z]
                new_sums = set()
                for s in right_sums:
                    ns = s + val
                    if ns < target_diff:
                        new_sums.add(ns)
                    elif ns == target_diff:
                        return False
                right_sums.update(new_sums)
            
            for rs in right_sums:
                if rs > target_diff:
                    continue
                if (target_diff - rs) in left_sums:
                    return False
                    
        return True

    excluded = [i for i in range(1, 25) if i not in best_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Sort excluded descending to try safer (larger) elements first?
        # Or just iterate.
        for x in sorted(excluded, reverse=True):
            if time.time() > deadline:
                break
            if _can_add(x, best_set):
                best_set.add(x)
                excluded.remove(x)
                improved = True
                break # Restart to re-evaluate with new set
    
    if not best_set:
        return [24]
        
    return sorted(best_set)
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_safe_int(candidate, others):
    """Return True if 1/candidate is NOT equal to sum of 1/y for any non-empty
    subset of `others`. Uses integer arithmetic via LCM scaling."""
    if not others:
        return True
    
    # Compute LCM of all elements in others plus candidate
    # We need: 1/candidate = sum of 1/y_i for some subset
    # Multiply both sides by L = lcm(candidate, y_1, ..., y_n)
    # L/candidate = sum of L/y_i
    
    L = candidate
    for y in others:
        L = _lcm(L, y)
    
    target_int = L // candidate
    
    # Now we need to check if any non-empty subset of {L//y for y in others} sums to target_int
    # This is a subset sum problem with integer values
    
    n = len(others)
    if n == 0:
        return True
    
    # If target_int is 0, impossible (all terms positive)
    if target_int <= 0:
        return True
    
    # Values to sum
    vals = [L // y for y in others]
    
    # Quick check: if any single val equals target_int, return False
    for v in vals:
        if v == target_int:
            return False
        if v > target_int:
            continue
    
    # Meet-in-the-middle
    mid = n // 2
    left_vals = vals[:mid]
    right_vals = vals[mid:]
    
    # Generate all subset sums for left half
    left_sums = set()
    left_sums.add(0)
    for v in left_vals:
        new_sums = set()
        for s in left_sums:
            ns = s + v
            if ns < target_int:
                new_sums.add(ns)
            elif ns == target_int:
                return False
        left_sums.update(new_sums)
    
    # Generate all subset sums for right half
    right_sums = set()
    right_sums.add(0)
    for v in right_vals:
        new_sums = set()
        for s in right_sums:
            ns = s + v
            if ns < target_int:
                new_sums.add(ns)
            elif ns == target_int:
                return False
        right_sums.update(new_sums)
    
    # Check combinations
    for rs in right_sums:
        if rs > target_int:
            continue
        remainder = target_int - rs
        if remainder in left_sums:
            # Need non-empty subset: if rs==0 and remainder==0, that's empty
            # But target_int > 0, so at least one side must be non-zero
            if rs == 0 and remainder == 0:
                continue
            return False
    
    return True

def _is_safe(candidate, others):
    """Return True if 1/candidate is NOT equal to sum of reciprocals of any non-empty
    subset of `others`."""
    return _is_safe_int(candidate, others)

def _is_valid_full(subset):
    """Check if the entire subset is reciprocal-subsum-free."""
    s = set(subset)
    for x in s:
        others = list(s - {x})
        if not _is_safe(x, others):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    best_set = set()
    best_size = 0
    
    # Strategy 1: Greedy construction from 24 down to 1
    selected = set()
    for i in range(24, 0, -1):
        if time.time() > deadline:
            break
        candidate_set = selected | {i}
        if _is_valid_full(candidate_set):
            selected = candidate_set
    
    if len(selected) > best_size:
        best_set = set(selected)
        best_size = len(selected)
    
    # Strategy 2: Try to add back some excluded elements (multiple passes)
    for _pass in range(3):
        if time.time() > deadline:
            break
        excluded = sorted(set(range(1, 25)) - selected, reverse=True)
        added_any = False
        for x in excluded:
            if time.time() > deadline:
                break
            new_candidate = selected | {x}
            if _is_valid_full(new_candidate):
                selected = new_candidate
                added_any = True
                if len(selected) > best_size:
                    best_set = set(selected)
                    best_size = len(selected)
        if not added_any:
            break
    
    # Strategy 3: Try different greedy orderings
    orderings = [
        list(range(24, 0, -1)),  # Already done
        list(range(1, 25)),      # Ascending
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [12, 24, 6, 18, 4, 20, 3, 21, 2, 22, 1, 23, 8, 16, 5, 15, 7, 14, 9, 13, 10, 11],  # Mixed
    ]
    
    for order in orderings:
        if time.time() > deadline - 5:
            break
        sel = set()
        for i in order:
            if time.time() > deadline - 2:
                break
            candidate_set = sel | {i}
            if _is_valid_full(candidate_set):
                sel = candidate_set
        if len(sel) > best_size:
            best_set = set(sel)
            best_size = len(sel)
    
    # Strategy 4: Local search - try to add elements by removing others
    if time.time() < deadline - 10:
        current = set(best_set)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            excluded = sorted(set(range(1, 25)) - current, reverse=True)
            current_list = sorted(current)
            
            for x in current_list:
                if time.time() > deadline - 5:
                    break
                for y in excluded:
                    if y in current:
                        continue
                    new_set = current - {x} | {y}
                    if len(new_set) <= best_size:
                        continue
                    # Try to add more elements after this swap
                    temp = set(new_set)
                    for z in sorted(set(range(1, 25)) - temp, reverse=True):
                        if time.time() > deadline - 3:
                            break
                        candidate_temp = temp | {z}
                        if _is_valid_full(candidate_temp):
                            temp = candidate_temp
                    if _is_valid_full(temp) and len(temp) > best_size:
                        best_set = set(temp)
                        best_size = len(temp)
                        current = set(temp)
                        improved = True
                        break
                if improved:
                    break
    
    # Strategy 5: Try starting with specific large subsets
    if time.time() < deadline - 15:
        # Try {13..24}
        base = set(range(13, 25))
        if _is_valid_full(base):
            for i in range(12, 0, -1):
                if time.time() > deadline - 5:
                    break
                new_set = base | {i}
                if _is_valid_full(new_set):
                    base = new_set
            if _is_valid_full(base) and len(base) > best_size:
                best_set = set(base)
                best_size = len(base)
    
    # Try {15..24}
    if time.time() < deadline - 10:
        base = set(range(15, 25))
        if _is_valid_full(base):
            for i in range(14, 0, -1):
                if time.time() > deadline - 5:
                    break
                new_set = base | {i}
                if _is_valid_full(new_set):
                    base = new_set
            if _is_valid_full(base) and len(base) > best_size:
                best_set = set(base)
                best_size = len(base)
    
    # Try {17..24}
    if time.time() < deadline - 10:
        base = set(range(17, 25))
        if _is_valid_full(base):
            for i in range(16, 0, -1):
                if time.time() > deadline - 5:
                    break
                new_set = base | {i}
                if _is_valid_full(new_set):
                    base = new_set
            if _is_valid_full(base) and len(base) > best_size:
                best_set = set(base)
                best_size = len(base)
    
    # Final check: ensure non-empty
    if not best_set:
        best_set = {24}
    
    return sorted(best_set)
# EVOLVE-BLOCK-END

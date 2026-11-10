import time
import random
from itertools import combinations

def _get_sums(S):
    """Compute all nonempty subset sums of S. Returns set or None if collision."""
    sums = set()
    n = len(S)
    for r in range(1, n + 1):
        for comb in combinations(S, r):
            s = sum(comb)
            if s in sums:
                return None
            sums.add(s)
    return sums

def _check_divisibility(sums_set):
    """Check if no two distinct values in sums_set divide one another."""
    if not sums_set:
        return True
    sums_list = sorted(sums_set)
    sum_set = set(sums_list)
    max_sum = sums_list[-1]
    for s in sums_list:
        multiple = s * 2
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += s
    return True

def _is_valid_set(S):
    """
    Check if S (list of distinct ints) satisfies:
    1) All nonempty subset sums are distinct.
    2) No two distinct nonempty subset sums divide one another.
    """
    if len(set(S)) != len(S):
        return False
    if any(not isinstance(x, int) or isinstance(x, bool) for x in S):
        return False
    sums = _get_sums(S)
    if sums is None:
        return False
    return _check_divisibility(sums)

def _try_construct_greedy(k, max_val=256, seed=None):
    """
    Attempt to construct a valid set of size k using greedy method with randomization.
    """
    if seed is not None:
        random.seed(seed)
    
    current = []
    current_sums = set()
    
    # Candidates: 1..max_val
    candidates = list(range(1, max_val + 1))
    random.shuffle(candidates)
    
    for x in candidates:
        if len(current) == k:
            break
        
        # Check if adding x is valid
        # New sums: {x} U {s + x for s in current_sums}
        new_sums = set(current_sums)
        valid = True
        
        # Check collision with existing sums
        if x in current_sums:
            continue # x is already a sum, so {x} collides with existing subset? 
            # Wait, x is an element. If x is a sum of other elements, then {x} and that subset have same sum.
            # So x cannot be in current_sums.
            
        for s in current_sums:
            ns = s + x
            if ns in new_sums:
                valid = False
                break
            new_sums.add(ns)
            
        if not valid:
            continue
            
        # Check divisibility
        if _check_divisibility(new_sums):
            current.append(x)
            current_sums = new_sums
            current.sort()
            
    if len(current) == k and _is_valid_set(current):
        return current
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_set = []
    
    # Try to find sets of size 11 down to 1
    for k in range(11, 0, -1):
        # Try multiple random restarts
        found = False
        attempts = 0
        max_attempts = 1000  # Limit attempts per k
        
        while not found and attempts < max_attempts and time.time() < deadline - 5:
            seed = attempts
            S = _try_construct_greedy(k, seed=seed)
            if S is not None:
                best_set = S
                found = True
            attempts += 1
            
        if found:
            break
            
    if not best_set:
        # Fallback to a known small valid set
        # [31, 32, 34] was valid?
        # Let's verify [31, 32, 34] again.
        # Sums: 31, 32, 34, 63, 65, 66, 97.
        # 31, 32, 34, 63, 65, 66, 97.
        # 31*2=62, 31*3=93.
        # 32*2=64, 32*3=96.
        # 34*2=68, 34*3=102.
        # 63*2=126.
        # 65*2=130.
        # 66*2=132.
        # 97*2=194.
        # No divisibility.
        # Distinct sums.
        # So [31, 32, 34] is valid.
        best_set = [31, 32, 34]
        
    # Final validation
    if _is_valid_set(best_set):
        return [int(x) for x in best_set]
    else:
        return [31, 32, 34]

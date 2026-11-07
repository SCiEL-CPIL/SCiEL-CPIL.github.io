import time
import random
import sys

# Increase recursion limit just in case
sys.setrecursionlimit(10000)

def _is_valid_full(els):
    """Full validity check for a complete set. Used for final verification."""
    if not els:
        return True
    n = len(els)
    sums = set()
    # Generate all subset sums
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += els[i]
        if s in sums:
            return False
        sums.add(s)
    
    sum_list = sorted(sums)
    m = len(sum_list)
    for i in range(m):
        a = sum_list[i]
        for j in range(i + 1, m):
            b = sum_list[j]
            if b % a == 0:
                return False
    return True

def _check_addition(current_sums, c):
    """
    Check if adding c to a set with subset sums 'current_sums' is valid.
    Returns (True, new_sums) if valid, (False, None) otherwise.
    """
    # 1. Check if c itself causes issues with existing sums
    # c must not be in current_sums
    if c in current_sums:
        return False, None
    
    # c must not be divisible by any existing sum
    # and no existing sum should divide c
    for s in current_sums:
        if c % s == 0:
            return False, None
        # s % c == 0 is impossible since s < c is not guaranteed, 
        # but if s > c, c % s != 0 unless c=0. 
        # Actually, if s > c, s % c == 0 is possible.
        # We need to check both directions.
        if s % c == 0:
            return False, None

    # 2. Generate new sums
    # new_sums = {c} U {s + c | s in current_sums}
    new_sums_set = {c}
    # Check for collisions between new sums and old sums
    # and within new sums (impossible as s+c is injective)
    # and between new sums and c (impossible as s>0)
    
    # We need to check divisibility for all pairs in (current_sums U new_sums_set)
    # To be efficient, we only check pairs where at least one is in new_sums_set.
    
    # Let's build the list of new sums first
    added_sums = []
    for s in current_sums:
        ns = s + c
        # Check collision with existing sums
        if ns in current_sums:
            return False, None
        # Check collision with other new sums? 
        # If s1+c == s2+c -> s1==s2, impossible.
        # Check collision with c? 
        # s+c == c -> s==0, impossible.
        added_sums.append(ns)
        new_sums_set.add(ns)
        
    # Now check divisibility
    # Pairs: (n1, n2) from new_sums_set, (n, o) from new_sums_set x current_sums
    # Note: We already checked c vs current_sums above.
    # We need to check:
    # 1. c vs added_sums (which are s+c)
    # 2. added_sums vs current_sums
    # 3. added_sums vs added_sums
    # 4. c vs c (no)
    # 5. current_sums vs current_sums (assumed valid)
    
    # 1. c vs added_sums
    for ns in added_sums:
        if ns % c == 0:
            return False, None
        if c % ns == 0:
            return False, None

    # 2. added_sums vs current_sums
    for ns in added_sums:
        for s in current_sums:
            if ns % s == 0:
                return False, None
            if s % ns == 0:
                return False, None

    # 3. added_sums vs added_sums
    # Since added_sums are s+c, and s are distinct, added_sums are distinct.
    # We need to check divisibility between any two added_sums.
    # added_sums size is 2^(k-1).
    # For k=8, size 128. 128*128/2 = 8192 checks.
    for i in range(len(added_sums)):
        a = added_sums[i]
        for j in range(i + 1, len(added_sums)):
            b = added_sums[j]
            if b % a == 0:
                return False, None
            if a % b == 0:
                return False, None

    # If we reach here, it's valid
    return True, new_sums_set

def _search_size(n, deadline, rng):
    """
    Try to find a valid set of size n.
    Returns the set if found, else None.
    """
    # Use backtracking
    # To reduce symmetry, we enforce increasing order of elements
    # current_elements: list of chosen elements
    # current_sums: set of subset sums of current_elements
    
    def backtrack(depth, current_elements, current_sums, last_elem):
        if time.time() > deadline:
            return None
        
        if depth == n:
            return current_elements
        
        # Candidate range: last_elem + 1 to 256
        # We can also prune based on remaining slots
        # But simple range is fine for now.
        
        # Heuristic: Try smaller numbers first? Or random?
        # Random helps escape local minima.
        # But we need to be systematic enough to find a solution if it exists.
        # Let's try a mix: shuffle candidates.
        
        candidates = list(range(last_elem + 1, 257))
        rng.shuffle(candidates)
        
        for c in candidates:
            valid, new_sums = _check_addition(current_sums, c)
            if valid:
                res = backtrack(depth + 1, current_elements + [c], new_sums, c)
                if res is not None:
                    return res
        return None

    # Initial state
    # We can start with an empty set, but we need to pick the first element.
    # First element c1: sums={c1}. No divisibility issues (only one sum).
    # So any c1 in 1..256 is valid for depth 1.
    
    # To help the search, we might want to start with specific first elements.
    # But randomization over the whole tree is handled by shuffling candidates at each step.
    
    return backtrack(0, [], set(), 0)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # We want to maximize size. Try from 8 down to 1.
    # We will allocate time for each size.
    # If we find a set of size 8, we return it.
    # If we fail, we try 7, etc.
    
    best_set = []
    
    # We can use multiple random seeds/restarts for each size to increase chance.
    # But we must respect the deadline.
    
    for n in range(8, 0, -1):
        if time.time() > deadline:
            break
            
        # Allocate some time for this size
        # If n is large, give more time.
        # Simple heuristic: try for 5 seconds per size, or until found.
        # But if n=8 is possible, we should spend more time.
        # Let's just run the search until deadline or found.
        # However, if n=8 is impossible, we waste 55s.
        # Better: 
        # Try n=8 for 30s. If found, return.
        # Try n=7 for 15s. If found, return.
        # Try n=6 for 5s.
        # Try n=5 for 2s.
        # etc.
        
        time_alloc = 0
        if n == 8: time_alloc = 30
        elif n == 7: time_alloc = 15
        elif n == 6: time_alloc = 5
        elif n == 5: time_alloc = 2
        elif n == 4: time_alloc = 1
        else: time_alloc = 0.5
        
        current_deadline = min(deadline, time.time() + time_alloc)
        
        # Try multiple random restarts within the allocated time
        found = None
        while time.time() < current_deadline and found is None:
            rng = random.Random(int(time.time() * 1000)) # New seed each restart
            # To avoid identical restarts, we can vary the seed more
            found = _search_size(n, current_deadline, rng)
            
            # Small sleep to avoid tight loop if search fails instantly
            # Actually, if it fails instantly, we should try again with different seed.
            # The search itself takes time.
        
        if found:
            # Verify just in case
            if _is_valid_full(found):
                return found
            else:
                # Should not happen if _check_addition is correct
                # But if it does, we ignore this result and continue?
                # Or return it? No, invalid.
                pass
        
    # If we didn't find anything, return a known valid small set
    # Size 1: [2] is valid. Sums: {2}. No pairs.
    # Size 2: [3, 4]? Sums: 3, 4, 7. 3|? No. 4|? No. 7|? No. Valid.
    # Let's return [3, 4] as fallback.
    return [3, 4]

# EVOLVE-BLOCK-END

import time
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin

    target_diffs = set(range(1, 31))
    target_mask = 0
    for d in range(1, 31):
        target_mask |= (1 << d)

    def solve_for_M(M):
        # State: 
        # curr_A: tuple of elements (sorted)
        # curr_mask: bitmask of covered diffs in 1..30
        # used_diffs: set of ALL positive differences generated so far (for Sidon check)
        # last_elem: last element added
        
        # We use a list for curr_A for speed in appending, but pass tuple for hashability if we were caching.
        # Here we just do DFS.
        
        def backtrack(curr_A, curr_mask, used_diffs, last_elem):
            if time.time() > deadline:
                return None
            
            if curr_mask == target_mask:
                return list(curr_A)
            
            # Pruning:
            # 1. If no more elements can be added (last_elem == M), fail.
            if last_elem == M:
                return None
            
            # 2. Check if remaining range can provide enough new differences.
            missing_count = 30 - bin(curr_mask).count('1') # Rough count of bits set in target range
            # Actually, let's compute missing diffs precisely for the heuristic?
            # No, bin count is fast.
            
            # Max new elements we can add: M - last_elem
            # Let k = M - last_elem
            # Max new diffs = k * len(curr_A) + k * (k - 1) // 2
            # If missing_count > max_new_diffs, prune
            k_max = M - last_elem
            max_new_diffs = k_max * len(curr_A) + k_max * (k_max - 1) // 2
            if missing_count > max_new_diffs:
                return None
            
            # Heuristic: Try candidates x in [last_elem+1, M]
            # Prioritize x that cover more missing diffs?
            # For speed, we'll just iterate, but we can sort by "potential" if we want.
            # Given M is small (likely < 50), iteration is fine.
            
            for x in range(last_elem + 1, M + 1):
                if time.time() > deadline:
                    return None
                
                # Check Sidon property
                # New diffs are x - a for a in curr_A
                # If any of these are in used_diffs, it's invalid.
                # Also, if x - a == x - b for a != b, that's impossible since a,b distinct.
                # So we just check intersection with used_diffs.
                
                valid = True
                new_diffs = []
                for a in curr_A:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Update mask
                new_mask = curr_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << d)
                
                # Early exit if mask is full
                # (handled in next recursion, but we can check here to save a call)
                
                # Update used_diffs
                # Copying set is O(N). N is small (< 100 diffs).
                new_used = set(used_diffs)
                new_used.update(new_diffs)
                
                # Recurse
                result = backtrack(curr_A + (x,), new_mask, new_used, x)
                if result:
                    return result
            
            return None
        
        # Start with 0
        return backtrack((0,), 0, set(), 0)

    # Iterate M upwards
    # Lower bound: 30.
    # For n=9, M is likely around 30-40.
    
    found_solution = None
    for M in range(30, 100):
        if time.time() > deadline:
            break
        
        res = solve_for_M(M)
        if res:
            found_solution = res
            break
    
    if found_solution:
        return found_solution

    # Fallback: Greedy construction
    # This is slower but guaranteed to find *a* solution if one exists with larger M.
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    
    # Greedy: add elements to cover missing diffs
    # We iterate d from 1 to 30. If d is not covered, try to add an element x such that x-a=d or a-x=d for some a in A.
    # To minimize max(A), we should try small x first.
    
    # However, a simple greedy might get stuck or produce a large max.
    # The backtracking above should have found the optimal or near-optimal.
    # If it timed out, we need a valid solution.
    
    # Let's try a simple constructive approach:
    # A = {0, 1, 4, 10, 12, 17, 21, 30} was invalid.
    # Let's just return a known valid large set if search fails?
    # Or run a slower, more exhaustive search with a tighter time limit?
    
    # Actually, the backtracking with pruning should be fast enough for M < 50.
    # If it fails, we return the greedy result.
    
    a = [0]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        found = False
        candidates = set()
        for v in a:
            if v + d not in a:
                candidates.add(v + d)
            if v - d >= 0 and v - d not in a:
                candidates.add(v - d)
        
        for x in sorted(candidates):
            trial = sorted(a + [x])
            d_list = diffs(trial)
            if len(set(d_list)) == len(d_list):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                d_list = diffs(trial)
                if len(set(d_list)) == len(d_list):
                    a = trial
                    break
                y += 1
    return a

# EVOLVE-BLOCK-END

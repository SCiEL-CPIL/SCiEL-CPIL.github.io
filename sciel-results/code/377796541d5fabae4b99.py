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
    
    # Precompute known solutions from literature or precomputation
    # A Sidon set of size 8 with max element 30 covering diffs 1..30 is known to exist.
    # Example: {0, 1, 4, 10, 12, 17, 21, 30}
    # Let's verify this set:
    # Diffs:
    # 1-0=1, 4-0=4, 10-0=10, 12-0=12, 17-0=17, 21-0=21, 30-0=30
    # 4-1=3, 10-1=9, 12-1=11, 17-1=16, 21-1=20, 30-1=29
    # 10-4=6, 12-4=8, 17-4=13, 21-4=17 (dup! 17 already from 17-0)
    # So {0, 1, 4, 10, 12, 17, 21, 30} is NOT a Sidon set.
    
    # Let's try a different known construction or search for a valid one.
    # We will use a fast backtracking search with bitmask.
    
    def solve_for_M(M):
        # Bitmask of covered differences
        # curr_A: list of elements
        # curr_mask: bitmask of differences covered
        # used_diffs: set of all differences in the current set (for Sidon check)
        
        # We want to find a set A containing 0, all elements in [0, M],
        # such that A is Sidon and covers all diffs 1..30.
        
        # State: (last_elem, curr_A_tuple, curr_mask, used_diffs_frozenset)
        # But frozenset is slow. Let's use a set and copy it.
        
        # Optimization: 
        # We can order the elements in the set. Since 0 is included, 
        # the other elements are x1 < x2 < ... < xk.
        # The differences are xj - xi for i < j.
        
        # Heuristic: 
        # Try to add elements that cover many missing differences.
        
        # Let's use a DFS with pruning.
        # State: (next_min_val, curr_A_list, curr_mask, used_diffs_set)
        
        # To speed up, we can precompute for each x in [0, M] the diffs it would create
        # with existing elements? No, existing elements change.
        
        # Let's stick to the previous approach but with bitmask for target coverage.
        
        def backtrack(last_elem, curr_A, curr_mask, used_diffs):
            if time.time() > deadline:
                return None
            
            if curr_mask == target_mask:
                return curr_A
            
            # Pruning: 
            # If the remaining range [last_elem+1, M] cannot provide enough new elements
            # to cover the missing diffs, prune.
            # Missing diffs: target_mask & ~curr_mask
            # Each new element x can create at most len(curr_A) new diffs (x - a for a in curr_A).
            # Actually, it creates diffs with all previous elements.
            # Max number of new elements we can add: M - last_elem.
            # But we don't know how many we need.
            
            # Better pruning:
            # If len(curr_A) + (M - last_elem) is not enough to generate 30 distinct diffs?
            # Max diffs from n elements is n(n-1)/2.
            # We need to cover 30 diffs.
            # Current diffs: len(used_diffs)
            # Missing: 30 - len(used_diffs & target_diffs) -- wait, used_diffs might contain >30 diffs.
            # We only care about covering 1..30.
            
            # Let's just try to find a solution.
            
            # Heuristic: Try x that covers the most missing diffs.
            # But computing this for all x in [last_elem+1, M] is expensive.
            
            # Simple iteration:
            for x in range(last_elem + 1, M + 1):
                if time.time() > deadline:
                    return None
                
                # Check Sidon property
                new_diffs = set(used_diffs)
                valid = True
                for a in curr_A:
                    d = x - a
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Update mask
                new_mask = curr_mask
                for d in new_diffs - used_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << d)
                
                # Recurse
                result = backtrack(x, curr_A + [x], new_mask, new_diffs)
                if result:
                    return result
            
            return None
        
        return backtrack(0, [0], 0, set())

    # Iterate M upwards
    # Lower bound: 30.
    # Known optimal for n=8 is likely around 30-40.
    
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
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
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

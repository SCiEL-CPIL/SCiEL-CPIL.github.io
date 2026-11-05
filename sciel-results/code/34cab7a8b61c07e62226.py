import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s margin for safety
    
    # Target differences
    target_diffs = set(range(1, 31))
    
    # We will search for the minimum possible max element M.
    # The set must contain 0.
    # We assume the set is sorted: 0 = x_0 < x_1 < ... < x_{k-1} = M.
    # We want to minimize M.
    
    # Lower bound for M:
    # A Sidon set of size k has at least k(k-1)/2 distinct differences.
    # We need to cover 1..30. So k(k-1)/2 >= 30 => k >= 8 (8*7/2 = 28 < 30, 9*8/2 = 36).
    # Actually, we don't know k. But M must be large enough.
    # For a perfect difference set of size k, M is roughly k^2.
    # Let's start searching M from a reasonable lower bound.
    
    # We can also use a greedy lower bound.
    # But let's just iterate M from 1 upwards.
    # Since the solution is small (likely < 100), this will be fast.
    
    def solve_for_max(M):
        # Returns a valid Sidon set with max element M, or None.
        # We use backtracking.
        # State: current set (list), used_diffs (set of differences).
        # We must end with max element M.
        
        # Optimization: 
        # If we are at the last step (adding M), we must check if adding M is valid.
        
        # We can prune if the current max is already > M (not possible since we build up to M).
        
        # Heuristic: 
        # To find a solution quickly, we can prioritize adding elements that cover missing diffs.
        
        # However, a simple DFS with pruning might be sufficient for small M.
        
        # Let's try a DFS that builds the set.
        # To ensure we finish with M, we can just search for a set of size k such that max is M.
        # But we don't know k.
        
        # Alternative: 
        # Just search for ANY Sidon set containing 0 and M, covering 1..30, with all elements <= M.
        
        # Let's use a recursive search.
        # current_set: sorted list of elements.
        # used_diffs: set of differences.
        
        # To speed up, we can use bitmasks or sets.
        
        def dfs(current_set, used_diffs, found_diffs):
            if time.time() > deadline:
                return None
            
            # Check if we can add M?
            # Actually, we just need to find a set.
            # If we have covered all diffs, we can stop?
            # No, we must have M in the set.
            # So we must add M at some point.
            
            # Let's enforce that M is in the set.
            # We can just search for a set containing 0 and M.
            
            # If M is already in current_set:
            if M in current_set:
                if found_diffs.issuperset(target_diffs):
                    return list(current_set)
                # If we have M but not all diffs, we can add more elements?
                # No, if we add more elements, max might exceed M?
                # The task says "at most 32 distinct integers in 0..10^6".
                # It does NOT say max element must be exactly M.
                # It says "Minimize the largest element".
                # So if we find a set with max M, that's a candidate.
                # But if we add more elements, max might stay M?
                # No, elements are distinct integers. If we add an element > M, max increases.
                # If we add an element < M, max stays M.
                # So we can add more elements < M to cover missing diffs.
                
                # So the condition is: 
                # 1. All elements <= M.
                # 2. M is in the set.
                # 3. All diffs 1..30 covered.
                # 4. Sidon property holds.
                
                # So if M is in set, but diffs not covered, we can try adding x < M.
                
            # If we have covered all diffs, and M is in set, we are done.
            if found_diffs.issuperset(target_diffs) and M in current_set:
                return list(current_set)
            
            # If we have covered all diffs, but M is NOT in set:
            # We MUST add M.
            # But we can also add other elements first?
            # No, if we add other elements, we might block M?
            # If we add x, it uses diffs x-y.
            # If we then add M, it uses M-y.
            # If M-y conflicts with x-y, we fail.
            # So it's safer to add M early?
            
            # Let's try to add M if it's not there.
            if M not in current_set:
                # Try adding M
                valid = True
                for y in current_set:
                    d = M - y
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    new_used = set(used_diffs)
                    new_found = set(found_diffs)
                    for y in current_set:
                        d = M - y
                        new_used.add(d)
                        if d in target_diffs:
                            new_found.add(d)
                    res = dfs(current_set + [M], new_used, new_found)
                    if res is not None:
                        return res
            
            # If we haven't found a solution yet (or adding M didn't work),
            # we can try adding other elements x < M.
            
            # What is the range for x?
            # x must be > last element? No, we keep set sorted.
            # So x > current_set[-1].
            # And x < M (if M not in set) or x < M (if M in set, x can be anything < M not in set).
            
            # Wait, if M is in set, we can add x < M.
            # If M is not in set, we can add x < M.
            
            # So x ranges from current_set[-1] + 1 to M - 1.
            # But we must ensure we don't exceed time.
            
            # Heuristic: 
            # We want to cover missing diffs.
            # missing = target_diffs - found_diffs
            # If missing is empty, we only care about adding M (if not present) or done.
            
            # Let's iterate x.
            # To prune, if we have too many elements (len > 31, since M is one), stop.
            # Actually, max size is 32.
            
            if len(current_set) >= 32:
                return None
            
            # Determine upper bound for x.
            # If M is in set, x can be up to M-1.
            # If M is not in set, x can be up to M-1.
            # So upper bound is M-1.
            
            # But we can prune x if it doesn't help?
            
            # Let's just iterate x from last+1 to M-1.
            # To speed up, we can check if x creates a valid Sidon set.
            
            last_val = current_set[-1]
            
            # Optimization: 
            # If we are missing a small diff d, and we haven't added M yet,
            # maybe we should add M first?
            # We already tried adding M above.
            
            # Let's iterate x.
            # We can break if x > M-1.
            
            for x in range(last_val + 1, M):
                if time.time() > deadline:
                    return None
                
                # Check if x can be added
                valid = True
                new_found = set(found_diffs)
                for y in current_set:
                    d = x - y
                    if d in used_diffs:
                        valid = False
                        break
                    if d in target_diffs:
                        new_found.add(d)
                
                if not valid:
                    continue
                
                new_used = set(used_diffs)
                for y in current_set:
                    new_used.add(x - y)
                
                res = dfs(current_set + [x], new_used, new_found)
                if res is not None:
                    return res
            
            return None
        
        # Start DFS with [0]
        return dfs([0], set(), set())

    # Search for minimum M
    # Lower bound: 
    # We need to cover 1..30.
    # Smallest Sidon set covering 1..30.
    # Let's start from M=1.
    # Actually, we can start from a better lower bound.
    # But M=1 is fine, it will fail fast.
    
    for M in range(1, 1000):
        if time.time() > deadline:
            break
        
        sol = solve_for_max(M)
        if sol is not None:
            # Verify solution
            # 1. Max is M
            if max(sol) != M:
                continue
            # 2. 0 in sol
            if 0 not in sol:
                continue
            # 3. Sidon
            diffs = []
            for i in range(len(sol)):
                for j in range(i+1, len(sol)):
                    diffs.append(sol[j] - sol[i])
            if len(diffs) != len(set(diffs)):
                continue
            # 4. Coverage
            if not set(diffs).issuperset(target_diffs):
                continue
            # 5. Size
            if len(sol) > 32:
                continue
            
            return sol
    
    # Fallback: Greedy
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                break
        else:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a

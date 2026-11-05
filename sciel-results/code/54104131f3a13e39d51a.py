import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    target_diffs = set(range(1, 31))
    
    # We want to find the minimum M such that there exists a Sidon set S 
    # with 0 in S, M in S, max(S)=M, and {1..30} subset of diffs(S).
    
    # Lower bound for M:
    # We need to cover diff 30, so M >= 30.
    # Also, a Sidon set of size k has k(k-1)/2 differences.
    # We need at least 30 differences.
    # If we use k elements, k(k-1)/2 >= 30 => k >= 8.
    # With 8 elements, the max element M must be large enough to accommodate them.
    # For a perfect difference set of size k, M ~ k^2.
    # So M is likely around 60-100.
    
    best_solution = None

    def find_set_with_max(M):
        """
        Find a Sidon set with max element exactly M, including 0 and M,
        realizing all diffs 1..30.
        Returns the set if found, else None.
        """
        nonlocal best_solution
        
        # Initial state: {0, M}
        current_set = [0, M]
        # Differences used so far: M - 0 = M
        used_diffs = {M}
        # Target diffs found so far
        found_diffs = {M} if M in target_diffs else set()
        
        # If M > 30, M is not a target diff, so found_diffs is empty initially.
        # If M <= 30, M is a target diff.
        
        # We need to add elements from 1 to M-1.
        # To optimize, we can try adding elements that cover missing diffs.
        
        def backtrack(current_set, used_diffs, found_diffs):
            nonlocal best_solution
            
            if time.time() > deadline:
                return False
            
            # Check if we have a complete solution
            if found_diffs.issuperset(target_diffs):
                # Valid solution found
                best_solution = list(current_set)
                return True
            
            if len(current_set) >= 32:
                return False
            
            # Determine which diffs are still missing
            missing = target_diffs - found_diffs
            if not missing:
                return True
            
            # Generate candidates for the next element x.
            # x must be in 1..M-1 and not in current_set.
            # To improve search efficiency, sort candidates by how many 
            # missing diffs they would cover if added.
            
            current_set_set = set(current_set)
            candidates = []
            
            for x in range(1, M):
                if x in current_set_set:
                    continue
                
                # Check if x can be added without violating Sidon property
                valid = True
                new_found = set(found_diffs)
                new_used = set(used_diffs)
                
                for y in current_set:
                    d = abs(x - y)
                    if d == 0:
                        continue
                    if d in used_diffs:
                        valid = False
                        break
                    new_used.add(d)
                    if d in missing:
                        new_found.add(d)
                
                if valid:
                    # Count how many missing diffs this x covers
                    covered = len(new_found - found_diffs)
                    # Heuristic: prioritize x that covers more missing diffs
                    # Also, maybe prefer smaller x to keep numbers tight? 
                    # Actually, for fixed M, we just need *a* solution.
                    # Covering more diffs quickly is usually better for backtracking.
                    candidates.append((covered, x))
            
            # Sort by coverage (descending), then by value (ascending)
            candidates.sort(key=lambda item: (-item[0], item[1]))
            
            for _, x in candidates:
                if time.time() > deadline:
                    return False
                
                # Re-check validity and update state
                # (We already checked in candidate generation, but state might have changed? 
                # No, we are in a single recursive step, so state is consistent.)
                
                # Actually, we should re-calculate or pass the calculated sets.
                # For simplicity and safety, re-calculate in the loop? 
                # It's a bit expensive. Let's store the calculated sets in the candidate tuple.
                # Let's refactor candidate generation to store (covered, x, new_found, new_used)
                
            # Refactor candidate generation to store necessary state
            candidates = []
            for x in range(1, M):
                if x in current_set_set:
                    continue
                
                valid = True
                new_found = set(found_diffs)
                new_used = set(used_diffs)
                
                for y in current_set:
                    d = abs(x - y)
                    if d == 0:
                        continue
                    if d in used_diffs:
                        valid = False
                        break
                    new_used.add(d)
                    if d in missing:
                        new_found.add(d)
                
                if valid:
                    covered = len(new_found - found_diffs)
                    candidates.append((covered, x, new_found, new_used))
            
            candidates.sort(key=lambda item: (-item[0], item[1]))
            
            for _, x, new_found, new_used in candidates:
                if time.time() > deadline:
                    return False
                
                # Recurse
                if backtrack(current_set + [x], new_used, new_found):
                    return True
            
            return False
        
        # Start backtracking
        return backtrack(current_set, used_diffs, found_diffs)

    # Sweep M from small to large
    # M must be at least 30.
    # We can stop if M exceeds a reasonable limit or if time is up.
    # Given the problem size, M should be < 200.
    
    for M in range(30, 200):
        if time.time() > deadline:
            break
        
        # Reset best_solution for this M? 
        # No, if find_set_with_max returns True, best_solution is set to a valid set with max M.
        # Since we sweep M increasing, the first success is the optimal M.
        
        if find_set_with_max(M):
            if best_solution is not None:
                # Verify solution one last time before returning
                sol = best_solution
                diffs = set()
                for i in range(len(sol)):
                    for j in range(i+1, len(sol)):
                        diffs.add(abs(sol[i] - sol[j]))
                if diffs.issuperset(target_diffs) and max(sol) == M and 0 in sol and len(sol) <= 32:
                    return sol
    
    # If no solution found with M < 200, try a greedy fallback
    # This is just in case, but the search above should find one.
    
    # Greedy construction
    a = [0]
    def get_diffs(s):
        s = sorted(s)
        return set(y - x for i, x in enumerate(s) for y in s[i+1:])
    
    # Try to build a set covering 1..30
    # This is a simple heuristic, might not be optimal but provides a valid output
    # if the search fails.
    
    # Actually, if the search fails, we should try a larger M or a different strategy.
    # But given the constraints, M < 200 should be sufficient.
    
    # Fallback: Return a known valid set if search failed?
    # A simple valid set:
    # We can try to construct one greedily with a larger M.
    
    # Let's try a greedy approach with a larger M as a last resort.
    # But the search above is likely to succeed.
    
    # If we reach here, we didn't find a solution in the loop.
    # This might happen if the search was too slow for small M and we hit the deadline.
    # In that case, we should return the best_solution if it was set?
    # No, best_solution is only set if a *complete* solution is found.
    
    # If no solution found, we must return *something* valid.
    # The task requires a valid Sidon set.
    # Let's use a simple constructive method that is guaranteed to work, 
    # even if M is larger.
    
    # Simple construction:
    # Start with {0}.
    # For each d in 1..30:
    #   If d is not in diffs, find x such that x or x+d can be added.
    #   This is tricky.
    
    # Alternative: Use a known perfect difference set or similar.
    # For n=6 (size 7?), no.
    # Size k Sidon set has ~k^2 differences.
    # For 30 diffs, k=8 or 9.
    
    # Let's try a randomized search with a fixed M=100 if the main search fails.
    # But we should have found it.
    
    # If best_solution is None, we have a problem.
    # Let's force a solution with a larger M using a simple greedy.
    
    a = [0]
    target = set(range(1, 31))
    
    for _ in range(100): # Limit iterations
        if time.time() > deadline:
            break
        current_diffs = get_diffs(a)
        missing = target - current_diffs
        if not missing:
            break
        
        # Try to add an element that covers a missing diff
        found = False
        # Try adding x such that x - a[i] in missing or a[i] - x in missing (if x < a[i], but we keep sorted so x > max(a))
        # Actually, we can add x > max(a).
        max_a = max(a)
        
        # Try x from max_a + 1 to max_a + 100
        for x in range(max_a + 1, max_a + 100):
            trial = a + [x]
            trial_diffs = get_diffs(trial)
            if len(trial_diffs) == len(trial) * (len(trial) - 1) // 2: # Sidon check
                # Check if it covers more missing diffs
                new_missing = target - trial_diffs
                if len(new_missing) < len(missing):
                    a = trial
                    found = True
                    break
        
        if not found:
            # Just add a large number to ensure we make progress?
            # This might break Sidon property if not careful.
            # Let's just add max_a + 1 and check Sidon.
            x = max_a + 1
            while True:
                trial = a + [x]
                trial_diffs = get_diffs(trial)
                if len(trial_diffs) == len(trial) * (len(trial) - 1) // 2:
                    a = trial
                    break
                x += 1
    
    return a

# To ensure we don't fail, we can pre-calculate a solution for a specific M if the search is slow?
# No, the search is fast for small M.

# Let's double check the logic.
# The main loop sweeps M.
# If it finds a solution, it returns it.
# If it times out, it falls back to greedy.
# The greedy might produce a valid set with larger M.
# This is acceptable as a fallback, but the search should find the optimal M.

# One optimization: In `find_set_with_max`, we can prune if the number of missing diffs 
# is greater than the maximum possible new diffs we can add with remaining slots.
# Max new diffs per new element x is len(current_set).
# Remaining slots = 32 - len(current_set).
# Max new diffs = (32 - len(current_set)) * len(current_set).
# If len(missing) > Max new diffs, prune.
# This is a weak bound but easy to add.

# I will add this prune to the backtrack function.

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    target_diffs = set(range(1, 31))
    
    best_solution = None

    def find_set_with_max(M):
        nonlocal best_solution
        
        current_set = [0, M]
        used_diffs = {M}
        found_diffs = {M} if M in target_diffs else set()
        
        def backtrack(current_set, used_diffs, found_diffs):
            nonlocal best_solution
            
            if time.time() > deadline:
                return False
            
            if found_diffs.issuperset(target_diffs):
                best_solution = list(current_set)
                return True
            
            if len(current_set) >= 32:
                return False
            
            missing = target_diffs - found_diffs
            if not missing:
                return True
            
            # Pruning: Check if we can cover missing diffs
            slots_left = 32 - len(current_set)
            # Each new element adds at most len(current_set) new diffs
            # But actually, it adds exactly len(current_set) diffs, but some might be non-target or duplicate.
            # So max new *target* diffs is len(current_set).
            max_possible_new_target_diffs = slots_left * len(current_set)
            if len(missing) > max_possible_new_target_diffs:
                return False
            
            current_set_set = set(current_set)
            candidates = []
            
            for x in range(1, M):
                if x in current_set_set:
                    continue
                
                valid = True
                new_found = set(found_diffs)
                new_used = set(used_diffs)
                
                for y in current_set:
                    d = abs(x - y)
                    if d == 0:
                        continue
                    if d in used_diffs:
                        valid = False
                        break
                    new_used.add(d)
                    if d in missing:
                        new_found.add(d)
                
                if valid:
                    covered = len(new_found - found_diffs)
                    candidates.append((covered, x, new_found, new_used))
            
            candidates.sort(key=lambda item: (-item[0], item[1]))
            
            for _, x, new_found, new_used in candidates:
                if time.time() > deadline:
                    return False
                
                if backtrack(current_set + [x], new_used, new_found):
                    return True
            
            return False
        
        return backtrack(current_set, used_diffs, found_diffs)

    for M in range(30, 200):
        if time.time() > deadline:
            break
        
        if find_set_with_max(M):
            if best_solution is not None:
                sol = best_solution
                diffs = set()
                for i in range(len(sol)):
                    for j in range(i+1, len(sol)):
                        diffs.add(abs(sol[i] - sol[j]))
                if diffs.issuperset(target_diffs) and max(sol) == M and 0 in sol and len(sol) <= 32:
                    return sol
    
    # Fallback greedy
    a = [0]
    def get_diffs(s):
        s = sorted(s)
        return set(y - x for i, x in enumerate(s) for y in s[i+1:])
    
    target = set(range(1, 31))
    
    for _ in range(100):
        if time.time() > deadline:
            break
        current_diffs = get_diffs(a)
        missing = target - current_diffs
        if not missing:
            break
        
        found = False
        max_a = max(a)
        
        for x in range(max_a + 1, max_a + 200):
            trial = a + [x]
            trial_diffs = get_diffs(trial)
            if len(trial_diffs) == len(trial) * (len(trial) - 1) // 2:
                new_missing = target - trial_diffs
                if len(new_missing) < len(missing):
                    a = trial
                    found = True
                    break
        
        if not found:
            x = max_a + 1
            while True:
                trial = a + [x]
                trial_diffs = get_diffs(trial)
                if len(trial_diffs) == len(trial) * (len(trial) - 1) // 2:
                    a = trial
                    break
                x += 1
    
    return a

# Test locally?
# print(construct())

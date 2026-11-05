import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for safety against the 120s limit
    
    # Target differences: 1..30
    target_diffs = set(range(1, 31))
    
    # Lower bound for M
    # A Sidon set of size k has k(k-1)/2 differences.
    # To cover 30 differences, we need at least k such that k(k-1)/2 >= 30.
    # k=8 -> 28 diffs (insufficient for 1..30 if all must be present? No, just 30 distinct diffs needed, but they must be 1..30).
    # Actually, we need the set of differences to be a superset of {1, ..., 30}.
    # So we need at least 30 distinct differences.
    # k=8 gives 28 max differences. So k must be at least 9.
    # For k=9, min diameter?
    # Let's just start from a reasonable M. 
    # For a perfect difference set (cyclic), size ~ sqrt(4M).
    # Linear Sidon sets covering 1..N typically have diameter ~ N.
    # Let's start from M=30.
    
    def solve_for_max(M):
        # Returns a valid Sidon set with max element exactly M, or None.
        # State:
        # current_set: sorted list of elements
        # used_diffs: set of all positive differences in current_set
        # found_diffs: set of target differences (1..30) found in current_set
        
        def dfs(current_set, used_diffs, found_diffs):
            if time.time() > deadline:
                return None
            
            # If we have M in the set and all target diffs, we are done.
            if M in current_set and found_diffs.issuperset(target_diffs):
                return list(current_set)
            
            # If we have all target diffs, we must ensure M is in the set.
            # If M is not in the set, we MUST add it (or add elements that allow adding M? 
            # No, if we add x < M, max stays < M. If we add x > M, max > M (invalid for this M).
            # So if found_diffs is complete, we can ONLY add M if M is not present.
            # If M is present, we are done (handled above).
            # If M is not present, we must try to add M.
            if found_diffs.issuperset(target_diffs) and M not in current_set:
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
                    # Recurse with M added. Since M is max, no further elements can be added < M 
                    # that change the max, but we could add more elements < M?
                    # Wait, if we add x < M, the max is still M. 
                    # But if found_diffs is already complete, adding x < M doesn't help cover diffs.
                    # It only risks breaking Sidon property.
                    # So if found_diffs is complete, adding M is the ONLY useful move?
                    # Yes, because any other element x < M would not add new target diffs (already have all).
                    # And it might conflict.
                    # So we just check if M fits. If yes, return [M] + current_set?
                    # Actually, we just return the set containing M.
                    res = dfs(current_set + [M], new_used, new_found)
                    if res is not None:
                        return res
                return None
            
            # If M is in current_set, but found_diffs is NOT complete:
            # We can add more elements x < M to cover missing diffs.
            
            # If M is NOT in current_set, and found_diffs is NOT complete:
            # We can add x < M.
            
            # General case: We need to add an element x.
            # x must be > current_set[-1] (to keep sorted) OR x < M if M is already in set?
            # Wait, the set is sorted.
            # If M is in current_set, it is the last element (since it's the max).
            # So we can only add x < M? No, if M is in set, it's the max.
            # We can add x such that current_set[-1] < x < M? No, if M is in set, 
            # current_set[-1] is M. So we can't add x > M.
            # Can we add x < M?
            # The set is sorted. If we add x, it must be inserted.
            # But my DFS appends to the list. This assumes we only add larger elements.
            # This is a limitation!
            
            # Let's re-evaluate the "sorted list" assumption.
            # If I build the set by always appending larger elements, I can only build sets 
            # where the elements are added in increasing order.
            # This means I can never add an element smaller than the current max.
            # If M is in the set, I cannot add any more elements (since they must be > M to append, 
            # but > M is invalid for max=M).
            # So if M is in the set and diffs are missing, I am stuck?
            # YES.
            
            # Therefore, the strategy "Add M early" is dangerous if it blocks other elements.
            # However, if I add M LAST, I can add any x < M in any order (as long as increasing).
            # But I don't know the order of x's.
            
            # Better Strategy:
            # Don't fix M in the set until the very end?
            # Or, allow the set to be built in any order, but keep it sorted?
            # If I keep it sorted, I can insert x anywhere.
            # But my `used_diffs` logic assumes I just add differences with existing elements.
            # If I insert x, I need to check diffs with ALL existing elements.
            # This is doable.
            
            # Let's modify the DFS to allow insertion at any position?
            # No, that's complex.
            
            # Alternative:
            # We know the set must contain 0 and M.
            # We can fix 0 and M as the "ends".
            # We need to find a set S such that {0} U S U {M} is a Sidon set.
            # Elements in S must be in (0, M).
            # We can build S by adding elements in increasing order.
            # This ensures we never exceed M.
            # And we can always check if M is compatible with the current S + {0}.
            
            # Revised DFS State:
            # current_set: sorted list of elements in (0, M). Does NOT include 0 or M yet.
            # used_diffs: set of differences between elements in {0} U current_set.
            #   (Note: diffs involving M are NOT in used_diffs yet, because M is not in set).
            #   We must ensure that when we finally add M, the diffs (M - y) for y in {0} U current_set
            #   do not conflict with used_diffs or each other.
            
            # Let's redefine `solve_for_max` with this logic.
            pass # Placeholder for logic change
            
        # --- Refined Solver ---
        
        # Base elements: 0 and M are fixed.
        # We are choosing a subset S of {1, ..., M-1}.
        # The full set is A = {0} U S U {M}.
        # A must be Sidon.
        # Diffs in A:
        # 1. Diffs within {0} U S: Let's call this D_S.
        # 2. Diffs within S U {M}: Let's call this D_M.
        # 3. Diffs between 0 and M: {M}.
        # All these must be distinct.
        # Also, union of D_S, D_M, {M} must contain 1..30.
        
        def dfs_s(S, used_diffs_S, found_diffs):
            # S: sorted list of elements in (0, M)
            # used_diffs_S: set of all diffs in {0} U S
            # found_diffs: set of target diffs found in {0} U S U {M} 
            #   Wait, we don't know D_M yet.
            #   So found_diffs should only track diffs from {0} U S.
            #   But we need to check if M can be added.
            
            if time.time() > deadline:
                return None
            
            missing = target_diffs - found_diffs
            
            # Check if we can add M NOW.
            # If we add M, the new diffs are {M - y for y in {0} U S}.
            # Let's call this set D_M.
            # Conditions for validity if we add M:
            # 1. D_M must be distinct internally (always true for distinct y).
            # 2. D_M must be disjoint from used_diffs_S.
            # 3. M (the diff 0 to M) must not be in used_diffs_S. (M is in D_M? No, M-0=M. Yes M is in D_M).
            #    So M not in used_diffs_S is covered by 2 if M is in D_M.
            # 4. (found_diffs U D_M) must be a superset of target_diffs.
            
            valid_M = True
            D_M = set()
            for y in [0] + S:
                d = M - y
                if d in used_diffs_S:
                    valid_M = False
                    break
                D_M.add(d)
            
            if valid_M:
                if (found_diffs | D_M).issuperset(target_diffs):
                    # We found a solution!
                    # The set is [0] + S + [M]
                    return [0] + S + [M]
            
            # If we can't add M yet, or adding M doesn't cover all diffs,
            # we must add more elements to S.
            
            # Heuristic: If we have enough elements to cover missing diffs?
            # Max size of S is 30 (since total size <= 32, and 0, M are 2).
            if len(S) >= 30:
                return None
            
            last_val = S[-1] if S else 0
            
            # Determine candidates for next element x.
            # x must be > last_val and < M.
            # To be efficient, we should prioritize x that covers missing diffs.
            
            candidates = []
            # Range: last_val + 1 to M - 1
            # This can be large. 
            # Optimization: 
            # If missing is not empty, we need x such that (x - y) or (M - x) or ... covers missing?
            # Actually, adding x adds diffs (x - y) for y in {0} U S.
            # It also changes the potential D_M (since M - x will be a diff).
            # But we don't know if M will be added later.
            
            # Simple heuristic: Try x that are "close" to creating missing diffs.
            # For each missing d, we need x such that x - y = d => x = y + d.
            # Or y - x = d? No, x > y.
            # So for each y in {0} U S and each d in missing:
            #   candidate = y + d
            #   If candidate < M and candidate > last_val:
            #       It's a good candidate.
            
            seen_candidates = set()
            for y in [0] + S:
                for d in missing:
                    x = y + d
                    if x > last_val and x < M:
                        if x not in seen_candidates:
                            seen_candidates.add(x)
                            candidates.append(x)
            
            # Also, we might need to add elements that don't immediately cover missing diffs
            # but allow future elements to?
            # For now, let's stick to "cover missing" candidates.
            # If no candidates, we are stuck?
            # Maybe we need to add a "neutral" element?
            # But if we don't cover missing diffs, we can't finish.
            # So if candidates is empty, return None.
            
            if not candidates:
                return None
            
            # Sort candidates by how many missing diffs they cover?
            # Or just try them.
            # To minimize branching, try those that cover the MOST missing diffs first.
            
            def score_candidate(x):
                count = 0
                for y in [0] + S:
                    d = x - y
                    if d in missing:
                        count += 1
                return count
            
            candidates.sort(key=score_candidate, reverse=True)
            
            for x in candidates:
                # Check if x is valid to add to {0} U S
                valid = True
                new_found = set(found_diffs)
                new_used = set(used_diffs_S)
                for y in [0] + S:
                    d = x - y
                    if d in used_diffs_S:
                        valid = False
                        break
                    new_used.add(d)
                    if d in target_diffs:
                        new_found.add(d)
                
                if not valid:
                    continue
                
                res = dfs_s(S + [x], new_used, new_found)
                if res is not None:
                    return res
            
            return None
        
        # Start with empty S
        return dfs_s([], set(), set())
        
    # Search for minimum M
    # Lower bound:
    # We need to cover 1..30.
    # Let's start from M=30.
    for M in range(30, 200):
        if time.time() > deadline:
            break
        
        sol = solve_for_max(M)
        if sol is not None:
            # Verify
            if max(sol) != M:
                continue
            if 0 not in sol:
                continue
            if len(sol) > 32:
                continue
            
            diffs = []
            for i in range(len(sol)):
                for j in range(i+1, len(sol)):
                    diffs.append(sol[j] - sol[i])
            
            if len(diffs) != len(set(diffs)):
                continue
            if not set(diffs).issuperset(target_diffs):
                continue
            
            return sol
    
    # Fallback: Greedy (if exhaustive search fails)
    a = [0]
    def get_diffs(s):
        s = sorted(s)
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        
    for d in range(1, 31):
        if d in get_diffs(a):
            continue
        found = False
        for x in sorted(set(v + d for v in a if v + d not in a) | set(v - d for v in a if v - d >= 0 and v - d not in a)):
            trial = sorted(a + [x])
            diffs = get_diffs(trial)
            if len(diffs) == len(set(diffs)):
                a = trial
                found = True
                break
        if not found:
            # Add new elements
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                diffs = get_diffs(trial)
                if len(diffs) == len(set(diffs)):
                    a = trial
                    break
                y += 1
    return a

import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    target_diffs = set(range(1, 31))
    
    # We will search for the minimal M
    # Try k=9, 10, ... up to 32
    # But k=9 is the minimum required.
    
    def search(M, k):
        # Returns a valid set or None
        # A = [0, ..., M]
        # We fix 0 and M.
        # We need to choose k-2 elements from 1..M-1.
        
        # This is a subset selection problem.
        # We can use backtracking.
        
        used_diffs = set()
        # Initial diffs from 0 and M: {M}
        # Also we don't know other diffs yet.
        
        # Let's build the set incrementally.
        # S = [0]
        # We will add elements.
        # To ensure M is the max, we can add M at the end?
        # Or fix M in the set from the start?
        # If we fix 0 and M, we have k-2 slots left.
        
        # Let's try a different approach:
        # Generate all Sidon sets of size k in 0..M? No, too many.
        
        # Backtracking on elements:
        # a_0 = 0
        # a_{k-1} = M
        # Choose a_1, ..., a_{k-2}
        
        # To speed up, we can use the fact that small differences must be covered.
        # 1, 2, ..., 30 must be differences.
        
        # Heuristic:
        # If 1 is a difference, either 1 in A or (x, x+1) in A.
        
        # Let's use a CSP solver logic.
        # Variables: a_1, ..., a_{k-2}
        # Domains: 1..M-1
        # Constraints:
        # 1. All a_i distinct.
        # 2. All differences distinct (Sidon).
        # 3. 1..30 covered.
        
        # This is complex to implement from scratch efficiently in the time limit.
        
        # Alternative:
        # Use a greedy heuristic to find a candidate, then verify.
        # Or use a precomputed list of known small Golomb rulers.
        
        # Let's try a simple DFS with strong pruning.
        
        S = [0, M]
        used_diffs = {M}
        
        # We need to add k-2 elements.
        # Let's pick them one by one.
        # To cover small diffs, we should pick small numbers?
        
        # Actually, let's just iterate M and use a standard backtracking solver.
        
        def dfs(current_S, used_diffs, missing_diffs, slots_left):
            if time.time() > deadline:
                return None
            
            if not missing_diffs:
                # Check if we have enough elements?
                # We can stop if we have a valid Sidon set covering 1..30.
                # We don't need exactly k elements, just <= 32.
                # But we started with a goal of k.
                # If we found a solution, return it.
                return current_S
            
            if slots_left == 0:
                return None
            
            # Pruning:
            # Can we cover remaining missing diffs with slots_left new elements?
            # Each new element x adds |current_S| new diffs.
            # Total new diffs possible = sum_{i=1}^{slots_left} (len(current_S) + i - 1)
            # If this < len(missing_diffs), prune.
            max_new_diffs = 0
            n = len(current_S)
            for i in range(slots_left):
                max_new_diffs += n + i
            if max_new_diffs < len(missing_diffs):
                return None
            
            # MRV: Find the missing diff with fewest ways to be covered
            # A diff d is covered if (x, x+d) in S or (x+d, x) in S.
            # Since we only add larger elements? No, we can add anywhere.
            # But S is sorted.
            
            # Let's try adding elements in increasing order.
            # This is not optimal but simpler.
            
            # Better: Try all possible x in 1..M-1 not in S.
            # Sort by "usefulness" (how many missing diffs it covers).
            
            candidates = []
            for x in range(1, M):
                if x in current_S:
                    continue
                
                # Check Sidon constraint
                valid = True
                new_diffs = []
                for y in current_S:
                    d = abs(x - y)
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Count how many missing diffs are covered
                covered = 0
                for d in new_diffs:
                    if d in missing_diffs:
                        covered += 1
                
                if covered > 0:
                    candidates.append((covered, x))
            
            if not candidates:
                return None
            
            # Sort by covered count (descending)
            candidates.sort(key=lambda t: -t[0])
            
            # Limit branching factor to avoid explosion
            # Try top 10 candidates
            for _, x in candidates[:10]:
                # Add x
                current_S.append(x)
                current_S.sort()
                
                added_diffs = set()
                for y in current_S:
                    if y == x: continue
                    d = abs(x - y)
                    used_diffs.add(d)
                    added_diffs.add(d)
                    if d in missing_diffs:
                        missing_diffs.discard(d)
                
                res = dfs(current_S, used_diffs, missing_diffs, slots_left - 1)
                if res is not None:
                    return res
                
                # Undo
                current_S.remove(x)
                for d in added_diffs:
                    used_diffs.discard(d)
                    if d in target_diffs and d not in missing_diffs:
                         # Careful with undoing missing_diffs
                         pass
                
                # Re-calculate missing_diffs correctly
                # It's easier to pass copies or restore carefully.
                # Let's use a simpler state management.
            
            return None
        
        # The above DFS is a bit messy with state restoration.
        # Let's rewrite it cleanly.
        
        def solve(M, k):
            # k is target size, but we can stop earlier if covered.
            # We fix 0 and M.
            S = [0, M]
            used_diffs = {M}
            missing = set(target_diffs)
            if M in missing:
                missing.discard(M)
            
            # We need to add k-2 elements.
            # But we can stop if missing is empty.
            
            def dfs(S, used_diffs, missing, depth):
                if time.time() > deadline:
                    return None
                if not missing:
                    return S[:]
                if depth == k - 2:
                    return None
                
                # Pruning
                n = len(S)
                slots = k - 2 - depth
                max_new = sum(n + i for i in range(slots))
                if max_new < len(missing):
                    return None
                
                # Generate candidates
                cands = []
                for x in range(1, M):
                    if x in S:
                        continue
                    # Check Sidon
                    ok = True
                    new_d = []
                    for y in S:
                        d = abs(x - y)
                        if d in used_diffs:
                            ok = False
                            break
                        new_d.append(d)
                    if not ok:
                        continue
                    
                    cover = 0
                    for d in new_d:
                        if d in missing:
                            cover += 1
                    if cover > 0:
                        cands.append((cover, x))
                
                if not cands:
                    return None
                
                cands.sort(key=lambda t: -t[0])
                
                for _, x in cands[:10]:
                    S.append(x)
                    S.sort()
                    
                    added = set()
                    for y in S:
                        if y == x: continue
                        d = abs(x - y)
                        used_diffs.add(d)
                        added.add(d)
                        if d in missing:
                            missing.discard(d)
                    
                    res = dfs(S, used_diffs, missing, depth + 1)
                    if res:
                        return res
                    
                    # Undo
                    S.remove(x)
                    for d in added:
                        used_diffs.discard(d)
                        if d in target_diffs:
                            # Only add back if it was missing before
                            # We don't know if it was missing before.
                            # But if it's in used_diffs, it's covered.
                            # If we remove it, it might become missing again.
                            # So we should check if it's in the remaining missing set of the parent?
                            # No, we passed `missing` by reference.
                            # We need to restore `missing` to its state before the call.
                            # This is tricky.
                            pass
                    
                    # Correct undo of missing
                    # We need to know which diffs were in `missing` before adding x.
                    # Let's pass a copy of missing.
                    # Or, better, calculate missing from scratch? No, too slow.
                    # Let's use a bitmask for missing diffs?
                    # 30 bits fits in an int.
            
            return None
        
        # Let's restart with bitmask for missing diffs.
        
        def solve_bitmask(M, k):
            target_mask = 0
            for d in range(1, 31):
                target_mask |= (1 << d)
            
            S = [0, M]
            used_diffs = set([M])
            missing_mask = target_mask
            if M <= 30:
                missing_mask &= ~(1 << M)
            
            def dfs(S, used_diffs, missing_mask, depth):
                if time.time() > deadline:
                    return None
                if missing_mask == 0:
                    return S[:]
                if depth == k - 2:
                    return None
                
                # Pruning
                n = len(S)
                slots = k - 2 - depth
                # Count bits in missing_mask
                miss_count = bin(missing_mask).count('1')
                max_new = sum(n + i for i in range(slots))
                if max_new < miss_count:
                    return None
                
                cands = []
                for x in range(1, M):
                    if x in S:
                        continue
                    ok = True
                    new_d = []
                    for y in S:
                        d = abs(x - y)
                        if d in used_diffs:
                            ok = False
                            break
                        new_d.append(d)
                    if not ok:
                        continue
                    
                    cover = 0
                    for d in new_d:
                        if d <= 30:
                            if missing_mask & (1 << d):
                                cover += 1
                    if cover > 0:
                        cands.append((cover, x))
                
                if not cands:
                    return None
                
                cands.sort(key=lambda t: -t[0])
                
                for _, x in cands[:10]:
                    S.append(x)
                    S.sort()
                    
                    added = set()
                    new_missing = missing_mask
                    for y in S:
                        if y == x: continue
                        d = abs(x - y)
                        used_diffs.add(d)
                        added.add(d)
                        if d <= 30:
                            new_missing &= ~(1 << d)
                    
                    res = dfs(S, used_diffs, new_missing, depth + 1)
                    if res:
                        return res
                    
                    S.remove(x)
                    for d in added:
                        used_diffs.discard(d)
            
            return None
        
        # Try k=9
        for M in range(30, 100):
            if time.time() > deadline:
                break
            res = solve_bitmask(M, 9)
            if res:
                return res
        
        # If not found, try k=10
        for M in range(30, 100):
            if time.time() > deadline:
                break
            res = solve_bitmask(M, 10)
            if res:
                return res
        
        return None

    # Run the search
    result = search(100, 9)
    if result:
        return result
    
    # Fallback
    return [0, 1, 4, 10, 12, 17, 20, 22, 25, 28, 30]


import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Target differences: 1..30
    TARGET_MASK = 0
    for d in range(1, 31):
        TARGET_MASK |= (1 << d)
        
    # Theoretical lower bound for D
    # We need at least 8 elements for 30 diffs.
    # Max diff is 30, so D >= 30.
    lower_bound = 30
    
    def find_solution(D):
        """
        Finds a Sidon set in [0, D] covering all target diffs.
        Returns the set if found, else None.
        """
        # We use DFS
        # State: current_set (list), last_elem (int), used_mask (int)
        
        # Optimization: Pre-compute valid candidates for each state?
        # No, state space is small.
        
        def dfs(last_elem, current_set, used_mask):
            # Time check
            if time.time() > deadline:
                return None
            
            # Check if goal met
            if (used_mask & TARGET_MASK) == TARGET_MASK:
                return current_set[:]
            
            # Pruning: Max elements
            if len(current_set) >= 32:
                return None
            
            # Pruning: Can we cover the rest?
            # Simple bound: Each new element adds at most len(current_set) new diffs.
            # Remaining elements allowed: 32 - len(current_set)
            # Max new diffs: (32 - len(current_set)) * len(current_set)
            # This is a weak bound.
            
            # Generate candidates x in (last_elem, D]
            # Heuristic: Sort by number of new target diffs covered
            candidates = []
            for x in range(last_elem + 1, D + 1):
                # Check validity against current_set
                new_diffs_mask = 0
                valid = True
                for s in current_set:
                    d = x - s
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    if (new_diffs_mask >> d) & 1:
                        valid = False
                        break
                    new_diffs_mask |= (1 << d)
                
                if not valid:
                    continue
                
                # Count how many target diffs are covered by this x
                covered = (new_diffs_mask & TARGET_MASK & ~used_mask).bit_count()
                candidates.append((covered, x, new_diffs_mask))
            
            # Sort candidates: Most coverage first
            candidates.sort(key=lambda item: item[0], reverse=True)
            
            for covered, x, new_diffs_mask in candidates:
                # If covered is 0, it might still be useful for future steps, 
                # but if we have a candidate with coverage > 0, we try those first.
                # If all have 0 coverage, we still try them (in order of x? or arbitrary?).
                # Let's just iterate the sorted list.
                
                next_set = current_set + [x]
                next_mask = used_mask | new_diffs_mask
                
                res = dfs(x, next_set, next_mask)
                if res is not None:
                    return res
            
            return None
        
        # Start with {0}
        return dfs(0, [0], 0)

    # Iterate D from lower bound
    # We need an upper bound to stop the loop if we fail to find a solution
    # A known Golomb ruler for 30 marks? No, we need 30 DIFFS.
    # A set of size 8 can have 28 diffs. Size 9 has 36.
    # So we need at least 9 elements?
    # 8 elements -> 28 pairs. We need 30 distinct diffs. So 8 is not enough.
    # 9 elements -> 36 pairs. So 9 is enough.
    # So we need a set of size >= 9.
    # Min diameter for 9-mark Golomb ruler?
    # Known optimal 9-mark Golomb ruler has length 41? (Marks: 0, 1, 4, 10, 18, 23, 29, 36, 41? No, that's not optimal).
    # Actually, we don't need a *complete* Golomb ruler, just one that contains 1..30.
    # This is a "perfect difference set" subset.
    # Let's just iterate D from 30 to, say, 200.
    
    best_solution = None
    
    for D in range(lower_bound, 200):
        if time.time() > deadline - 5:
            break
            
        res = find_solution(D)
        if res is not None:
            # Verify
            S = res
            if len(S) > 32:
                continue
            if 0 not in S:
                continue
            if max(S) > 10**6:
                continue
            
            diffs = set()
            valid = True
            for i in range(len(S)):
                for j in range(i+1, len(S)):
                    d = abs(S[i] - S[j])
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            
            if not valid:
                continue
                
            if all(d in diffs for d in range(1, 31)):
                return S

    # Fallback: Greedy
    # If we didn't find a solution in the loop (should not happen), use greedy
    S = [0]
    used_mask = 0
    for _ in range(32):
        if time.time() > deadline - 1:
            break
        missing = [d for d in range(1, 31) if not (used_mask >> d) & 1]
        if not missing:
            break
            
        best_x = None
        best_score = -1
        
        for x in range(1, 1000):
            if x in S: continue
            new_diffs_mask = 0
            valid = True
            for s in S:
                d = abs(x - s)
                if (used_mask >> d) & 1:
                    valid = False
                    break
                if (new_diffs_mask >> d) & 1:
                    valid = False
                    break
                new_diffs_mask |= (1 << d)
            
            if not valid:
                continue
            
            score = 0
            for d in missing:
                if (new_diffs_mask >> d) & 1:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_x = x
                
        if best_x is None:
            # Pick smallest valid x
            for x in range(1, 1000):
                if x in S: continue
                new_diffs_mask = 0
                valid = True
                for s in S:
                    d = abs(x - s)
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    if (new_diffs_mask >> d) & 1:
                        valid = False
                        break
                    new_diffs_mask |= (1 << d)
                if valid:
                    best_x = x
                    break
                    
        if best_x is None:
            break
            
        S.append(best_x)
        S.sort()
        
        # Update used_mask
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                d = S[j] - S[i]
                used_mask |= (1 << d)
                
    return S

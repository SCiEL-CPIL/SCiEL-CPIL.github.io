import time

def construct():
    deadline = time.time() + 140  # safety margin
    
    # We'll search for minimum M by trying increasing M
    # For each M, try to find a Sidon set in [0,M] containing 0 with all diffs 1..30
    
    # Backtracking state:
    # - chosen: sorted list of chosen elements (starts with [0])
    # - used_diffs: set of all differences realized so far
    # - covered: set of differences from 1..30 that are covered
    # - remaining: 1..30 not yet covered
    
    def backtrack(chosen, used_diffs, covered, M, max_elements=32):
        if time.time() > deadline:
            return None
        
        # Check if all 30 differences are covered
        if len(covered) == 30:
            return chosen[:]
        
        # Can we add more elements?
        if len(chosen) >= max_elements:
            return None
        
        # Pruning: can we cover remaining differences?
        # Remaining differences: 30 - len(covered)
        # With r more elements, we can add at most r * len(chosen) + r*(r-1)//2 new differences
        # (each new element x creates differences x - c for c in chosen, plus differences between new elements)
        # Actually, if we add r elements, new differences = r * len(chosen) + C(r, 2)
        # But we need specific differences, not just any
        # Simple bound: remaining = 30 - len(covered)
        # Max new diffs with r elements: r * len(chosen) + r*(r-1)//2
        # We need at least 'remaining' new diffs that are in 1..30 and not in used_diffs
        
        rem_needed = 30 - len(covered)
        r_max = max_elements - len(chosen)
        # Max possible new differences
        max_new = 0
        for r in range(1, r_max + 1):
            max_new = r * len(chosen) + r * (r - 1) // 2
            if max_new >= rem_needed:
                break
        if max_new < rem_needed:
            return None
        
        last = chosen[-1]
        # Try next element value from last+1 to M
        for x in range(last + 1, M + 1):
            if time.time() > deadline:
                return None
            
            # Compute new differences x - c for c in chosen
            new_diffs = []
            conflict = False
            for c in chosen:
                d = x - c
                if d in used_diffs:
                    conflict = True
                    break
                new_diffs.append(d)
            if conflict:
                continue
            
            new_used = used_diffs | set(new_diffs)
            new_covered = covered | set(d for d in new_diffs if 1 <= d <= 30)
            
            result = backtrack(chosen + [x], new_used, new_covered, M, max_elements)
            if result is not None:
                return result
        
        return None
    
    # Try increasing M
    # Lower bound: M must be at least 30 (to have difference 30)
    # Upper bound: start with something reasonable
    best_result = None
    
    for M in range(30, 200):  # try up to 200 first
        if time.time() > deadline:
            break
        
        result = backtrack([0], set(), set(), M)
        if result is not None:
            best_result = result
            break  # found minimum M
    
    if best_result is None:
        # Try larger M
        for M in range(200, 1000):
            if time.time() > deadline:
                break
            result = backtrack([0], set(), set(), M)
            if result is not None:
                best_result = result
                break
    
    if best_result is None:
        # Fallback: use the original greedy approach
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            found = False
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        best_result = a
    
    return best_result

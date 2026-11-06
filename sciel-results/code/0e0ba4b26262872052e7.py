import time
import sys
from itertools import combinations

def _check_sidon(a):
    """Check if a is a Sidon set."""
    diffs = set()
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _construct_diam(D, deadline):
    """
    Try to find a Sidon set in [0, D] covering diffs 1..30.
    Returns the set if found, else None.
    """
    target = set(range(1, 31))
    
    # If D == 30, 0 and 30 must be in the set.
    # If D > 30, we can be more flexible, but for minimal D, 
    # we start with D=30.
    
    start_set = [0]
    if D == 30:
        start_set.append(30)
        start_set.sort()
        # Pre-compute diffs from 30
        used = set()
        for x in start_set:
            for y in start_set:
                if x != y:
                    d = abs(x - y)
                    if d in used:
                        return None # Should not happen for {0, 30}
                    used.add(d)
        # 30 is covered.
        target.remove(30)
    else:
        used = set()

    # We need to fill the rest.
    # Max size 32.
    
    def backtrack(current, used_diffs, missing_diffs):
        if time.time() > deadline:
            return None
        
        if not missing_diffs:
            return list(current)
        
        if len(current) > 32:
            return None
            
        last = current[-1]
        
        # Pruning: Can we cover missing_diffs with remaining elements?
        # Each new element adds at most len(current) new diffs.
        # We have D - last space left.
        # This is a weak bound, but helps.
        
        candidates = []
        
        # Determine range for next element
        # We can add any x in (last, D]
        # But we must ensure we can still cover missing diffs.
        
        for x in range(last + 1, D + 1):
            new_diffs = set()
            valid = True
            for v in current:
                d = x - v
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
                
            # Check if this candidate helps
            covers = len(new_diffs & missing_diffs)
            if covers == 0:
                # Heuristic: skip if it doesn't cover any missing diff immediately?
                # Risky, but for D=30, space is tight.
                # Let's keep it but score it low.
                pass
                
            candidates.append((covers, x, new_diffs))
            
        if not candidates:
            return None
            
        # Sort by coverage (descending)
        candidates.sort(key=lambda t: -t[0])
        
        for covers, x, new_diffs in candidates:
            current.append(x)
            new_missing = missing_diffs - new_diffs
            new_used = used_diffs | new_diffs
            
            res = backtrack(current, new_used, new_missing)
            if res:
                return res
            current.pop()
            
        return None

    return backtrack(start_set, used, target)

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    # Minimal diameter must be at least 30.
    for D in range(30, 100):
        if time.time() > deadline - 5:
            break
            
        sol = _construct_diam(D, deadline - 5)
        if sol:
            # Verify
            if _check_sidon(sol):
                diffs = set()
                for i in range(len(sol)):
                    for j in range(i+1, len(sol)):
                        diffs.add(sol[j] - sol[i])
                if set(range(1, 31)).issubset(diffs):
                    return sol
                    
    # Fallback if not found (should not happen)
    # Greedy construction
    a = [0]
    used = set()
    for d in range(1, 31):
        found = False
        for x in range(1, 1000):
            if x in a:
                continue
            new_diffs = set()
            ok = True
            for v in a:
                diff = abs(x - v)
                if diff in used or diff in new_diffs:
                    ok = False
                    break
                new_diffs.add(diff)
            if ok and d in new_diffs:
                a.append(x)
                a.sort()
                used.update(new_diffs)
                found = True
                break
        if not found:
             # Force add
             x = max(a) + d + 1
             while True:
                 new_diffs = set()
                 ok = True
                 for v in a:
                     diff = x - v
                     if diff in used or diff in new_diffs:
                         ok = False
                         break
                     new_diffs.add(diff)
                 if ok:
                     a.append(x)
                     a.sort()
                     used.update(new_diffs)
                     break
                 x += 1
    return a

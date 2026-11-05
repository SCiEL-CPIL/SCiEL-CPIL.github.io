import sys
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Target differences
    target_diffs = set(range(1, 31))
    
    # Lower bound for M
    # For k=9, min length is approx 36-40.
    # Let's start from 36.
    
    def dfs(S, used_diffs, covered, M):
        """
        S: sorted list of elements
        used_diffs: set of all differences generated so far
        covered: set of differences in 1..30 generated so far
        M: max allowed element
        """
        if time.time() > deadline:
            return None
            
        # Check if all target differences are covered
        if target_diffs.issubset(covered):
            return S
            
        # If we have used too many elements, fail
        if len(S) >= 32:
            return None
            
        curr_max = S[-1]
        
        # Pruning: Can we cover the remaining differences?
        missing = target_diffs - covered
        if not missing:
            return S
            
        # Max possible new differences we can generate
        # If we add R more elements, the max number of new pairs is 
        # R * len(S) + R*(R-1)/2
        # But we only care about differences <= 30 for coverage.
        # A simple bound: if len(S) is small, we might not have enough pairs.
        # For now, just check if we can even add more elements.
        
        # Try adding new elements x
        # x must be > curr_max and <= M
        # We iterate x from curr_max + 1 to M
        # Optimization: We can skip x if it creates a difference already in used_diffs
        
        for x in range(curr_max + 1, M + 1):
            # Check Sidon property
            valid = True
            new_diffs = set()
            for s in S:
                d = x - s
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
                
            # Update state
            new_used_diffs = used_diffs | new_diffs
            new_covered = covered | (new_diffs & target_diffs)
            
            # Recurse
            res = dfs(S + [x], new_used_diffs, new_covered, M)
            if res is not None:
                return res
                
        return None

    # Iterate M from lower bound upwards
    # We know k>=9. 
    # Let's try M from 36 to 100.
    for M in range(36, 101):
        if time.time() > deadline:
            break
            
        # Start with 0
        initial_S = [0]
        initial_used_diffs = set()
        initial_covered = set()
        
        # Optimization: We can force the set to have a specific size?
        # No, let it grow.
        # But to speed up, we can try to find a solution with minimal size?
        # The DFS will find *a* solution.
        
        res = dfs(initial_S, initial_used_diffs, initial_covered, M)
        if res is not None:
            return res
            
    # If no solution found in range, return a fallback (should not happen)
    # Fallback: Greedy construction
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

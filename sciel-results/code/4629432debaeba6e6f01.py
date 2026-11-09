"""Rewritten program: backtracking DFS to find a tight Sidon set covering differences 1..30."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 135.0

    required = set(range(1, 31))
    
    best_result = [None]
    best_max_val = [float('inf')]
    
    def backtrack(current, diffs_set, n_target, max_bound):
        if time.time() > DEADLINE:
            return True  # signal to stop
        
        if len(current) == n_target:
            if required.issubset(diffs_set):
                mv = current[-1]
                if mv < best_max_val[0]:
                    best_max_val[0] = mv
                    best_result[0] = list(current)
            return False
        
        missing = required - diffs_set
        rem = n_target - len(current)
        max_new_diffs = rem * (len(current) + rem - 1)
        if len(missing) > max_new_diffs:
            return False
        
        lower = current[-1] + 1
        upper = min(max_bound, best_max_val[0] - 1)
        
        if lower > upper:
            return False
        
        # Generate candidates
        candidates = set()
        for a in current:
            for d in list(missing)[:15]:  # limit to keep it fast
                c = a + d
                if lower <= c <= upper:
                    candidates.add(c)
        
        # Add some sequential candidates
        for v in range(lower, min(lower + 10, upper + 1)):
            candidates.add(v)
        
        candidates = sorted(candidates)
        
        # Score and limit
        if len(candidates) > 30:
            def score(c):
                cnt = 0
                for a in current:
                    if (c - a) in missing:
                        cnt += 1
                return cnt
            candidates = sorted(candidates, key=score, reverse=True)[:30]
        
        for x in candidates:
            # Check validity
            valid = True
            for a in current:
                if (x - a) in diffs_set:
                    valid = False
                    break
            if not valid:
                continue
            
            new_diffs = diffs_set | {x - a for a in current}
            
            backtrack(current + [x], new_diffs, n_target, max_bound)
            
            if time.time() > DEADLINE:
                return True
            if best_result[0] is not None and best_max_val[0] <= upper:
                break
        
        return False
    
    # Try sizes from 9 up
    for n in range(9, 14):
        backtrack([0], set(), n, 10**6)
        if best_result[0] is not None:
            break
    
    if best_result[0] is not None:
        return best_result[0]
    
    # Fallback greedy
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
# EVOLVE-BLOCK-END

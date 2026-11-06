import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    best_solution = None
    best_max_val = float('inf')
    
    target_diffs = set(range(1, 31))
    
    def is_sidon(S):
        diffs = set()
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
        
    def get_covered(S):
        diffs = set()
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                diffs.add(S[j] - S[i])
        return diffs

    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Randomize the order of missing differences or start slightly differently
        # Let's just run the greedy algorithm with some randomization in tie-breaking
        
        S = [0]
        covered = get_covered(S)
        missing = target_diffs - covered
        
        while missing:
            if time.time() > deadline:
                break
                
            # Try to find a single element to add
            best_y = None
            best_gain = -1
            best_score = 0 # For tie breaking: smaller y is better
            
            # Candidates: s + d for s in S, d in missing
            # Also s - d
            candidates = set()
            for s in S:
                for d in missing:
                    candidates.add(s + d)
                    if s - d >= 0:
                        candidates.add(s - d)
                        
            # Filter candidates to a reasonable range to keep diameter small
            # Max element so far is max(S). We don't want to jump too far.
            # But we might need to.
            # Let's limit candidates to < 1000 for speed in this heuristic
            
            for y in candidates:
                if y in S:
                    continue
                if y > 1000: # Arbitrary bound, can be lowered
                    continue
                    
                new_S = sorted(S + [y])
                if not is_sidon(new_S):
                    continue
                
                new_covered = get_covered(new_S)
                new_missing = target_diffs - new_covered
                gain = len(missing) - len(new_missing)
                
                if gain > best_gain or (gain == best_gain and (best_y is None or y < best_y)):
                    best_gain = gain
                    best_y = y
                    best_score = y
                    
            if best_y is not None:
                S.append(best_y)
                S.sort()
                covered = get_covered(S)
                missing = target_diffs - covered
            else:
                # Single element didn't work. Try adding a pair?
                # This is complex. Let's just restart with different randomization
                # or break and try a different strategy
                # For now, restart
                break
        
        if not missing:
            max_val = max(S)
            if max_val < best_max_val:
                best_max_val = max_val
                best_solution = S.copy()
                
    if best_solution is not None:
        return best_solution
    else:
        # Fallback: return the seed solution or a valid one if found
        # The seed solution is valid, so we can just return a known valid set
        # Or run the seed logic
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

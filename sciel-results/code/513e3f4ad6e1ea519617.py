def construct():
    import sys
    import time
    sys.setrecursionlimit(10000)
    
    start_time = time.time()
    timeout = 110  # leave 10s margin
    
    needed_init = frozenset(range(1, 31))
    best_sol = [None]
    best_max = [10**7]
    
    def backtrack(elements, used_diffs, needed):
        if time.time() - start_time > timeout:
            return
        
        if not needed:
            curr_max = max(elements) if elements else 0
            if curr_max < best_max[0]:
                best_max[0] = curr_max
                best_sol[0] = list(elements)
            return
        
        if len(elements) >= 32:
            return
        
        curr_max = max(elements) if elements else 0
        
        # If current max already exceeds best, prune
        if curr_max >= best_max[0] and needed:
            # Still might find better with smaller max? No, max only increases.
            # But we might add elements that don't increase max? No, we add x > curr_max.
            # So if curr_max >= best_max[0], any solution from here has max >= curr_max >= best_max[0]
            # Unless we don't add more elements, but we need to cover needed.
            return
        
        # Generate candidates
        candidates = set()
        for a_i in elements:
            for d in needed:
                x = a_i + d
                if x > curr_max:
                    candidates.add(x)
        
        # Sort by x, but also prioritize those that cover more needed diffs
        def score(x):
            cnt = 0
            for a_i in elements:
                d = x - a_i
                if d in needed:
                    cnt += 1
            return (-cnt, x)
        
        for x in sorted(candidates, key=score):
            valid = True
            new_diffs = []
            for a_i in elements:
                d = x - a_i
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            new_elements = elements + [x]
            new_used = used_diffs | set(new_diffs)
            new_needed = needed - set(new_diffs)
            
            # Pruning
            rem_slots = 32 - len(new_elements)
            if rem_slots < 0:
                continue
            s = len(new_elements)
            max_new = rem_slots * s + rem_slots * (rem_slots - 1) // 2
            if max_new < len(new_needed):
                continue
            
            # Additional pruning: each remaining needed diff d must be achievable
            # d = x_new - a_i for some a_i in new_elements, or d = a_i - x_new (impossible since x_new > all)
            # So d must be in {x_new - a_i : a_i in new_elements} for some future x_new
            # This means for each d in new_needed, there must exist a_i in new_elements such that
            # x_new = a_i + d is "feasible" (not creating duplicate). Hard to check fully.
            
            backtrack(new_elements, new_used, new_needed)
            
            if time.time() - start_time > timeout:
                return
        
        # Fallback: try small x that don't cover needed diffs
        if best_max[0] > curr_max + 1 and len(needed) > 0:
            for x in range(curr_max + 1, min(curr_max + 20, best_max[0])):
                if x in elements:
                    continue
                valid = True
                new_diffs = []
                for a_i in elements:
                    d = x - a_i
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if not valid:
                    continue
                new_elements = elements + [x]
                new_used = used_diffs | set(new_diffs)
                new_needed = needed - set(new_diffs)
                rem_slots = 32 - len(new_elements)
                if rem_slots < 0:
                    continue
                s = len(new_elements)
                max_new = rem_slots * s + rem_slots * (rem_slots - 1) // 2
                if max_new < len(new_needed):
                    continue
                backtrack(new_elements, new_used, new_needed)
                if time.time() - start_time > timeout:
                    return
    
    backtrack([0], set(), needed_init)
    
    if best_sol[0] is not None:
        return best_sol[0]
    
    # Fallback greedy
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i+1:]]
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

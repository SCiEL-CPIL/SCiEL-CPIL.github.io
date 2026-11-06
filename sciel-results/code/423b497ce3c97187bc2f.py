import itertools
import time

def construct():
    n = 24
    elements = list(range(1, n+1))
    recips = {i: 1.0/i for i in elements}
    
    best_set = set(range(13, 25))  # Start with known good solution
    start_time = time.time()
    TIME_LIMIT = 55  # seconds
    
    def check_full_constraint(s):
        """Check if set s is reciprocal-subsum-free."""
        s_list = list(s)
        for x in s_list:
            target = recips[x]
            others = [y for y in s_list if y != x]
            for r in range(1, len(others)+1):
                for subset in itertools.combinations(others, r):
                    if abs(sum(recips[y] for y in subset) - target) < 1e-10:
                        return False
        return True
    
    def dfs(start_idx, current_set, remaining):
        nonlocal best_set
        if time.time() - start_time > TIME_LIMIT:
            return
        
        if len(current_set) + remaining <= len(best_set):
            return
        
        # Try to extend
        for i in range(start_idx, n):
            elem = elements[i]
            new_set = current_set | {elem}
            if len(new_set) <= len(best_set):
                continue
            # Quick check: only check constraints involving elem
            # 1/elem not in subset sums of current_set
            valid = True
            target = recips[elem]
            cs = list(current_set)
            for r in range(1, len(cs)+1):
                for subset in itertools.combinations(cs, r):
                    if abs(sum(recips[y] for y in subset) - target) < 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check for each y in current_set: 1/y not in subset sums of (current_set - {y}) ∪ {elem}
            for y in current_set:
                rem = recips[y] - recips[elem]
                if rem <= 1e-10:
                    continue
                others = [x for x in current_set if x != y]
                for r in range(0, len(others)+1):
                    if r == 0:
                        if abs(rem) < 1e-10:
                            valid = False
                            break
                        continue
                    for subset in itertools.combinations(others, r):
                        if abs(sum(recips[x] for x in subset) - rem) < 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                dfs(i+1, new_set, remaining - 1)
    
    dfs(0, set(), n)
    return sorted(best_set)

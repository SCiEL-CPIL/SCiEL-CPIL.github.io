import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    REQUIRED = set(range(1, 31))
    
    def valid_addition(elements, x, used_diffs):
        for y in elements:
            d = x - y
            if d > 0 and d in used_diffs:
                return False
        return True
    
    def get_new_diffs(elements, x):
        return set(x - y for y in elements if x - y > 0)
    
    def backtrack(elements, used_diffs, remaining, min_next, max_elem, max_size):
        if time.time() > deadline:
            return None
        
        if not remaining:
            return list(elements)
        
        if len(elements) >= max_size:
            return None
        
        m = max_size - len(elements)
        max_new = sum(len(elements) + i for i in range(m))
        if len(remaining) > max_new:
            return None
        
        candidates = []
        for x in range(min_next, max_elem + 1):
            if not valid_addition(elements, x, used_diffs):
                continue
            new_diffs = get_new_diffs(elements, x)
            covered = new_diffs & remaining
            candidates.append((len(covered), x))
        
        candidates.sort(key=lambda t: (-t[0], t[1]))
        
        for _, x in candidates[:50]:
            if time.time() > deadline:
                return None
            
            new_diffs = get_new_diffs(elements, x)
            new_remaining = remaining - new_diffs
            
            result = backtrack(
                elements + [x],
                used_diffs | new_diffs,
                new_remaining,
                x + 1,
                max_elem,
                max_size
            )
            if result is not None:
                return result
        
        return None
    
    for D in range(35, 5000):
        if time.time() > deadline:
            break
        for k in range(9, 33):
            if k * (k - 1) // 2 > D:
                break
            result = backtrack([0], set(), REQUIRED, 1, D, k)
            if result is not None:
                return result
    
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
    return a

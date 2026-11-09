import time
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin

    target_diffs = set(range(1, 31))
    target_mask = 0
    for d in range(1, 31):
        target_mask |= (1 << d)

    def solve_for_M(M):
        # State: 
        # curr_A: tuple of elements (sorted)
        # curr_mask: bitmask of covered diffs in 1..30
        # used_diffs: set of ALL positive differences generated so far (for Sidon check)
        # last_elem: last element added
        
        def backtrack(curr_A, curr_mask, used_diffs, last_elem):
            if time.time() > deadline:
                return None
            
            if curr_mask == target_mask:
                return list(curr_A)
            
            # Pruning 1: If no more elements can be added
            if last_elem == M:
                return None
            
            # Pruning 2: Capacity check
            missing_count = 30 - bin(curr_mask).count('1')
            if missing_count == 0:
                return list(curr_A)
                
            # Max elements we can add: M - last_elem
            # Let k = M - last_elem
            # Max new diffs = k * len(curr_A) + k * (k - 1) // 2
            k_max = M - last_elem
            max_new_diffs = k_max * len(curr_A) + k_max * (k_max - 1) // 2
            if missing_count > max_new_diffs:
                return None

            # Heuristic: Generate candidates and sort by coverage
            candidates = []
            for x in range(last_elem + 1, M + 1):
                # Check Sidon property
                valid = True
                new_diffs = []
                for a in curr_A:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Count how many missing diffs this x covers
                new_mask = curr_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << d)
                
                covered_new = bin(new_mask ^ curr_mask).count('1')
                candidates.append((covered_new, x, new_mask, new_diffs))
            
            # Sort by coverage descending, then x ascending (to prefer smaller x for compactness if coverage is same)
            candidates.sort(key=lambda t: (-t[0], t[1]))
            
            for covered_new, x, new_mask, new_diffs in candidates:
                if time.time() > deadline:
                    return None
                
                # If this x doesn't cover any new missing diff, and we are deep, maybe skip?
                # No, it might be necessary for Sidon structure even if it doesn't cover 1..30 directly?
                # Actually, if it doesn't cover any missing diff, it's only useful if it enables future elements to cover diffs?
                # But future elements are constrained by Sidon.
                # Usually, we want elements that cover diffs.
                # However, sometimes you need "filler" elements? No, Sidon sets are sparse.
                # If covered_new == 0, it's a very weak candidate. We can deprioritize it (already done by sort).
                
                new_used = set(used_diffs)
                new_used.update(new_diffs)
                
                result = backtrack(curr_A + (x,), new_mask, new_used, x)
                if result:
                    return result
            
            return None
        
        # Start with 0
        return backtrack((0,), 0, set(), 0)

    # Iterate M upwards
    # Lower bound: 30.
    # For n=9, M is likely around 30-40.
    # Actually, minimal Sidon set covering 1..30.
    # A perfect difference set of size k has k(k-1) differences.
    # We need 30 differences.
    # If k=8, 8*7=56 diffs.
    # If k=9, 9*8=72 diffs.
    # Minimal diameter for k=8 Sidon set covering 1..30?
    # Let's start from 30.
    
    found_solution = None
    for M in range(30, 100):
        if time.time() > deadline:
            break
        
        res = solve_for_M(M)
        if res:
            found_solution = res
            break
    
    if found_solution:
        return found_solution

    # Fallback: Greedy construction
    # This is slower but guaranteed to find *a* solution if one exists with larger M.
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        found = False
        candidates = set()
        for v in a:
            if v + d not in a:
                candidates.add(v + d)
            if v - d >= 0 and v - d not in a:
                candidates.add(v - d)
        
        for x in sorted(candidates):
            trial = sorted(a + [x])
            d_list = diffs(trial)
            if len(set(d_list)) == len(d_list):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                d_list = diffs(trial)
                if len(set(d_list)) == len(d_list):
                    a = trial
                    break
                y += 1
    return a

# EVOLVE-BLOCK-END

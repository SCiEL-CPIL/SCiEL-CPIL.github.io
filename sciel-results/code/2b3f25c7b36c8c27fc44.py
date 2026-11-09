import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    TARGET_MASK = (1 << 30) - 1  # bits 0..29 for diffs 1..30
    sys.setrecursionlimit(100000)
    
    def solve_for_D(D):
        """Find a Sidon set in [0,D] containing 0, covering all diffs 1..30."""
        
        used_diffs = [False] * (D + 1)
        elements = [0]
        
        def dfs(covered_mask, last_elem):
            if time.time() > deadline:
                return False
            
            if covered_mask == TARGET_MASK:
                return True
            
            # Generate candidates
            candidates = []
            
            for x in range(last_elem + 1, D + 1):
                valid = True
                new_cover = 0
                new_diffs = []
                
                for y in elements:
                    d = x - y
                    if d > D:
                        valid = False
                        break
                    if used_diffs[d]:
                        valid = False
                        break
                    new_diffs.append(d)
                    if 1 <= d <= 30 and not (covered_mask >> (d-1)) & 1:
                        new_cover += 1
                
                if valid and new_cover > 0:
                    candidates.append((new_cover, x, new_diffs))
            
            # Sort by coverage descending
            candidates.sort(key=lambda c: -c[0])
            
            for new_cover, x, new_diffs in candidates:
                # Apply
                for d in new_diffs:
                    used_diffs[d] = True
                
                new_mask = covered_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                
                elements.append(x)
                
                if dfs(new_mask, x):
                    return True
                
                # Undo
                elements.pop()
                for d in new_diffs:
                    used_diffs[d] = False
            
            return False
        
        # Start DFS
        if dfs(0, 0):
            return elements[:]
        return None

    # Try increasing D
    # Lower bound estimation:
    # Need 30 diffs. Min elements k such that k(k-1)/2 >= 30 -> k=9 (36 diffs).
    # Golomb ruler for 9 marks: optimal length is 70? 
    # Let's start from 50 to be safe, but 40 might work if sparse.
    
    for D in range(40, 250):
        if time.time() > deadline:
            break
        
        result = solve_for_D(D)
        if result is not None:
            return result

    # Fallback: greedy construction
    a = [0]
    all_diffs = set()
    for d_target in range(1, 31):
        if d_target in all_diffs:
            continue
        found = False
        for x in range(1, 500):
            if x in a:
                continue
            new_diffs = [x - y for y in a]
            if any(nd in all_diffs for nd in new_diffs):
                continue
            # Check all pairwise diffs distinct
            trial = sorted(a + [x])
            trial_diffs = set()
            ok = True
            for i in range(len(trial)):
                for j in range(i + 1, len(trial)):
                    dd = trial[j] - trial[i]
                    if dd in trial_diffs:
                        ok = False
                        break
                    trial_diffs.add(dd)
                if not ok:
                    break
            if ok:
                a = trial
                all_diffs = trial_diffs
                found = True
                break
        if not found:
            y = 2 * max(a) + d_target + 1
            while True:
                trial = sorted(a + [y, y + d_target])
                trial_diffs = set()
                ok = True
                for i in range(len(trial)):
                    for j in range(i + 1, len(trial)):
                        dd = trial[j] - trial[i]
                        if dd in trial_diffs:
                            ok = False
                            break
                        trial_diffs.add(dd)
                    if not ok:
                        break
                if ok:
                    a = trial
                    all_diffs = trial_diffs
                    break
                y += 1
    return a

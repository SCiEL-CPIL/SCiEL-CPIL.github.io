import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    TARGET_MASK = (1 << 30) - 1  # bits 0..29 for diffs 1..30
    sys.setrecursionlimit(100000)
    
    def solve_for_D(D):
        """Find a Sidon set in [0,D] containing 0, covering all diffs 1..30."""
        # Precompute: for each x in 1..D, which diffs 1..30 it could contribute
        # diff_mask[x] = bitmask of diffs d in 1..30 such that there exists y in [0, x-1] with x-y=d
        # But we need to check against actual current set elements.
        
        # used_diffs: boolean array of size D+1 (or use set for larger D)
        # For D up to ~200, a boolean array is fine
        
        best_result = [None]
        node_count = [0]
        
        def dfs(elements, used_diffs, covered_mask, last_elem):
            if time.time() > deadline:
                return False
            
            node_count[0] += 1
            
            if covered_mask == TARGET_MASK:
                best_result[0] = elements[:]
                return True
            
            # Forward checking: can remaining diffs be covered?
            # Remaining diffs count
            remaining_mask = TARGET_MASK & ~covered_mask
            remaining_count = bin(remaining_mask).count('1')
            
            # Max new diffs from one more element = len(elements)
            # We need at least ceil(remaining_count / len(elements)) more elements
            # But we also need each new element to be > last_elem and <= D
            max_addable = D - last_elem  # rough upper bound on new elements
            if max_addable < 1:
                return False
            
            # Generate candidates: elements x in (last_elem, D] that are valid
            # and prioritize those covering more missing diffs
            candidates = []
            for x in range(last_elem + 1, D + 1):
                # Check if x can be added (no diff collision)
                valid = True
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
                
                if not valid:
                    continue
                
                # Count how many missing diffs this covers
                new_cover = 0
                for d in new_diffs:
                    if 1 <= d <= 30 and not (covered_mask >> (d-1)) & 1:
                        new_cover += 1
                
                if new_cover > 0:
                    candidates.append((new_cover, x, new_diffs))
            
            # Sort by coverage (descending) to find solutions faster
            candidates.sort(key=lambda c: -c[0])
            
            for new_cover, x, new_diffs in candidates:
                # Apply
                new_used = used_diffs[:]
                new_mask = covered_mask
                for d in new_diffs:
                    if d <= D:
                        new_used[d] = True
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                
                if dfs(elements + [x], new_used, new_mask, x):
                    return True
            
            return False
        
        # Initial: elements=[0], no diffs used, no coverage
        used_diffs = [False] * (D + 1)
        success = dfs([0], used_diffs, 0, 0)
        
        if success and best_result[0] is not None:
            return best_result[0]
        return None
    
    # Try increasing D
    # Lower bound: need diff 30, so D >= 30. 
    # With k=9 elements in [0,D], max possible distinct diffs = 36, need 30 to be 1..30.
    # Known optimal Golomb rulers: for 9 marks, optimal diameter is ~70-80.
    # Let's start from a reasonable lower bound.
    
    for D in range(55, 250):
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

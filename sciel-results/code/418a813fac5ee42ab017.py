import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # We need a Sidon set S containing 0 with all differences 1..30 present
    # Minimize max(S)
    # 
    # Strategy: backtracking search
    # - Build set incrementally
    # - Track realized differences
    # - Prune using Sidon constraint
    
    def search(min_max, max_size=12):
        """
        Search for Sidon set with max element <= min_max
        that realizes all differences 1..30
        """
        target_diffs = set(range(1, 31))
        
        # State: current sorted list of elements, set of realized diffs
        # We always include 0
        elements = [0]
        realized = set()
        
        # For efficient checking, we need to know all current differences
        def get_all_diffs(els):
            d = set()
            for i in range(len(els)):
                for j in range(i+1, len(els)):
                    d.add(els[j] - els[i])
            return d
        
        realized = get_all_diffs(elements)
        
        def backtrack(els, real_diffs, depth, max_depth):
            if time.time() > deadline:
                return None
            
            # Check if we've realized all target differences
            if target_diffs.issubset(real_diffs):
                return els
            
            # Pruning: if we can't possibly realize all remaining diffs
            remaining = target_diffs - real_diffs
            if not remaining:
                return els
            
            # Maximum number of new differences we can add
            max_new_diffs = (max_depth - len(els)) * len(els) + (max_depth - len(els)) * (max_depth - len(els) - 1) // 2
            if len(remaining) > max_new_diffs:
                return None
            
            # Try adding new elements
            # The new element must be > max(els) to keep sorted
            # and <= min_max
            current_max = max(els)
            
            # Determine which differences we still need
            # For each candidate new element x, check Sidon constraint
            # x - e for e in els must not be in real_diffs, and must be distinct among themselves
            
            # Try candidates from current_max+1 to min_max
            # But we can be smarter: we need to realize specific differences
            # If d is in remaining, we need some pair (a,b) with b-a=d
            # If a is in els, then b = a+d must be added (if b > max(els) or b not in els)
            
            # Let's try a guided approach: prioritize elements that help realize remaining diffs
            candidates = set()
            for d in remaining:
                for e in els:
                    x = e + d
                    if x > current_max and x <= min_max:
                        candidates.add(x)
                    x2 = e - d
                    if 0 <= x2 < current_max and x2 not in els:
                        # This would need to be added, but we're building in increasing order
                        # So skip unless we allow reordering - we won't
                        pass
            
            # Also try some heuristic candidates
            # If we have few elements, try to build a structured set
            if not candidates:
                # Fallback: try sequential
                for x in range(current_max + 1, min_max + 1):
                    candidates.add(x)
                    if len(candidates) > 50:
                        break
            
            # Sort candidates by how many new differences they realize
            def score_candidate(x):
                new_diffs = set()
                for e in els:
                    d = x - e
                    if d > 0:
                        new_diffs.add(d)
                # Count how many of these are in remaining
                useful = len(new_diffs & remaining)
                # Penalize if any new diff is already in real_diffs (Sidon violation)
                conflicts = 0
                for e in els:
                    d = x - e
                    if d > 0 and d in real_diffs:
                        conflicts += 1
                # Also check pairwise conflicts among new diffs
                # For distinct e1, e2: (x-e1) and (x-e2) are distinct since e1 != e2
                # But (x-e1) might equal (e2'-e1') for some existing pair
                # Already covered by checking against real_diffs
                return (-conflicts, -useful, x)
            
            # Filter out candidates that violate Sidon
            valid_candidates = []
            for x in candidates:
                ok = True
                for e in els:
                    d = x - e
                    if d > 0 and d in real_diffs:
                        ok = False
                        break
                if ok:
                    valid_candidates.append(x)
            
            if not valid_candidates:
                return None
            
            # Sort by score
            valid_candidates.sort(key=score_candidate)
            
            # Limit branching factor
            for x in valid_candidates[:20]:
                if time.time() > deadline:
                    return None
                
                new_els = els + [x]
                new_real = set(real_diffs)
                for e in els:
                    d = x - e
                    if d > 0:
                        new_real.add(d)
                
                result = backtrack(new_els, new_real, depth + 1, max_depth)
                if result is not None:
                    return result
            
            return None
        
        # Try different max sizes
        for max_size in range(8, max_size + 1):
            result = backtrack(elements, realized, 1, max_size)
            if result is not None:
                return result
        return None
    
    # Binary search on the maximum element
    # Lower bound: need at least 30 diffs, min size ~8, so max element at least ~30
    # Upper bound: from seed program, likely a few hundred
    
    lo = 30
    hi = 500
    
    # First, try to find any solution with a reasonable upper bound
    # Use exponential search to find hi
    test_hi = 100
    while test_hi < 10000:
        result = search(test_hi)
        if result is not None:
            hi = test_hi
            break
        test_hi *= 2
    
    if hi > 10000:
        # Fallback to a known construction
        # Use the greedy approach from seed
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
    
    # Binary search for minimum
    best_result = None
    while lo < hi:
        if time.time() > deadline - 10:
            break
        mid = (lo + hi) // 2
        result = search(mid)
        if result is not None:
            best_result = result
            hi = mid
        else:
            lo = mid + 1
    
    if best_result is not None:
        return best_result
    
    # Try hi one more time
    result = search(hi)
    if result is not None:
        return result
    
    # Ultimate fallback: greedy
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

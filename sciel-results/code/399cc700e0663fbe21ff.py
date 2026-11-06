import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit

    # Required differences: 1 to 30
    # We represent the set of differences as a bitmask.
    # Bit i corresponds to difference i+1? Or difference i?
    # Let's use bit d for difference d. d ranges 1..30.
    # Mask for 1..30: (1 << 31) - 2  (bits 1 to 30 set)
    target_mask = 0
    for d in range(1, 31):
        target_mask |= (1 << d)

    def is_valid(a):
        """Check if a is a valid Sidon set covering 1..30."""
        diffs = 0
        n = len(a)
        for i in range(n):
            for j in range(i + 1, n):
                d = a[j] - a[i]
                if d > 30:
                    continue
                if diffs & (1 << d):
                    return False
                diffs |= (1 << d)
        return (diffs & target_mask) == target_mask

    # Try known candidates first (fast path)
    # Based on literature, minimal diameter for covering 1..30 might be around 32-36.
    # Let's try to find a good one quickly.
    
    # Backtracking search
    # State: current list of elements, bitmask of used diffs, last element added
    # We want to minimize the last element (diameter).
    
    best_result = None
    best_diameter = float('inf')

    def dfs(current, diff_mask, last, D_limit):
        nonlocal best_result, best_diameter
        
        if time.time() > deadline:
            return True # Timeout
        
        # Check if we have covered all required differences
        if (diff_mask & target_mask) == target_mask:
            # We have a valid set.
            # Since we build in increasing order, the last element is the diameter.
            if last < best_diameter:
                best_diameter = last
                best_result = list(current)
            return True # Stop this branch, we found a solution for this D_limit
            
        # Pruning:
        # 1. If we've exceeded the current best diameter, no need to continue.
        if last >= best_diameter:
            return False

        # 2. Can we possibly cover the remaining required diffs?
        # Remaining required diffs: target_mask & ~diff_mask
        # Max additional diffs we can generate by adding elements up to D_limit?
        # This is hard to estimate exactly, but we can use a loose bound.
        # Number of elements we can still add: roughly (D_limit - last)
        # Each new element adds at most len(current) new diffs.
        # So max new diffs ~ (D_limit - last) * len(current)
        # If this is less than the number of missing diffs, prune.
        missing_count = bin(target_mask & ~diff_mask).count('1')
        max_possible_new = (D_limit - last) * len(current) + (D_limit - last) * (D_limit - last - 1) // 2
        if max_possible_new < missing_count:
            return False

        # Try adding next element x
        # x must be > last
        # Also, to keep diameter small, we try smaller x first.
        for x in range(last + 1, D_limit + 1):
            if time.time() > deadline:
                return True
            
            # Check Sidon property for x
            ok = True
            new_diffs = 0
            for elem in current:
                d = x - elem
                if d > 30:
                    # Diffs > 30 don't affect the target mask, but they must be distinct from others.
                    # However, the problem says "all positive differences distinct".
                    # So we must track ALL differences, not just 1..30.
                    # My current diff_mask only tracks 1..30.
                    # I need to track all diffs to ensure Sidon property.
                    # This requires a different approach or a larger bitmask.
                    # Max D is likely < 100. So diffs < 100.
                    # Bitmask of size 100 is fine.
                    pass
                
                # Re-evaluating: I must track ALL differences to ensure uniqueness.
                # If I only track 1..30, I might miss a collision like d=50.
                # So I need a set of all differences or a larger bitmask.
                # Let's use a set for all diffs in the DFS state? 
                # That's slow.
                # Better: Use a bitmask for all diffs up to D_limit.
                # D_limit is small (< 100).
                # So I can use a bitmask of size 100.
                
                if diff_mask & (1 << d):
                    ok = False
                    break
                new_diffs |= (1 << d)
            
            if not ok:
                continue
            
            # Also check if new_diffs collide with each other?
            # x - elem1 == x - elem2 implies elem1 == elem2, which is false.
            # So new_diffs are distinct from each other.
            # They are distinct from old diffs because we checked against diff_mask.
            
            current.append(x)
            if dfs(current, diff_mask | new_diffs, x, D_limit):
                current.pop()
                return True
            current.pop()
        
        return False

    # Search for minimal D
    # Lower bound: 30 (must cover 30)
    # Upper bound: start with a reasonable guess, e.g., 50
    for D in range(30, 60):
        if time.time() > deadline:
            break
        # Reset best for this D? No, we want global best.
        # But if we find a solution with diameter D, we can stop searching for larger D?
        # Yes, we are iterating D upwards. The first valid D is the minimal diameter.
        # However, the DFS might not find a solution for D even if one exists, due to pruning or timeout.
        # So we should be careful.
        
        # We need to track ALL diffs in the mask now.
        # Let's modify the DFS to use a mask for all diffs up to D.
        
        def dfs_all(current, diff_mask, last, D_limit):
            nonlocal best_result, best_diameter
            
            if time.time() > deadline:
                return True
            
            if (diff_mask & target_mask) == target_mask:
                if last < best_diameter:
                    best_diameter = last
                    best_result = list(current)
                return True
            
            if last >= best_diameter:
                return False
            
            # Pruning
            missing_count = bin(target_mask & ~diff_mask).count('1')
            # Loose bound on max new diffs
            # Elements remaining: D_limit - last
            # Max new diffs: k * (D_limit - last) where k = len(current)
            if (len(current) * (D_limit - last)) < missing_count:
                return False

            for x in range(last + 1, D_limit + 1):
                if time.time() > deadline:
                    return True
                
                ok = True
                new_diffs = 0
                for elem in current:
                    d = x - elem
                    if d > D_limit: # d cannot exceed D_limit
                        ok = False
                        break
                    if diff_mask & (1 << d):
                        ok = False
                        break
                    new_diffs |= (1 << d)
                
                if not ok:
                    continue
                
                current.append(x)
                if dfs_all(current, diff_mask | new_diffs, x, D_limit):
                    current.pop()
                    return True
                current.pop()
            
            return False

        dfs_all([0], 0, 0, D)
        if best_result is not None:
            return best_result

    # Fallback if no solution found in range
    # Greedy construction
    def greedy():
        a = [0]
        covered = set()
        for d in range(1, 31):
            if d in covered:
                continue
            cands = sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a))
            found = False
            for x in cands:
                trial = a + [x]
                diffs = set()
                ok = True
                for i in range(len(trial)):
                    for j in range(i + 1, len(trial)):
                        dd = trial[j] - trial[i]
                        if dd in diffs:
                            ok = False
                            break
                        diffs.add(dd)
                    if not ok:
                        break
                if ok and d in diffs:
                    a = sorted(trial)
                    covered = diffs
                    found = True
                    break
            if not found:
                # Add a large element to ensure d is covered
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = a + [y, y + d]
                    diffs = set()
                    ok = True
                    for i in range(len(trial)):
                        for j in range(i + 1, len(trial)):
                            dd = trial[j] - trial[i]
                            if dd in diffs:
                                ok = False
                                break
                            diffs.add(dd)
                        if not ok:
                            break
                    if ok:
                        a = sorted(trial)
                        covered = diffs
                        break
                    y += 1
        return a

    return greedy()

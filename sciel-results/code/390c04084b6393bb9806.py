import time
import random

def construct():
    """
    Find a Sidon set containing 0 with all differences 1..30, minimizing the largest element.
    Uses CSP with MRV (Minimum Remaining Values) ordering: at each step, select the
    most constrained uncovered difference and try placements that satisfy it.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Leave 5s margin before 150s kill
    TARGET = frozenset(range(1, 31))
    TARGET_SET = set(range(1, 31))

    def get_diffs(S):
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                diffs.add(S[j] - S[i])
        return diffs

    def is_sidon(S):
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def covers_all(S):
        diffs = get_diffs(S)
        return TARGET_SET.issubset(diffs)

    def solve_csp(max_diameter, max_size=32):
        """
        CSP solver: find a Sidon set S with 0 in S, max(S) <= max_diameter,
        |S| <= max_size, and all differences 1..30 present.
        
        Uses MRV: at each step, pick the uncovered difference with the fewest
        valid placement options, then try those placements.
        """
        # State: S (sorted list), used_diffs (set of differences already in S)
        # We build S incrementally. Start with S = [0].
        
        def search(S, used_diffs, depth):
            if time.time() - start_time > DEADLINE - 5:
                return None
            
            if TARGET_SET.issubset(used_diffs):
                return list(S)
            
            if len(S) >= max_size:
                return None
            
            # MRV: find uncovered diff with fewest placement options
            uncovered = TARGET_SET - used_diffs
            if not uncovered:
                return list(S)
            
            # For each uncovered difference d, find valid placements (x, y) where
            # y - x = d, x >= 0, y <= max_diameter, and adding both to S maintains Sidon.
            # A placement is valid if:
            #   - x not in S, y not in S (or x/y already in S)
            #   - For all s in S: |s - x| and |s - y| not in used_diffs (and not 0)
            #   - x != y (guaranteed since d > 0)
            
            best_diff = None
            best_placements = None
            best_count = float('inf')
            
            for d in uncovered:
                placements = []
                # x can range from 0 to max_diameter - d
                # y = x + d
                for x in range(0, max_diameter - d + 1):
                    y = x + d
                    # Check x and y not already in S (unless they are)
                    x_in_S = x in S
                    y_in_S = y in S
                    if x_in_S and y_in_S:
                        continue  # Both already present, diff already covered
                    # Check that adding x and y doesn't create duplicate differences
                    valid = True
                    new_elements = []
                    if not x_in_S:
                        new_elements.append(x)
                    if not y_in_S:
                        new_elements.append(y)
                    
                    for elem in new_elements:
                        for s in S:
                            diff = abs(elem - s)
                            if diff == 0:
                                continue
                            if diff in used_diffs:
                                valid = False
                                break
                        if not valid:
                            break
                        # Also check diff against other new elements
                        for other in new_elements:
                            if other != elem:
                                diff2 = abs(elem - other)
                                if diff2 in used_diffs:
                                    valid = False
                                    break
                        if not valid:
                            break
                    
                    if valid:
                        # Also check: new differences from new elements to each other
                        # and to existing elements are all distinct
                        all_new_diffs = set()
                        trial_S = list(S)
                        for elem in new_elements:
                            trial_S.append(elem)
                        trial_S.sort()
                        # Check all pairs in trial_S
                        for i in range(len(trial_S)):
                            for j in range(i + 1, len(trial_S)):
                                dd = trial_S[j] - trial_S[i]
                                if dd in all_new_diffs:
                                    valid = False
                                    break
                                all_new_diffs.add(dd)
                            if not valid:
                                break
                        if valid:
                            placements.append((x, y))
                
                count = len(placements)
                if count < best_count:
                    best_count = count
                    best_diff = d
                    best_placements = placements
                    if count == 0:
                        break
                    if count <= 3:
                        break  # Enough to pick from
            
            if best_count == 0:
                return None
            
            # Try placements for best_diff in order (prefer smaller max element)
            random.shuffle(best_placements)
            # Sort by resulting max element
            best_placements.sort(key=lambda p: max(p))
            
            for (x, y) in best_placements:
                new_elements = []
                if x not in S:
                    new_elements.append(x)
                if y not in S:
                    new_elements.append(y)
                
                trial_S = sorted(S + new_elements)
                
                # Recompute used_diffs
                trial_diffs = get_diffs(trial_S)
                
                result = search(trial_S, trial_diffs, depth + 1)
                if result is not None:
                    return result
            
            return None
        
        # Initial state
        S0 = [0]
        used0 = set()  # No differences yet
        return search(S0, used0, 0)

    def local_refine(S, time_limit=10.0):
        """Try to reduce max(S) by replacing elements with smaller values."""
        S = sorted(S)
        best = S[:]
        best_max = max(S)
        t0 = time.time()
        
        while time.time() - t0 < time_limit:
            improved = False
            for i in range(len(S)):
                if S[i] == 0:
                    continue
                old_val = S[i]
                for new_val in range(1, old_val):
                    if new_val in S:
                        continue
                    trial = S[:]
                    trial[i] = new_val
                    trial.sort()
                    if not is_sidon(trial):
                        continue
                    if not covers_all(trial):
                        continue
                    if max(trial) < best_max:
                        S = trial
                        best = S[:]
                        best_max = max(S)
                        improved = True
                        break
            if not improved:
                break
        return best

    # Binary search on diameter
    # Lower bound: need at least 30 distinct differences, so diameter >= 30
    # Upper bound: start with a known feasible value
    lo = 30
    hi = 200  # Reasonable upper bound
    
    # First, verify hi is feasible with a quick check
    best_S = None
    
    # Try increasing diameters
    for diam in range(lo, hi + 1):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        result = solve_csp(diam)
        if result is not None:
            best_S = result
            break
    
    # If we found a solution, try to refine it
    if best_S is not None:
        remaining = DEADLINE - 5 - (time.time() - start_time)
        if remaining > 2:
            refined = local_refine(best_S, time_limit=min(remaining - 1, 15))
            if refined and max(refined) < max(best_S):
                best_S = refined
    
    # Fallback if CSP didn't find anything
    if best_S is None:
        # Greedy fallback
        S = [0]
        used_diffs = set()
        target = TARGET_SET.copy()
        
        for step in range(31):
            if time.time() - start_time > DEADLINE - 5:
                break
            if not target:
                break
            
            best_x = None
            best_gain = 0
            
            for x in range(1, 500):
                if x in S:
                    continue
                trial = sorted(S + [x])
                if not is_sidon(trial):
                    continue
                trial_diffs = get_diffs(trial)
                gain = len(trial_diffs & target)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            S.append(best_x)
            S.sort()
            used_diffs = get_diffs(S)
            target = TARGET_SET - used_diffs
        
        if covers_all(S) and is_sidon(S):
            best_S = S
        else:
            # Last resort: use a known construction
            # Golomb ruler-like: use powers of 3
            best_S = [0]
            for i in range(1, 6):
                best_S.append(3**i)
            # Verify and extend if needed
            while not covers_all(best_S) and len(best_S) < 32:
                for x in range(1, 1000):
                    if x in best_S:
                        continue
                    trial = sorted(best_S + [x])
                    if is_sidon(trial) and covers_all(trial):
                        best_S = trial
                        break
                else:
                    best_S.append(2 * max(best_S) + 1)

    return best_S

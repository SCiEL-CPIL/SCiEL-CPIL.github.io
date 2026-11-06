"""Construct a 4-AP-free subset of [1,512] maximizing cardinality."""
import time
import random

N = 512
TIME_LIMIT = 85  # seconds, safety margin before 90s kill

def _has_4ap_conflict(x, selected):
    """Check if adding x to selected would create a 4-AP."""
    # x could be the 1st, 2nd, 3rd, or 4th term
    # 1st: x, x+d, x+2d, x+3d all in set (x is smallest)
    # 4th: x-3d, x-2d, x-d, x all in set (x is largest)
    # 2nd: x-d, x, x+d, x+2d
    # 3rd: x-2d, x-d, x, x+d
    
    # x as 1st term: need x+d, x+2d, x+3d in selected
    # d >= 1, x+3d <= N
    for d in range(1, (N - x) // 3 + 1):
        if (x + d) in selected and (x + 2*d) in selected and (x + 3*d) in selected:
            return True
    
    # x as 4th term: need x-3d, x-2d, x-d in selected
    # d >= 1, x-3d >= 1
    for d in range(1, (x - 1) // 3 + 1):
        if (x - d) in selected and (x - 2*d) in selected and (x - 3*d) in selected:
            return True
    
    # x as 2nd term: need x-d, x+d, x+2d in selected
    # d >= 1, x-d >= 1, x+2d <= N
    max_d = min(x - 1, (N - x) // 2)
    for d in range(1, max_d + 1):
        if (x - d) in selected and (x + d) in selected and (x + 2*d) in selected:
            return True
    
    # x as 3rd term: need x-2d, x-d, x+d in selected
    # d >= 1, x-2d >= 1, x+d <= N
    max_d = min((x - 1) // 2, N - x)
    for d in range(1, max_d + 1):
        if (x - 2*d) in selected and (x - d) in selected and (x + d) in selected:
            return True
    
    return False

def _greedy_randomized(seed_val=0):
    """Greedy construction with randomization."""
    rng = random.Random(seed_val)
    selected = set()
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if not _has_4ap_conflict(x, selected):
            selected.add(x)
    
    return selected

def _greedy_structured():
    """Greedy construction using number-theoretic structure.
    
    Use numbers with small base-3 representation digits (0 or 1) mapped to [1,512].
    The classic construction: numbers whose base-3 expansion uses only 0s and 1s
    form a 3-AP-free set. For 4-AP-free, we need a more careful approach.
    
    Actually, let's use the greedy algorithm but with a good ordering.
    """
    selected = set()
    # Try adding in order of increasing "density" - prefer numbers that are
    # harder to form APs with
    for x in range(1, N + 1):
        if not _has_4ap_conflict(x, selected):
            selected.add(x)
    return selected

def _local_search_add(selected, rng, time_limit):
    """Try to add elements that were rejected, removing conflicting elements if beneficial."""
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Try each element not in selected
        for x in range(1, N + 1):
            if x in selected:
                continue
            # Find which APs would be violated
            conflicts = []
            _find_conflicts(x, selected, conflicts)
            # If removing the conflicting elements and adding x increases size
            # (i.e., we remove fewer than 1 element... well we add 1, remove k, net = 1-k)
            # Only beneficial if we remove 0 elements, which means no conflict
            # OR if we can remove some and the net is positive (impossible since we add 1 remove k>=1)
            # So only add if no conflict
            if not conflicts:
                selected.add(x)
                improved = True
    return selected

def _find_conflicts(x, selected, conflicts):
    """Find elements in selected that form a 4-AP with x."""
    # x as 1st: x+d, x+2d, x+3d
    for d in range(1, (N - x) // 3 + 1):
        a, b, c = x+d, x+2*d, x+3*d
        if a in selected and b in selected and c in selected:
            conflicts.extend([a, b, c])
    
    # x as 4th: x-3d, x-2d, x-d
    for d in range(1, (x - 1) // 3 + 1):
        a, b, c = x-3*d, x-2*d, x-d
        if a in selected and b in selected and c in selected:
            conflicts.extend([a, b, c])
    
    # x as 2nd: x-d, x+d, x+2d
    max_d = min(x - 1, (N - x) // 2)
    for d in range(1, max_d + 1):
        a, b, c = x-d, x+d, x+2*d
        if a in selected and b in selected and c in selected:
            conflicts.extend([a, b, c])
    
    # x as 3rd: x-2d, x-d, x+d
    max_d = min((x - 1) // 2, N - x)
    for d in range(1, max_d + 1):
        a, b, c = x-2*d, x-d, x+d
        if a in selected and b in selected and c in selected:
            conflicts.extend([a, b, c])

def _local_search_remove(selected, rng, time_limit):
    """Try removing elements to allow adding more elements elsewhere."""
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        for x in list(selected):
            # Try removing x, then see if we can add more
            selected.discard(x)
            added = 0
            for y in range(1, N + 1):
                if y not in selected and not _has_4ap_conflict(y, selected):
                    selected.add(y)
                    added += 1
            if added > 1:
                improved = True
            else:
                # Put x back if it didn't help
                if added <= 1:
                    selected.add(x)
                    # Remove the ones we added
                    for y in range(1, N + 1):
                        if y != x and y in selected:
                            # Check if y was added in this round
                            pass
                    # Actually this is getting complicated. Let's simplify.
                    break
    return selected

def _local_search_swap(selected, rng, time_limit):
    """Try swapping: remove one element, add one or more others."""
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        for x in list(selected):
            selected.discard(x)
            # Try adding elements
            added_count = 0
            added_elements = []
            for y in range(1, N + 1):
                if y not in selected and not _has_4ap_conflict(y, selected):
                    selected.add(y)
                    added_elements.append(y)
                    added_count += 1
            if added_count >= 2:
                improved = True
            else:
                # Revert
                for y in added_elements:
                    selected.discard(y)
                selected.add(x)
    return selected

def construct():
    start_time = time.time()
    
    # Phase 1: Multiple greedy constructions with different orderings
    best = set()
    
    # Standard greedy
    s1 = _greedy_structured()
    best = s1.copy()
    
    # Randomized greedy
    for seed in range(20):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        s = _greedy_randomized(seed)
        if len(s) > len(best):
            best = s.copy()
    
    # Phase 2: Local search - try to add elements
    remaining_time = TIME_LIMIT - (time.time() - start_time)
    if remaining_time > 5:
        best = _local_search_add(best, random.Random(42), remaining_time * 0.3)
    
    # Phase 3: Local search - swap
    remaining_time = TIME_LIMIT - (time.time() - start_time)
    if remaining_time > 5:
        best = _local_search_swap(best, random.Random(43), remaining_time * 0.5)
    
    # Phase 4: More randomized greedy + local search
    remaining_time = TIME_LIMIT - (time.time() - start_time)
    if remaining_time > 10:
        for seed in range(50, 200):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            s = _greedy_randomized(seed)
            if len(s) > len(best):
                best = s.copy()
                # Quick local search
                remaining = TIME_LIMIT - (time.time() - start_time)
                if remaining > 5:
                    best = _local_search_add(best, random.Random(seed), remaining * 0.3)
    
    # Final: try to add any remaining elements
    remaining_time = TIME_LIMIT - (time.time() - start_time)
    if remaining_time > 2:
        best = _local_search_add(best, random.Random(99), remaining_time * 0.5)
    
    return sorted(best)

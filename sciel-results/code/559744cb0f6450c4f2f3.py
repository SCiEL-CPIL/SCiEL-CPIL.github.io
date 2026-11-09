"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _check_divisibility(sum_set):
    """Check no two distinct sums divide each other using multiple-checking."""
    if len(sum_set) <= 1:
        return True
    sums = sorted(sum_set)
    max_val = sums[-1]
    for a in sums:
        if a <= 1:
            if len(sum_set) > 1:
                return False
            continue
        if a > max_val // 2:
            continue
        for k in range(2, max_val // a + 1):
            if a * k in sum_set:
                return False
    return True

def _can_add_fast(current_sums, x):
    """
    Fast incremental check: can x be added to a set whose non-empty subset sums are current_sums?
    Returns (True, new_sums) if valid, (False, None) otherwise.
    """
    if x in current_sums:
        return False, None
    
    new_sums = set(current_sums)
    new_sums.add(x)
    
    for s in current_sums:
        ns = s + x
        if ns in new_sums:
            return False, None
        new_sums.add(ns)
    
    # Check divisibility on the full new set
    if not _check_divisibility(new_sums):
        return False, None
    
    return True, new_sums

def _is_valid(elements):
    """Full validity check for a list of elements."""
    if len(elements) != len(set(elements)):
        return False
    if any(x < 1 or x > 256 for x in elements):
        return False
    if not elements:
        return True
    
    # Build subset sums incrementally
    current_sums = set()
    for x in elements:
        ok, new_sums = _can_add_fast(current_sums, x)
        if not ok:
            return False
        current_sums = new_sums
    
    return True

def _greedy_build(pool, deadline, current_sums=None):
    """Greedy: add elements from pool in order if valid. Returns (elements, sums)."""
    elements = []
    if current_sums is None:
        current_sums = set()
    
    for x in pool:
        if time.time() > deadline:
            break
        ok, new_sums = _can_add_fast(current_sums, x)
        if ok:
            elements.append(x)
            current_sums = new_sums
    
    return elements, current_sums

def _local_search_improve(elements, sums, deadline, max_size=30):
    """Try to improve by adding or swapping elements using fast checks."""
    current_elements = list(elements)
    current_sums = set(sums)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding a new element
        for c in range(1, 257):
            if time.time() > deadline:
                break
            if c in current_elements:
                continue
            ok, new_sums = _can_add_fast(current_sums, c)
            if ok:
                current_elements.append(c)
                current_sums = new_sums
                improved = True
                break
        
        # Try single swap: remove one element, add another
        if not improved and len(current_elements) > 0:
            for i in range(len(current_elements)):
                if time.time() > deadline:
                    break
                removed = current_elements[i]
                # Remove element i: need to recompute sums without it
                remaining = current_elements[:i] + current_elements[i+1:]
                if not remaining:
                    rem_sums = set()
                else:
                    rem_sums = set()
                    for x in remaining:
                        ok2, rem_sums = _can_add_fast(rem_sums, x)
                        if not ok2:
                            break
                
                for c in range(1, 257):
                    if time.time() > deadline:
                        break
                    if c in current_elements:
                        continue
                    ok, new_sums = _can_add_fast(rem_sums, c)
                    if ok:
                        new_elements = remaining + [c]
                        # Try to add one more after swap
                        added_more = False
                        for c2 in range(1, 257):
                            if time.time() > deadline:
                                break
                            if c2 in new_elements:
                                continue
                            ok2, new_sums2 = _can_add_fast(new_sums, c2)
                            if ok2:
                                current_elements = new_elements + [c2]
                                current_sums = new_sums2
                                improved = True
                                added_more = True
                                break
                        if not added_more:
                            current_elements = new_elements
                            current_sums = new_sums
                            improved = True
                        break
                if improved:
                    break
    
    return current_elements, current_sums

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    best_set = [31, 32, 34]
    best_sums = set()
    # Build sums for best_set
    for x in best_set:
        _, best_sums = _can_add_fast(best_sums, x)
    
    best_size = len(best_set)
    
    # Strategy 1: Greedy with various orderings
    pools = [
        list(range(256, 1, -1)),  # Descending
        list(range(2, 257)),      # Ascending
        [x for x in range(2, 257) if x % 2 != 0],  # Odds
        [x for x in range(2, 257) if x % 2 == 0],  # Evens
        list(range(128, 257)),    # High range
        list(range(64, 257)),     # Mid-high range
    ]
    
    rng = random.Random(42)
    for _ in range(30):
        p = list(range(2, 257))
        rng.shuffle(p)
        pools.append(p)
    
    for pool in pools:
        if time.time() > deadline:
            break
        elements, sums = _greedy_build(pool, deadline)
        if len(elements) > best_size:
            best_size = len(elements)
            best_set = elements
            best_sums = sums
            if best_size >= 30:
                break
    
    # Strategy 2: Local search improvement
    if time.time() < deadline:
        improved_set, improved_sums = _local_search_improve(best_set, best_sums, deadline)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
            best_sums = improved_sums
    
    # Strategy 3: More random greedy attempts with local search
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        pool = list(range(2, 257))
        rng.shuffle(pool)
        elements, sums = _greedy_build(pool, deadline - 5)
        if len(elements) > best_size:
            best_size = len(elements)
            best_set = elements
            best_sums = sums
        
        # Try local search on this result
        if time.time() < deadline - 5:
            improved_set, improved_sums = _local_search_improve(elements, sums, deadline - 2)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set
                best_sums = improved_sums
    
    # Strategy 4: Try specific starting points and extend
    if time.time() < deadline:
        starting_sets = [
            [128, 192, 224],
            [100, 150, 200],
            [64, 96, 128, 160],
            [80, 120, 160, 200, 240],
            [50, 80, 110, 140, 170, 200],
            [100, 140, 180, 220],
            [75, 125, 175, 225],
        ]
        for ss in starting_sets:
            if time.time() > deadline - 5:
                break
            # Validate starting set
            valid = True
            ss_sums = set()
            for x in ss:
                ok, ss_sums = _can_add_fast(ss_sums, x)
                if not ok:
                    valid = False
                    break
            if not valid:
                continue
            
            pool = [c for c in range(1, 257) if c not in ss]
            rng.shuffle(pool)
            elements, sums = _greedy_build(pool, deadline, current_sums=ss_sums)
            elements = ss + elements
            
            if len(elements) > best_size:
                best_size = len(elements)
                best_set = elements
                best_sums = sums
            
            # Local search
            if time.time() < deadline - 3:
                improved_set, improved_sums = _local_search_improve(elements, sums, deadline - 1)
                if len(improved_set) > best_size:
                    best_size = len(improved_set)
                    best_set = improved_set
                    best_sums = improved_sums
    
    # Final local search
    if time.time() < deadline - 5:
        improved_set, improved_sums = _local_search_improve(best_set, best_sums, deadline)
        if len(improved_set) > best_size:
            best_set = improved_set
    
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END

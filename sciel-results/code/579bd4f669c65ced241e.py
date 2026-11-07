"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from itertools import combinations

def _subset_sums_bitmask(elements):
    """Compute all non-empty subset sums using bitmask. Returns set of sums."""
    n = len(elements)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    return sums

def _check_divisibility(sums):
    """Check if any two distinct sums have divisibility relation."""
    sums_list = sorted(sums)
    max_s = sums_list[-1] if sums_list else 0
    for s in sums_list:
        if s == 0:
            continue
        for m in range(2 * s, max_s + 1, s):
            if m in sums:
                return False
    return True

def _is_valid(elements):
    """Check if the set satisfies all constraints."""
    if not elements:
        return True
    if len(elements) != len(set(elements)):
        return False
    for x in elements:
        if not isinstance(x, int) or isinstance(x, bool) or not (1 <= x <= 256):
            return False
    sums = _subset_sums_bitmask(elements)
    n = len(elements)
    if len(sums) != (1 << n) - 1:
        return False
    if not _check_divisibility(sums):
        return False
    return True

def _try_add_fast(elements, sums, candidate):
    """Try to add candidate. Returns (new_elements, new_sums) or (None, None).
    Uses the fact that new_sums = sums ∪ (sums + candidate) ∪ {candidate}.
    Collision check: if (sums + candidate) intersects sums, or candidate in sums."""
    if candidate in elements:
        return None, None
    
    # Check for collisions: s1 + candidate == s2 for s1, s2 in sums
    # i.e., s2 - candidate in sums for some s2 in sums
    # Also candidate itself must not be in sums (but sums are all >= min element, 
    # and candidate > all elements if we add in increasing order... not necessarily)
    
    shifted = set()
    for s in sums:
        ns = s + candidate
        if ns in sums:
            return None, None
        shifted.add(ns)
    
    if candidate in sums:
        return None, None
    
    new_sums = sums | shifted | {candidate}
    expected_size = (1 << (len(elements) + 1)) - 1
    if len(new_sums) != expected_size:
        return None, None
    
    if not _check_divisibility(new_sums):
        return None, None
    
    return elements + [candidate], new_sums

def _superincreasing_search(elements, sums, deadline, max_depth=12):
    """Try to extend elements using superincreasing-like ordering.
    At each step, try candidates in order of increasing value,
    preferring those that don't create divisibility issues."""
    if time.time() > deadline:
        return elements
    
    current_elements = list(elements)
    current_sums = set(sums)
    
    # Try to add more elements, starting from larger values first
    # to leave room for smaller ones
    candidates = list(range(256, 0, -1))
    
    while time.time() < deadline and len(current_elements) < max_depth:
        added = False
        for c in candidates:
            if c in current_elements:
                continue
            ne, ns = _try_add_fast(current_elements, current_sums, c)
            if ne is not None:
                current_elements = ne
                current_sums = ns
                added = True
                break
        if not added:
            break
    
    return current_elements

def _backtrack_with_pruning(elements, sums, target_size, deadline, rng, 
                            tried_set=None):
    """Depth-first backtracking with pruning to find a valid set of target_size."""
    if time.time() > deadline:
        return None
    if tried_set is None:
        tried_set = set()
    
    if len(elements) == target_size:
        return list(elements)
    
    # Generate candidates: prefer values that are "safe"
    # Sort candidates by how few divisibility conflicts they create
    candidates = list(range(256, 0, -1))
    rng.shuffle(candidates)
    
    # Prune: skip candidates that are too small or too large
    current_sum = sum(elements) if elements else 0
    
    for c in candidates:
        if c in elements:
            continue
        if c in tried_set:
            continue
        
        # Quick pruning: if c > 256 or c < 1, skip
        if c < 1 or c > 256:
            continue
        
        # Check if adding c would create immediate divisibility issues
        # with existing sums
        ne, ns = _try_add_fast(elements, sums, c)
        if ne is None:
            tried_set.add(c)
            continue
        
        # Recurse
        result = _backtrack_with_pruning(ne, ns, target_size, deadline, rng, tried_set)
        if result is not None:
            return result
    
    return None

def _local_search_improve(elements, deadline, rng):
    """Try to improve the set by swapping elements."""
    best = list(elements)
    best_len = len(elements)
    sums = _subset_sums_bitmask(elements) if elements else set()
    
    candidates_all = set(range(1, 257))
    iterations = 0
    max_iterations = 2000
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        if len(elements) == 0:
            c = rng.choice(list(candidates_all))
            ne, ns = _try_add_fast(elements, sums, c)
            if ne is not None:
                elements = ne
                sums = ns
            continue
        
        r = rng.random()
        if r < 0.6:
            # Try to add a random candidate
            avail = candidates_all - set(elements)
            if not avail:
                break
            c = rng.choice(list(avail))
            ne, ns = _try_add_fast(elements, sums, c)
            if ne is not None:
                elements = ne
                sums = ns
                if len(elements) > best_len:
                    best = list(elements)
                    best_len = len(elements)
        else:
            # Try a swap: remove one, try to add a different one
            if len(elements) < 2:
                continue
            rem_idx = rng.randrange(len(elements))
            removed = elements[rem_idx]
            new_elements = elements[:rem_idx] + elements[rem_idx+1:]
            new_sums = _subset_sums_bitmask(new_elements) if new_elements else set()
            
            # Try to add a candidate
            avail = candidates_all - set(new_elements)
            if avail:
                shuffled_avail = list(avail)
                rng.shuffle(shuffled_avail)
                found = False
                for c in shuffled_avail[:15]:
                    ne, ns = _try_add_fast(new_elements, new_sums, c)
                    if ne is not None:
                        elements = ne
                        sums = ns
                        if len(elements) > best_len:
                            best = list(elements)
                            best_len = len(elements)
                        found = True
                        break
                if not found:
                    # Revert
                    pass
    
    return best

def construct():
    deadline = time.time() + 85
    
    best = [31, 32, 34]
    best_len = len(best)
    
    rng = random.Random(42)
    
    # Strategy 1: Superincreasing-style greedy with multiple restarts
    for trial in range(150):
        if time.time() > deadline:
            break
        rng_t = random.Random(1000 + trial)
        
        # Start with 1-3 random elements in the upper range
        start_size = rng_t.randint(1, 3)
        start = rng_t.sample(range(100, 257), start_size)
        
        if not _is_valid(start):
            continue
        
        sums = _subset_sums_bitmask(start)
        result = _superincreasing_search(start, sums, deadline, max_depth=10)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Strategy 2: Backtracking from small bases
    for base_size in range(1, 5):
        if time.time() > deadline:
            break
        for trial in range(30):
            if time.time() > deadline:
                break
            rng_b = random.Random(7000 + base_size * 100 + trial)
            base = rng_b.sample(range(1, 257), base_size)
            if not _is_valid(base):
                continue
            base_sums = _subset_sums_bitmask(base)
            
            # Try to extend to various target sizes
            for target in range(base_size + 1, min(base_size + 5, 12)):
                if time.time() > deadline:
                    break
                result = _backtrack_with_pruning(base, base_sums, target, deadline, rng_b)
                if result is not None and len(result) > best_len:
                    best = result
                    best_len = len(result)
                    break
    
    # Strategy 3: Local search improvement from best found
    if time.time() < deadline - 10:
        improved = _local_search_improve(best, deadline, rng)
        if len(improved) > best_len:
            best = improved
            best_len = len(improved)
    
    # Final validation
    if not _is_valid(best):
        best = [31, 32, 34]
    
    return [int(x) for x in best]
# EVOLVE-BLOCK-END

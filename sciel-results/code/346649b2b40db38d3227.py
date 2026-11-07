"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _get_sums(elements):
    """Get all non-empty subset sums as a set."""
    n = len(elements)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    return sums

def _is_valid(elements):
    """Check if the set satisfies all constraints."""
    if not elements:
        return True
    # Check distinct
    if len(elements) != len(set(elements)):
        return False
    # Check range
    for x in elements:
        if not (1 <= x <= 256):
            return False
    sums = _get_sums(elements)
    # Check no divisibility
    sums_list = sorted(sums)
    for i in range(len(sums_list)):
        for j in range(i + 1, len(sums_list)):
            if sums_list[j] % sums_list[i] == 0:
                return False
    return True

def _check_addition(current_sums, new_elem):
    """Check if adding new_elem to a set with current_sums maintains validity.
    Returns new_sums if valid, None otherwise."""
    # New sums: current_sums ∪ {s + new_elem for s in current_sums} ∪ {new_elem}
    new_sums = set(current_sums)
    new_sums.add(new_elem)
    for s in current_sums:
        new_sums.add(s + new_elem)
    
    # Check distinct (automatically satisfied since we're using a set, but we need 
    # to ensure no two different subsets give same sum - this is guaranteed if 
    # all elements are positive and we're just extending)
    # Actually we need to verify: if current set has unique subset sums, and we add
    # new_elem, the new sums are old_sums and old_sums + new_elem. These are disjoint
    # iff new_elem > max(old_sums) ... no, that's not right.
    # Actually: old sums are in [min_elem, sum_old]. New sums are in [min_elem, sum_old + new_elem].
    # The overlap region is [new_elem, sum_old]. We need old_sums and (old_sums + new_elem) to be disjoint.
    # old_sums ∩ (old_sums + new_elem) = {s : s in old_sums AND s - new_elem in old_sums}
    # So we need: for all s in old_sums, s - new_elem not in old_sums (when s - new_elem > 0)
    
    # Check disjointness of old_sums and shifted sums
    for s in current_sums:
        if s + new_elem in current_sums:
            return None  # duplicate sum
    
    # Check no divisibility among all new_sums
    sums_list = sorted(new_sums)
    for i in range(len(sums_list)):
        for j in range(i + 1, len(sums_list)):
            if sums_list[j] % sums_list[i] == 0:
                return None
    return new_sums

def _greedy_grow(start_elements, deadline, rng):
    """Try to grow the set greedily."""
    elements = list(start_elements)
    sums = _get_sums(elements)
    
    # Try adding elements from 256 down to 1
    candidates = list(range(256, 0, -1))
    rng.shuffle(candidates)
    
    while time.time() < deadline:
        added = False
        for c in candidates:
            if c in elements:
                continue
            new_sums = _check_addition(sums, c)
            if new_sums is not None:
                elements.append(c)
                sums = new_sums
                candidates.remove(c)
                added = True
                break
        if not added:
            break
    
    return elements

def construct():
    deadline = time.time() + 80  # 80 second budget, 10s margin
    
    best = [31, 32, 34]
    best_len = len(best)
    
    rng = random.Random(42)
    
    # Try multiple strategies
    # Strategy 1: Start with large numbers
    for start_size in range(1, 8):
        if time.time() > deadline:
            break
        for trial in range(50):
            if time.time() > deadline:
                break
            # Start with random large numbers
            start = rng.sample(range(100, 257), start_size)
            if not _is_valid(start):
                continue
            result = _greedy_grow(start, deadline, rng)
            if len(result) > best_len:
                best = result
                best_len = len(result)
    
    # Strategy 2: Start with specific patterns
    # Try consecutive large numbers
    for base in range(150, 250):
        if time.time() > deadline:
            break
        start = list(range(base, min(base + 5, 257)))
        if len(start) < 2:
            continue
        if not _is_valid(start):
            continue
        result = _greedy_grow(start, deadline, rng)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Strategy 3: Pure greedy from scratch with different orderings
    for order_seed in range(20):
        if time.time() > deadline:
            break
        rng2 = random.Random(order_seed)
        elements = []
        sums = set()
        candidates = list(range(256, 0, -1))
        rng2.shuffle(candidates)
        
        for c in candidates:
            if time.time() > deadline:
                break
            new_sums = _check_addition(sums, c)
            if new_sums is not None:
                elements.append(c)
                sums = new_sums
        
        if len(elements) > best_len:
            best = elements
            best_len = len(elements)
    
    return best
# EVOLVE-BLOCK-END

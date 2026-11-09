import time
import random
import sys
from collections import defaultdict

START_TIME = time.time()
DEADLINE = 55.0  # Safety margin before 60s kill

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def try_add(x, sums, current_size):
    """Try adding element x to the set represented by sums (list of Python ints as bitsets).
    Returns new_sums if valid, else None."""
    new_sums = list(sums)
    for k in range(current_size, -1, -1):
        shifted = sums[k] << x
        candidate = new_sums[k + 1] | shifted
        # Check cross-cardinality collision
        for m in range(current_size + 2):
            if m != k + 1 and (candidate & new_sums[m]):
                return None
        new_sums[k + 1] = candidate
    return new_sums

def compute_sums(elements):
    """Compute the sum bitsets for a set of elements."""
    n = len(elements)
    sums = [0] * (n + 1)
    sums[0] = 1  # empty subset sum 0
    for i, x in enumerate(elements):
        for k in range(i, -1, -1):
            sums[k + 1] |= sums[k] << x
    return sums

def is_valid_set(elements):
    """Check if a set of elements satisfies the cross-cardinality constraint."""
    sums = compute_sums(elements)
    n = len(elements)
    for k in range(n + 1):
        for m in range(k + 1, n + 1):
            if sums[k] & sums[m]:
                return False
    return True

def greedy_construct(order):
    """Greedy: iterate through 'order', add each element if valid.
    Then do a second pass over all remaining candidates."""
    sums = [0] * (len(order) + 1)
    sums[0] = 1  # empty subset sum 0
    current_size = 0
    result = []
    used = set()
    
    # First pass
    for x in order:
        if is_done():
            break
        new_sums = try_add(x, sums, current_size)
        if new_sums is not None:
            sums = new_sums
            current_size += 1
            result.append(x)
            used.add(x)
    
    # Second pass: try all remaining candidates
    remaining = [x for x in order if x not in used]
    for x in remaining:
        if is_done():
            break
        new_sums = try_add(x, sums, current_size)
        if new_sums is not None:
            sums = new_sums
            current_size += 1
            result.append(x)
            used.add(x)
    
    return result

def local_search_improve(elements, all_candidates, max_iter=500):
    """Local search: try swapping elements to improve the set.
    Returns improved set if found."""
    current = list(elements)
    current_set = set(current)
    n = len(current)
    best = list(current)
    best_len = n
    
    for iteration in range(max_iter):
        if is_done():
            break
        
        improved = False
        
        # Try adding a new element
        for x in all_candidates:
            if x in current_set:
                continue
            new_elements = current + [x]
            if is_valid_set(new_elements):
                current = new_elements
                current_set = set(current)
                best = list(current)
                best_len = len(current)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping: remove one, add one
        for i in range(len(current)):
            if is_done():
                break
            for x in all_candidates:
                if x in current_set:
                    continue
                # Try replacing current[i] with x
                new_elements = [y for j, y in enumerate(current) if j != i] + [x]
                if is_valid_set(new_elements):
                    # Check if this actually improves (same size but different composition)
                    # For now, accept any valid swap to explore
                    current = new_elements
                    current_set = set(current)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best

def construct():
    best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # Powers of 2 baseline (9 elements)
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Increasing order
    strategies.append(list(range(1, 257)))
    
    # Strategy 2: Decreasing order
    strategies.append(list(range(256, 0, -1)))
    
    # Strategy 3: Random orders (more of them)
    rng = random.Random(42)
    for _ in range(30):
        order = list(range(1, 257))
        rng.shuffle(order)
        strategies.append(order)
    
    # Strategy 4: Sorted by value, but with some structure
    # Try multiples of small primes first
    for prime in [2, 3, 5, 7, 11]:
        order = [x for x in range(1, 257) if x % prime == 0]
        order += [x for x in range(1, 257) if x % prime != 0]
        strategies.append(order)
    
    # Strategy 5: Large elements first (to leave room for small ones later)
    strategies.append(sorted(range(1, 257), key=lambda x: -x))
    
    # Strategy 6: Interleaved small and large
    small = list(range(1, 65))
    large = list(range(193, 257))
    interleaved = []
    si, li = 0, 0
    while si < len(small) or li < len(large):
        if si < len(small):
            interleaved.append(small[si])
            si += 1
        if li < len(large):
            interleaved.append(large[li])
            li += 1
    strategies.append(interleaved)
    
    all_candidates = list(range(1, 257))
    
    for order in strategies:
        if is_done():
            break
        result = greedy_construct(order)
        if len(result) > len(best_set):
            best_set = result
    
    # Local search improvement on the best set found
    if not is_done():
        improved = local_search_improve(best_set, all_candidates, max_iter=300)
        if len(improved) > len(best_set):
            best_set = improved
    
    # Ensure Python ints in a list
    return [int(x) for x in best_set]

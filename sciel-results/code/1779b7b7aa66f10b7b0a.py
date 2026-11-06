"""Improved Sidon set construction for [1,284] containing {1,2,5,11}."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def _is_valid_sidon(s):
    """Check if sorted list s is a valid Sidon set (all pairwise sums distinct)."""
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            v = s[i] + s[j]
            if v in sums:
                return False
            sums.add(v)
    return True

def _get_sums(s):
    """Get all pairwise sums (i <= j) of sorted list s."""
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            sums.add(s[i] + s[j])
    return sums

def _can_add(s, sums, x):
    """Check if x can be added to Sidon set s with existing sums."""
    for y in s:
        if x + y in sums:
            return False
    if 2 * x in sums:
        return False
    return True

def _add_to_sums(sums, s, x):
    """Add x to sums set, assuming compatibility."""
    for y in s:
        sums.add(x + y)
    sums.add(2 * x)

def _greedy_construct(start_set, order):
    """Greedy construction using a given order of candidates."""
    s = list(start_set)
    sums = _get_sums(s)
    for x in order:
        if x in s:
            continue
        if _can_add(s, sums, x):
            s.append(x)
            _add_to_sums(sums, s, x)
    s.sort()
    return s

def _local_search(s, deadline, rng):
    """Local search: try to add elements, and try swaps to enable additions."""
    s = sorted(s)
    sums = _get_sums(s)
    best = list(s)
    
    # Phase 1: Try to add any missing element
    changed = True
    while changed:
        changed = False
        for x in range(1, 285):
            if x in s:
                continue
            if _can_add(s, sums, x):
                s.append(x)
                _add_to_sums(sums, s, x)
                s.sort()
                best = list(s)
                changed = True
                if time.time() > deadline:
                    return best
    
    # Phase 2: Try single swaps (remove one, try to add two)
    for _ in range(2000):
        if time.time() > deadline:
            return best
        if len(s) < len(best):
            s = list(best)
            sums = _get_sums(s)
        
        # Try removing one element and adding two others
        if len(s) < 2:
            break
        i = rng.randrange(len(s))
        removed = s[i]
        s_temp = s[:i] + s[i+1:]
        sums_temp = _get_sums(s_temp)
        
        # Try to add two elements
        added = 0
        new_elements = []
        for x in range(1, 285):
            if x in s_temp or x == removed:
                continue
            if _can_add(s_temp, sums_temp, x):
                s_temp.append(x)
                _add_to_sums(sums_temp, s_temp, x)
                added += 1
                new_elements.append(x)
                if added == 2:
                    break
        
        if added >= 1 and len(s_temp) > len(best):
            s = s_temp
            sums = sums_temp
            s.sort()
            best = list(s)
    
    # Phase 3: Try to add more elements after swaps
    changed = True
    while changed:
        changed = False
        for x in range(1, 285):
            if x in s:
                continue
            if _can_add(s, sums, x):
                s.append(x)
                _add_to_sums(sums, s, x)
                s.sort()
                best = list(s)
                changed = True
                if time.time() > deadline:
                    return best
    
    return best

def _random_order(start_set, rng):
    """Generate a random order of candidates excluding start_set."""
    candidates = [x for x in range(1, 285) if x not in start_set]
    rng.shuffle(candidates)
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    start_set = [1, 2, 5, 11]
    best = list(start_set)
    best_size = len(start_set)
    
    rng = random.Random(42)
    
    # First, do a deterministic greedy pass
    order = list(range(12, 285))
    s = _greedy_construct(start_set, order)
    if len(s) > best_size:
        best = list(s)
        best_size = len(s)
    
    # Then, do randomized restarts with different orders
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Vary the random seed for diversity
        rng_restarts = random.Random(r * 12345 + 67890)
        
        # Different strategies for ordering
        strategy = r % 4
        
        if strategy == 0:
            # Pure random
            order = _random_order(start_set, rng_restarts)
        elif strategy == 1:
            # Random with bias toward smaller numbers
            order = _random_order(start_set, rng_restarts)
            order.sort(key=lambda x: (rng_restarts.random(), x))
        elif strategy == 2:
            # Random with bias toward larger numbers
            order = _random_order(start_set, rng_restarts)
            order.sort(key=lambda x: (rng_restarts.random(), -x))
        else:
            # Interleaved small/large random
            small = [x for x in range(12, 150) if x not in start_set]
            large = [x for x in range(150, 285) if x not in start_set]
            rng_restarts.shuffle(small)
            rng_restarts.shuffle(large)
            order = []
            si, li = 0, 0
            while si < len(small) or li < len(large):
                if si < len(small):
                    order.append(small[si])
                    si += 1
                if li < len(large):
                    order.append(large[li])
                    li += 1
        
        s = _greedy_construct(start_set, order)
        
        # Local search on the result
        s = _local_search(s, deadline, rng_restarts)
        
        if len(s) > best_size:
            best = list(s)
            best_size = len(s)
        
        # Also try local search on the deterministic greedy result
        if r == 0:
            s_det = _greedy_construct(start_set, list(range(12, 285)))
            s_det = _local_search(s_det, deadline, rng_restarts)
            if len(s_det) > best_size:
                best = list(s_det)
                best_size = len(s_det)
    
    # Final local search on the best found
    if time.time() < deadline - 5:
        best = _local_search(best, deadline, rng)
    
    # Verify the result is valid
    if not _is_valid_sidon(best):
        # Fallback to a known valid set
        best = [1, 2, 5, 11]
        # Try to rebuild
        sums = _get_sums(best)
        for x in range(12, 285):
            if _can_add(best, sums, x):
                best.append(x)
                _add_to_sums(sums, best, x)
        best.sort()
    
    return best
# EVOLVE-BLOCK-END

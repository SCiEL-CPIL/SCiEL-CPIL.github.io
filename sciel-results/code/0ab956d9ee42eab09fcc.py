"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from itertools import combinations

def _subset_sums_distinct(seq):
    """Check that all 2^n subset sums are distinct using a set."""
    n = len(seq)
    sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += seq[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _min_next_element(current_seq, max_val):
    """Find the minimum next element that keeps all subset sums distinct."""
    if not current_seq:
        return 1
    current_sums = set()
    n = len(current_seq)
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += current_seq[i]
        current_sums.add(s)
    
    # Try values from sum(current_seq) + 1 upward (since new element must be > any previous element for sorted order, 
    # but actually we just need distinctness)
    # Actually, the new element can be anything as long as all new subset sums are distinct
    # New subset sums are: old_sums and old_sums + new_elem
    # So we need: (old_sums) ∩ (old_sums + new_elem) = empty
    # i.e., for all a, b in old_sums: a != b + new_elem, i.e., new_elem != a - b for any a,b in old_sums
    # So new_elem must not be in {a - b : a, b in old_sums, a > b}
    
    forbidden = set()
    for a in current_sums:
        for b in current_sums:
            if a > b:
                forbidden.add(a - b)
    
    # Also new_elem must be positive and not already in current_seq (for distinctness of the sequence)
    existing = set(current_seq)
    
    # Find smallest positive integer not in forbidden and not in existing
    candidate = 1
    while candidate in forbidden or candidate in existing:
        candidate += 1
        if candidate > max_val:
            return None
    return candidate

def _greedy_construct(n, max_val, deadline):
    """Greedy construction: at each step, pick the smallest element that keeps subset sums distinct."""
    seq = []
    for i in range(n):
        if time.time() > deadline:
            return None
        nxt = _min_next_element(seq, max_val)
        if nxt is None or nxt > max_val:
            return None
        seq.append(nxt)
    return seq

def _verify(seq, n, max_val):
    """Verify the solution is valid."""
    if len(seq) != n:
        return False
    if any(x <= 0 for x in seq):
        return False
    if any(x > max_val for x in seq):
        return False
    if len(set(seq)) != n:
        return False
    if not _subset_sums_distinct(seq):
        return False
    return True

def construct():
    """Construct exactly 12 distinct positive integers <= 1048576 with all subset sums distinct.
    Minimize the largest integer. Returns the list of 12 integers."""
    n = 12
    max_val = 1048576
    deadline = time.time() + 80  # Safety margin
    
    # Start with the known superincreasing solution as fallback
    fallback = [1 << i for i in range(n)]
    best = fallback
    best_max = max(fallback)
    
    # Try greedy construction
    result = _greedy_construct(n, max_val, deadline)
    if result is not None and _verify(result, n, max_val):
        if max(result) < best_max:
            best = result
            best_max = max(result)
    
    # Try to improve by local search: try replacing elements with smaller values
    # For each position, try to reduce the element
    for attempt in range(100):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n - 1, -1, -1):
            if time.time() > deadline:
                break
            current_val = best[i]
            # Try to reduce best[i]
            for candidate in range(1, current_val):
                if time.time() > deadline:
                    break
                new_seq = best.copy()
                new_seq[i] = candidate
                # Check distinctness of sequence
                if len(set(new_seq)) != n:
                    continue
                if _verify(new_seq, n, max_val):
                    best = new_seq
                    best_max = max(best)
                    improved = True
                    break
        if not improved:
            break
    
    return best

# EVOLVE-BLOCK-END

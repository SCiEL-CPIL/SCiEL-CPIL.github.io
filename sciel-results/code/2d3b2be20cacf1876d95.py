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

def _all_subset_sums(seq):
    """Compute all 2^n subset sums."""
    n = len(seq)
    sums = []
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += seq[i]
        sums.append(s)
    return sums

def _greedy_construct(n, max_val, deadline):
    """Greedy construction: at each step, pick the smallest element that keeps subset sums distinct."""
    seq = []
    for i in range(n):
        if time.time() > deadline:
            return None
        if not seq:
            seq.append(1)
            continue
        current_sums = _all_subset_sums(seq)
        current_sums_set = set(current_sums)
        
        # Find forbidden values: a - b for a, b in current_sums where a > b
        forbidden = set()
        for a in current_sums:
            for b in current_sums:
                if a > b:
                    forbidden.add(a - b)
        
        existing = set(seq)
        candidate = 1
        while candidate in forbidden or candidate in existing:
            candidate += 1
            if candidate > max_val:
                return None
        seq.append(candidate)
    return seq

def _verify_fast(seq, n, max_val):
    """Verify the solution is valid - fast version."""
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

def _try_reduce_element(seq, idx, max_val, deadline):
    """Try to reduce seq[idx] to the smallest valid value. Returns new seq or None."""
    n = len(seq)
    old_val = seq[idx]
    
    # Compute subset sums without element idx
    other_elements = [seq[i] for i in range(n) if i != idx]
    other_sums = _all_subset_sums(other_elements)
    other_sums_set = set(other_sums)
    
    # For a new value v at position idx:
    # All subset sums = {s for s in other_sums} ∪ {s + v for s in other_sums}
    # We need all 2^n sums to be distinct, which means:
    # 1. other_sums are all distinct (already true)
    # 2. other_sums + v are all distinct (true since other_sums are)
    # 3. other_sums ∩ (other_sums + v) = empty, i.e., v not in {a - b : a,b in other_sums, a > b}
    
    forbidden = set()
    for a in other_sums:
        for b in other_sums:
            if a > b:
                forbidden.add(a - b)
    
    existing = set(seq)
    existing.discard(old_val)
    
    # Try values from 1 upward, skipping forbidden and existing
    for v in range(1, old_val):
        if time.time() > deadline:
            return None
        if v in forbidden or v in existing:
            continue
        # Check that v is not in forbidden (already checked)
        # Verify full distinctness
        new_seq = seq.copy()
        new_seq[idx] = v
        if _subset_sums_distinct(new_seq):
            return new_seq
    return None

def _local_search_improve(seq, n, max_val, deadline):
    """Iteratively try to reduce each element from largest to smallest."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Process from largest index to smallest (try to reduce larger elements first)
        for idx in range(n - 1, -1, -1):
            if time.time() > deadline:
                return seq
            new_seq = _try_reduce_element(seq, idx, max_val, deadline)
            if new_seq is not None:
                seq = new_seq
                improved = True
    return seq

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
    if result is not None and _verify_fast(result, n, max_val):
        if max(result) < best_max:
            best = result
            best_max = max(result)
    
    # Local search to improve
    best = _local_search_improve(best, n, max_val, deadline)
    
    return best

# EVOLVE-BLOCK-END

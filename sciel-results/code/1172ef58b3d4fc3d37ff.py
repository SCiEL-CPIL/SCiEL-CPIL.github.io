"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _compute_subset_sums(seq):
    """Compute all 2^n subset sums efficiently using DP."""
    n = len(seq)
    sums = [0]
    for x in seq:
        sums = sums + [s + x for s in sums]
    return sums

def _subset_sums_distinct(seq):
    """Check that all 2^n subset sums are distinct using a set."""
    n = len(seq)
    sums = _compute_subset_sums(seq)
    return len(set(sums)) == (1 << n)

def _min_next_element(current_seq, max_val):
    """Find the minimum next element that keeps all subset sums distinct."""
    if not current_seq:
        return 1
    current_sums = set(_compute_subset_sums(current_seq))
    
    # New subset sums are: old_sums and old_sums + new_elem
    # We need: (old_sums) ∩ (old_sums + new_elem) = empty
    # i.e., new_elem != a - b for any a, b in old_sums where a > b
    
    forbidden = set()
    for a in current_sums:
        for b in current_sums:
            if a > b:
                forbidden.add(a - b)
    
    existing = set(current_seq)
    
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

def _local_search_reduce(seq, n, max_val, deadline):
    """Try to reduce elements one by one, starting from the largest."""
    best = list(seq)
    best_max = max(best)
    
    while True:
        if time.time() > deadline:
            break
        improved = False
        # Try reducing from largest to smallest
        for i in range(n - 1, -1, -1):
            if time.time() > deadline:
                break
            current_val = best[i]
            # Binary search for the smallest valid value
            lo, hi = 1, current_val - 1
            found = None
            while lo <= hi:
                if time.time() > deadline:
                    break
                mid = (lo + hi) // 2
                new_seq = best.copy()
                new_seq[i] = mid
                if len(set(new_seq)) == n and _subset_sums_distinct(new_seq):
                    found = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            if found is not None:
                best[i] = found
                best_max = max(best)
                improved = True
        if not improved:
            break
    
    return best, best_max

def _randomized_restart(n, max_val, deadline, seed):
    """Try a randomized construction approach."""
    rng = random.Random(seed)
    best = None
    best_max = max_val + 1
    
    # Start with greedy, then perturb
    base = _greedy_construct(n, max_val, deadline)
    if base is None:
        return None, max_val
    
    candidates = [base]
    
    # Add some random perturbations
    for _ in range(5):
        if time.time() > deadline:
            break
        cand = list(base)
        # Randomly increase some elements to allow reduction of others
        for _ in range(rng.randint(1, 3)):
            idx = rng.randrange(n)
            if cand[idx] < max_val:
                cand[idx] = min(max_val, cand[idx] + rng.randint(1, max(cand) // 4))
        if len(set(cand)) == n and _subset_sums_distinct(cand):
            candidates.append(cand)
    
    for cand in candidates:
        if time.time() > deadline:
            break
        improved, imp_max = _local_search_reduce(cand, n, max_val, deadline)
        if max(improved) < best_max:
            best = improved
            best_max = max(improved)
    
    if best is None:
        best = base
        best_max = max(base)
    
    return best, best_max

def construct():
    """Construct exactly 12 distinct positive integers <= 1048576 with all subset sums distinct.
    Minimize the largest integer. Returns the list of 12 integers."""
    n = 12
    max_val = 1048576
    deadline = time.time() + 85  # Safety margin (90s limit)
    
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
    
    # Local search on the greedy result
    if time.time() < deadline:
        improved, imp_max = _local_search_reduce(best, n, max_val, deadline)
        if imp_max < best_max:
            best = improved
            best_max = imp_max
    
    # Try randomized restarts if time remains
    if time.time() < deadline - 5:
        for seed in range(20):
            if time.time() > deadline:
                break
            cand, cand_max = _randomized_restart(n, max_val, deadline, seed)
            if cand is not None and cand_max < best_max:
                best = cand
                best_max = cand_max
    
    return best

# EVOLVE-BLOCK-END

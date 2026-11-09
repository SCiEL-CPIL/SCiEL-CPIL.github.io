"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 5000
# Hard deadline: return best result before this many seconds have elapsed
_DEADLINE = 320.0

# EVOLVE-BLOCK-START

def _primes_upto(limit):
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(sieve[p*p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]


def _residue_counts(points, p):
    """Return list of counts for each residue class mod p."""
    # Use numpy for speed
    arr = np.array(points, dtype=np.int64) % p
    return np.bincount(arr, minlength=p)


def _is_admissible(points, primes):
    """Check if points form an admissible tuple for the given primes."""
    n = len(points)
    if n == 0:
        return True
    for p in primes:
        if p > n:
            break
        counts = _residue_counts(points, p)
        if np.all(counts > 0):
            return False
    return True


def _greedy_construct():
    """
    Start with all integers in [0, N] and remove elements to make the set admissible.
    For each prime p <= len(set), if all residue classes mod p are covered,
    remove the least-populated residue class.
    """
    points = list(range(N + 1))
    # We need primes up to N initially
    all_primes = _primes_upto(N)
    
    # We'll iterate: for each prime p <= len(points), check coverage
    # If covered, remove the least-populated class
    # Repeat until admissible
    changed = True
    while changed:
        changed = False
        n = len(points)
        # Only need to check primes up to n
        for p in all_primes:
            if p > n:
                break
            counts = _residue_counts(points, p)
            if np.all(counts > 0):
                # Find the residue class with minimum count
                min_count = np.min(counts)
                # Get all residue classes with this min count
                min_classes = np.where(counts == min_count)[0]
                # Remove all elements in the least-populated class(es)
                # Choose the class with the fewest elements; if tie, pick first
                # Actually, remove ALL elements in the min-count class(es)
                # But we only need to remove enough to leave one class empty
                # So remove all elements in ONE min-count class
                rc = int(min_classes[0])
                points = [v for v in points if v % p != rc]
                changed = True
                break
    return points


def _local_search_add(points, primes, time_limit):
    """
    Try to add back elements that were removed, checking admissibility.
    Start from the greedy result and try to add elements from [0, N] that are not in points.
    """
    points_set = set(points)
    candidates = [v for v in range(N + 1) if v not in points_set]
    # Shuffle candidates to avoid bias
    import random
    random.shuffle(candidates)
    
    current = list(points)
    current_set = points_set
    best = current
    
    start_time = time.time()
    for v in candidates:
        if time.time() - start_time > time_limit:
            break
        # Try adding v
        new_set = current_set | {v}
        new_list = current + [v]
        # Check admissibility
        admissible = True
        n = len(new_list)
        for p in primes:
            if p > n:
                break
            counts = _residue_counts(new_list, p)
            if np.all(counts > 0):
                admissible = False
                break
        if admissible:
            current = new_list
            current_set = new_set
            best = current
    
    return best


def _multi_start_greedy(seed_offset=0):
    """
    Alternative greedy: start from a random subset and grow.
    """
    import random
    rng = random.Random(seed_offset)
    
    # Start with a random subset of size ~ N/2
    init_size = N // 2
    points = set(rng.sample(range(N + 1), init_size))
    
    all_primes = _primes_upto(N)
    
    # First, make it admissible by removing
    changed = True
    while changed:
        changed = False
        n = len(points)
        for p in all_primes:
            if p > n:
                break
            arr = np.array(list(points), dtype=np.int64) % p
            counts = np.bincount(arr, minlength=p)
            if np.all(counts > 0):
                min_count = np.min(counts)
                min_classes = np.where(counts == min_count)[0]
                rc = int(min_classes[0])
                points = {v for v in points if v % p != rc}
                changed = True
                break
    
    # Now try to add back
    candidates = [v for v in range(N + 1) if v not in points]
    rng.shuffle(candidates)
    
    for v in candidates:
        new_set = points | {v}
        n = len(new_set)
        admissible = True
        for p in all_primes:
            if p > n:
                break
            arr = np.array(list(new_set), dtype=np.int64) % p
            counts = np.bincount(arr, minlength=p)
            if np.all(counts > 0):
                admissible = False
                break
        if admissible:
            points = new_set
    
    return list(points)


def construct():
    start_time = time.time()
    
    # Primes up to N
    all_primes = _primes_upto(N)
    
    # Phase 1: Greedy construction
    best = _greedy_construct()
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE - 30:
        return best
    
    # Phase 2: Local search to add back elements
    time_left = _DEADLINE - 30 - (time.time() - start_time)
    if time_left > 5:
        best = _local_search_add(best, all_primes, time_left)
    
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE - 20:
        return best
    
    # Phase 3: Try multi-start greedy with different seeds
    time_left = _DEADLINE - 20 - (time.time() - start_time)
    if time_left > 10:
        for seed in range(5):
            if time.time() - start_time > _DEADLINE - 10:
                break
            candidate = _multi_start_greedy(seed + 1)
            if len(candidate) > len(best):
                best = candidate
    
    return best

# EVOLVE-BLOCK-END

"""Stronger solver for the fixed-diameter admissible-tuple maximization task."""
import time
import math
import random
import sys

N = 5000
DEADLINE = 310  # seconds, safety margin before 330s kill

def _primes_up_to(limit):
    """Return list of primes <= limit via sieve."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:limit + 1:step] = bytearray(len(range(start, limit + 1, step)))
    return [p for p in range(2, limit + 1) if sieve[p]]

# Precompute primes up to N+1 (max possible tuple size is N+1)
PRIMES = _primes_up_to(N + 1)
PRIME_SET = set(PRIMES)

def _is_admissible(points, primes=None):
    """Check if points is admissible: for every prime p <= len(points), some residue class mod p is missed."""
    if primes is None:
        primes = PRIMES
    L = len(points)
    if L <= 1:
        return True
    # Build residue coverage for each prime
    for p in primes:
        if p > L:
            break
        seen = bytearray(p)
        for v in points:
            seen[v % p] = 1
        if all(seen):
            return False
    return True

def _greedy_remove():
    """Start with all [0..N], remove least-populated residue class for each prime in order."""
    points = list(range(N + 1))
    for p in PRIMES:
        if p > len(points):
            break
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        if min(counts) > 0:
            min_val = min(counts)
            min_idx = counts.index(min_val)
            points = [v for v in points if v % p != min_idx]
    return points

def _fast_insert_check(points_set, points_list, cand, primes):
    """
    Check if adding cand to current admissible set keeps it admissible.
    points_set: set of current points (for O(1) membership)
    points_list: list of current points
    Returns True if cand can be added.
    """
    L_new = len(points_list) + 1
    
    # For each prime p <= L_new, check if cand would fill the last missing class
    for p in primes:
        if p > L_new:
            break
        # Count how many residue classes are already covered
        # and whether cand's class is already covered
        # A prime is currently safe (admissible) iff some class is missed.
        # Adding cand makes it unsafe iff:
        #   - p was previously safe (had exactly 1 missed class), AND
        #   - cand % p is that missed class
        # OR
        #   - p was newly relevant (p == L_new, wasn't checked before), AND
        #   - after adding cand, all p classes are covered
        
        # Efficient approach: for each p, check if cand's residue class is the unique missing one.
        # We need to know the current residue counts for p.
        # But computing that from scratch each time is O(L * pi(N)) which is too slow for many insertions.
        
        # Alternative: precompute and maintain residue counts? That's complex.
        # For now, do a direct check but with early exit.
        
        seen = bytearray(p)
        cand_r = cand % p
        seen[cand_r] = 1  # cand covers this class
        for v in points_list:
            seen[v % p] = 1
        if all(seen):
            return False
    return True

def _try_add_points(points_list, time_limit):
    """Try to add points to an admissible set using randomized greedy insertion."""
    in_set = set(points_list)
    points = points_list[:]
    
    # Candidates: all numbers not in points
    candidates = [i for i in range(N + 1) if i not in in_set]
    random.shuffle(candidates)
    
    # We'll maintain residue count state for speed
    # For each prime p <= N+1, maintain:
    #   count[p][r] = number of points in points with point % p == r
    #   missing[p] = number of residue classes with count 0
    
    # Initialize
    max_p = N + 1
    # Only track primes <= len(points) + 1 initially, but len grows
    # Let's precompute for all primes up to max_p
    # But that's a lot of memory: sum of p for primes p <= 5001 ~ 1.5M entries, fine
    
    # Actually, let's be smarter. Only track primes p where p <= current_len + 1.
    # As we add points, current_len increases, so we may need to add new primes.
    
    # Initialize counts for primes <= len(points)
    cur_len = len(points)
    
    # count_state[p] = list of length p, count_state[p][r] = count of points with v%p==r
    count_state = {}
    for p in PRIMES:
        if p > cur_len:
            break
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        count_state[p] = counts
    
    # missing_count[p] = number of classes with count 0
    missing_count = {}
    for p in count_state:
        missing_count[p] = sum(1 for c in count_state[p] if c == 0)
    
    # Now try to insert candidates
    inserted = 0
    for cand in candidates:
        if time.time() > time_limit:
            break
        
        new_len = len(points) + 1
        
        # Check all primes p <= new_len
        valid = True
        for p in PRIMES:
            if p > new_len:
                break
            
            if p not in count_state:
                # This prime was not previously relevant (p > old_len)
                # So p > cur_len, meaning p == new_len (since we add one at a time)
                # We need to check if adding cand makes all p classes covered
                # Count residues of points + cand mod p
                seen = bytearray(p)
                for v in points:
                    seen[v % p] = 1
                seen[cand % p] = 1
                if all(seen):
                    valid = False
                    break
            else:
                # p was relevant before
                # Check if cand's class is the unique missing one
                r = cand % p
                if count_state[p][r] == 0:
                    # This class is currently missing
                    # If it's the only missing class, adding cand would make p unsafe
                    if missing_count[p] == 1:
                        valid = False
                        break
                # If count_state[p][r] > 0, class is already covered, no problem
        
        if not valid:
            continue
        
        # Insert cand
        points.append(cand)
        in_set.add(cand)
        inserted += 1
        
        # Update state
        new_len = len(points)
        for p in PRIMES:
            if p > new_len:
                break
            if p not in count_state:
                # New prime, compute from scratch
                counts = [0] * p
                for v in points:
                    counts[v % p] += 1
                count_state[p] = counts
                missing_count[p] = sum(1 for c in counts if c == 0)
            else:
                r = cand % p
                old_count = count_state[p][r]
                if old_count == 0:
                    missing_count[p] -= 1
                count_state[p][r] = old_count + 1
    
    return points

def construct():
    start_time = time.time()
    
    # 1. Greedy removal to get a good starting point
    points = _greedy_remove()
    best_points = points[:]
    best_size = len(points)
    
    # 2. Try to improve by inserting points
    deadline = start_time + DEADLINE
    points = _try_add_points(points, deadline)
    
    if len(points) > best_size:
        best_size = len(points)
        best_points = points[:]
    
    # 3. If time remains, try another round with different random seed
    if time.time() < deadline - 30:
        random.seed(time.time())
        points2 = _greedy_remove()
        points2 = _try_add_points(points2, deadline)
        if len(points2) > best_size:
            best_size = len(points2)
            best_points = points2[:]
    
    # 4. Final verification
    if not _is_admissible(best_points):
        # Should not happen, but if it does, fall back to greedy only
        fallback = _greedy_remove()
        if _is_admissible(fallback):
            return fallback
        return [0]
    
    return best_points

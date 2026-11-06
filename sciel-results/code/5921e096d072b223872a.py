import time
import math
import random

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _greedy_reduce(primes, seed_val):
    random.seed(seed_val)
    points = list(range(N + 1))
    removed = []
    
    # We track the current set of points
    # To speed up counting, we can just use a list and filter
    # But for N=5000, list filtering is fast enough.
    
    for p in primes:
        current_size = len(points)
        if p > current_size:
            break
            
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
            
        if 0 in counts:
            # Already admissible for this p, no need to remove
            continue
            
        min_count = min(counts)
        candidates = [r for r in range(p) if counts[r] == min_count]
        
        # Random tie breaking
        target_residue = candidates[random.randint(0, len(candidates) - 1)]
        
        new_points = []
        for v in points:
            if v % p == target_residue:
                removed.append(v)
            else:
                new_points.append(v)
        points = new_points
        
    return points, removed

def _try_recover(primes, S, removed):
    # S is a list of points
    # removed is a list of points that were removed
    # We want to add back as many as possible from 'removed' to S
    
    if not removed:
        return S
    
    # Initialize present arrays and missing counts
    # We only care about primes p <= len(S) + len(removed) potentially?
    # Actually, the max size we can reach is N+1.
    # So we need present arrays for all primes p <= N+1.
    
    max_p = max(primes)
    # primes list contains all primes up to N.
    
    present = {}
    missing_count = {}
    
    # Initialize for all primes in primes
    for p in primes:
        present[p] = [False] * p
        missing_count[p] = p
        
    # Mark present residues for current S
    for v in S:
        for p in primes:
            if p > len(S) + 1: # Optimization: only care about p <= current max size?
                # But size can grow. Let's just update all.
                # N=5000, primes ~669. 5000 * 669 = 3.3M ops. Acceptable.
                pass
            present[p][v % p] = True
            
    # Recalculate missing counts
    for p in primes:
        missing_count[p] = p - sum(present[p])
        
    # Try to add removed elements
    # Sort removed elements? Maybe random order is better to explore?
    # Let's shuffle removed elements to break symmetries
    random.shuffle(removed)
    
    current_size = len(S)
    added_any = True
    # We might need to try multiple passes?
    # Adding one element might make another element addable?
    # Actually, adding an element only *decreases* missing_count.
    # If an element was not addable because missing_count[p]==1, adding another element
    # that fills a *different* hole for p doesn't change the status of the first element
    # unless the first element was the *only* one that could fill the *other* holes?
    # No. If missing_count[p] == 1, the only missing residue is $r$.
    # Only an element $x$ with $x \equiv r \pmod p$ can fill it.
    # If we add such an element, missing_count[p] becomes 0.
    # Then *no* other element can be added that would fill a hole for p (since there are none).
    # But adding an element might make it *impossible* to add other elements for *other* primes.
    # So a single pass through the shuffled list should be sufficient for a greedy addition.
    # However, the order matters.
    
    for x in removed:
        # Check if x can be added
        can_add = True
        # The new size will be current_size + 1
        # We need to check primes p <= current_size + 1
        for p in primes:
            if p > current_size + 1:
                break
            r = x % p
            if not present[p][r]:
                if missing_count[p] == 1:
                    can_add = False
                    break
        
        if can_add:
            # Add x
            S.append(x)
            current_size += 1
            # Update present and missing_count
            for p in primes:
                if p > current_size: # Only need to update for p <= new size?
                    # Actually, if p > current_size, it wasn't a constraint.
                    # But it might become a constraint later?
                    # No, if p > current_size, the condition "some residue missed" is
                    # vacuously true if we don't care about p?
                    # The condition is: for all p <= k, some residue missed.
                    # If p > k, we don't check it.
                    # So we only need to maintain state for p <= current_size.
                    # But when current_size increases, we need to start checking new primes.
                    # So we should update state for all primes, or at least those <= current_size.
                    # Updating all is simpler and N is small.
                    pass
                r = x % p
                if not present[p][r]:
                    present[p][r] = True
                    missing_count[p] -= 1
                    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    primes = _primes(N)
    
    best_points = None
    best_size = -1
    
    seed_counter = 0
    while time.time() < deadline:
        S, removed = _greedy_reduce(primes, seed_counter)
        S = _try_recover(primes, S, removed)
        
        current_size = len(S)
        
        if current_size > best_size:
            best_size = current_size
            best_points = S
            
        seed_counter += 1
        
        if best_size == N + 1:
            break

    if best_points is None:
        S, removed = _greedy_reduce(primes, 0)
        best_points = _try_recover(primes, S, removed)
        
    return best_points

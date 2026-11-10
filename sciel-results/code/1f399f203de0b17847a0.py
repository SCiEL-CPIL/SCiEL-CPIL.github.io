"""Improved program for the narrow admissible tuple task.

Strategy: multi-strategy greedy construction with randomized restarts and iterative local search (add-back).
The core idea is to start with a large candidate set (all integers in [0, N]) and iteratively remove
residue classes that are fully covered, while trying to add back removed elements when possible.
Randomized restarts with different removal orders help escape local optima.
"""

import time
import random
import math

N = 5000
DEADLINE = 280  # seconds; leave safety margin before 330s kill

def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(sieve[p*p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]

# Precompute primes up to N (max possible tuple length is N+1, but realistically much smaller)
_PRIMES_UP_TO_N = _primes(N)

def _is_admissible(points, primes_needed):
    """Check if points form an admissible tuple with respect to primes_needed."""
    # Optimization: If the number of points is less than the smallest prime, it's trivially admissible
    # but we still need to check up to len(points).
    # We can speed this up by checking if any prime p <= len(points) has all residues.
    n_points = len(points)
    if n_points == 0:
        return True
    
    # We only need to check primes up to n_points
    for p in primes_needed:
        if p > n_points:
            break
        # Use a boolean array for speed if p is small, set if p is large?
        # For p <= 5000, a set is usually fine, but we can try a list of booleans.
        seen = [False] * p
        count = 0
        for v in points:
            r = v % p
            if not seen[r]:
                seen[r] = True
                count += 1
                if count == p:
                    return False
    return True

def _count_covered_residues(points, p):
    """Return the set of residues mod p that are covered by points."""
    return set(v % p for v in points)

def _greedy_remove(points, primes_to_check, rng):
    """
    Greedy removal: while there exists a prime p in primes_to_check such that all p residues
    are covered, remove the least-populated residue class mod p.
    Returns the final list of points.
    """
    points = list(points)
    n = len(points)
    # We only need to check primes up to len(points)
    relevant_primes = [p for p in primes_to_check if p <= len(points)]
    
    while relevant_primes:
        # Find a prime where all residues are covered
        violated = None
        for p in relevant_primes:
            residues = set(v % p for v in points)
            if len(residues) == p:
                violated = p
                break
        
        if violated is None:
            break
        
        p = violated
        # Count residues
        classes = {}
        for v in points:
            r = v % p
            if r not in classes:
                classes[r] = []
            classes[r].append(v)
        
        # Remove the smallest class (with random tie-breaking)
        min_size = min(len(c) for c in classes.values())
        smallest_classes = [c for c in classes.values() if len(c) == min_size]
        chosen = rng.choice(smallest_classes)
        chosen_set = set(chosen)
        points = [v for v in points if v not in chosen_set]
        
        # Update relevant_primes: if len(points) changed, some primes may no longer be relevant
        relevant_primes = [p for p in relevant_primes if p <= len(points)]
    
    return points

def _add_back_search(points, all_candidates, primes_to_check, rng, max_rounds=10, start_time=0, deadline=280):
    """
    Try to add back removed elements one at a time if it doesn't violate admissibility.
    all_candidates: set of all integers in [0, N]
    points: current admissible tuple
    """
    points_set = set(points)
    removed = [v for v in all_candidates if v not in points_set]
    rng.shuffle(removed)
    
    improved = True
    round_num = 0
    while improved and round_num < max_rounds:
        # Check time
        if time.time() - start_time > deadline - 5:
            break
            
        improved = False
        round_num += 1
        # Re-check in random order
        order = list(removed)
        rng.shuffle(order)
        for v in order:
            if time.time() - start_time > deadline - 5:
                break
            if v in points_set:
                continue
            
            # Fast check: see if adding v creates a full residue class for any p <= len(points)+1
            new_len = len(points_set) + 1
            ok = True
            for p in primes_to_check:
                if p > new_len:
                    break
                # Check if v % p is the missing residue
                # We need to know which residues are present.
                # To avoid rebuilding the set every time, we could maintain counts, 
                # but for simplicity and safety, we do a quick check.
                # Actually, checking if len(set(x % p for x in points_set) | {v % p}) == p is expensive.
                # Better: if v % p is NOT in {x % p for x in points_set}, it's safe for this p.
                # If v % p IS in the set, we need to check if it's the *only* missing one?
                # No, admissibility fails if ALL residues are covered.
                # So it fails if (current residues) U {v%p} == {0..p-1}.
                # This implies current residues had p-1 distinct values, and v%p is the missing one.
                
                # Efficient check:
                # 1. If len(points_set) < p - 1, adding v cannot complete the set (max size p).
                #    Wait, if len(points_set) is k, we have at most k residues.
                #    To have all p residues, we need at least p points.
                #    So if len(points_set) < p - 1, it's impossible to cover all p residues even after adding v.
                if len(points_set) < p - 1:
                    continue
                    
                # 2. Check if v%p is the unique missing residue.
                #    This requires knowing the set of residues.
                #    We can compute the set of residues for points_set mod p.
                #    This is O(len(points_set)).
                #    Doing this for every v is O(N^2).
                #    Given N=5000, this might be slow if done many times.
                
                # Let's stick to the previous logic but with early breaks and time checks.
                # The previous logic was:
                # new_points = points_set | {v}
                # check admissibility of new_points
                # This is O(len(points_set) * num_primes).
                
                # Let's try a slightly optimized version:
                # For each p, check if v % p is in the set of residues of points_set.
                # If not, adding v doesn't complete the set for p.
                # If yes, we need to check if the set of residues of points_set already had p-1 elements.
                # If it had p-1 elements, and v%p is the missing one, then it fails.
                # If it had p elements, it already failed (but points_set is admissible, so it can't have p elements).
                # So, points_set mod p has size <= p-1.
                # It fails if size == p-1 AND v%p is the missing one.
                
                # We can precompute the residue counts for the current points_set?
                # That would be expensive to update.
                
                # Let's just use the direct check but break early if time is tight.
                residues = set()
                for x in points_set:
                    residues.add(x % p)
                if len(residues) == p - 1:
                    # Check if v%p is the missing one
                    if v % p not in residues:
                        ok = False
                        break
                # If len(residues) < p-1, adding v cannot make it p.
                # If len(residues) == p-1 and v%p is in residues, adding v doesn't change the set size, so it's still p-1 < p. Safe.
            
            if ok:
                points_set.add(v)
                improved = True
        # If we added anything, continue; the removed list is implicitly updated via points_set
    
    return list(points_set)

def _construct_one_strategy(rng, strategy):
    """
    Construct an admissible tuple using a specific strategy.
    Returns the list of points.
    """
    all_candidates = set(range(N + 1))
    
    if strategy == 0:
        # Start with all candidates, greedy remove
        points = list(all_candidates)
        points = _greedy_remove(points, _PRIMES_UP_TO_N, rng)
    elif strategy == 1:
        # Start with a random subset of ~70% density, then greedy remove
        density = 0.7
        points = [v for v in range(N + 1) if rng.random() < density]
        points = _greedy_remove(points, _PRIMES_UP_TO_N, rng)
    elif strategy == 2:
        # Start with all candidates, but use a different removal heuristic:
        # remove the residue class that has the most "conflict" (i.e., removing it helps most primes)
        points = list(all_candidates)
        n = len(points)
        relevant_primes = [p for p in _PRIMES_UP_TO_N if p <= n]
        
        while relevant_primes:
            violated = None
            for p in relevant_primes:
                residues = set(v % p for v in points)
                if len(residues) == p:
                    violated = p
                    break
            if violated is None:
                break
            
            p = violated
            classes = {}
            for v in points:
                r = v % p
                if r not in classes:
                    classes[r] = []
                classes[r].append(v)
            
            # Score each class: how many other primes would be helped by removing this class?
            # Approximation: count how many of these elements are in "tight" classes for other primes
            # For simplicity, just pick smallest with random tie-break
            min_size = min(len(c) for c in classes.values())
            smallest_classes = [c for c in classes.values() if len(c) == min_size]
            chosen = rng.choice(smallest_classes)
            chosen_set = set(chosen)
            points = [v for v in points if v not in chosen_set]
            relevant_primes = [p for p in relevant_primes if p <= len(points)]
        
        points = _greedy_remove(points, _PRIMES_UP_TO_N, rng)
    elif strategy == 3:
        # Start with primes > sqrt(N) plus 0, then add back
        points = [0] + [p for p in _PRIMES_UP_TO_N if p * p > N]
        points = _add_back_search(points, all_candidates, _PRIMES_UP_TO_N, rng, max_rounds=200)
    else:
        # Strategy 4: Random initial subset of varying density
        density = rng.uniform(0.3, 0.9)
        points = [v for v in range(N + 1) if rng.random() < density]
        points = _greedy_remove(points, _PRIMES_UP_TO_N, rng)
        points = _add_back_search(points, all_candidates, _PRIMES_UP_TO_N, rng, max_rounds=50)
    
    return points

def construct():
    """
    Main entry point. Uses multi-strategy greedy construction with randomized restarts
    and iterative local search (add-back) to find a large admissible tuple in [0, N].
    """
    start_time = time.time()
    best_points = []
    best_size = 0
    
    # First, do a quick baseline to have something valid
    rng0 = random.Random(42)
    baseline = _construct_one_strategy(rng0, 0)
    if _is_admissible(baseline, [p for p in _PRIMES_UP_TO_N if p <= len(baseline)]):
        best_points = baseline
        best_size = len(baseline)
    
    # Multi-restart search
    seed = 12345
    rng = random.Random(seed)
    num_restarts = 0
    all_candidates_set = set(range(N + 1))
    
    while time.time() - start_time < DEADLINE - 10:
        strategy = rng.randint(0, 4)
        points = _construct_one_strategy(rng, strategy)
        
        # Verify admissibility
        if not _is_admissible(points, [p for p in _PRIMES_UP_TO_N if p <= len(points)]):
            # Repair if needed
            points = _greedy_remove(points, _PRIMES_UP_TO_N, rng)
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = points
            # Try add-back on the new best (limited rounds to save time)
            if time.time() - start_time < DEADLINE - 20:
                improved = _add_back_search(best_points, all_candidates_set, _PRIMES_UP_TO_N, rng, max_rounds=5, start_time=start_time, deadline=DEADLINE)
                if _is_admissible(improved, [p for p in _PRIMES_UP_TO_N if p <= len(improved)]):
                    if len(improved) > best_size:
                        best_size = len(improved)
                        best_points = improved
        
        num_restarts += 1
        # Periodic polish
        if num_restarts % 10 == 0 and time.time() - start_time < DEADLINE - 30:
            improved = _add_back_search(best_points, all_candidates_set, _PRIMES_UP_TO_N, rng, max_rounds=10, start_time=start_time, deadline=DEADLINE)
            if _is_admissible(improved, [p for p in _PRIMES_UP_TO_N if p <= len(improved)]):
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_points = improved
    
    # Final safety check
    if not _is_admissible(best_points, [p for p in _PRIMES_UP_TO_N if p <= len(best_points)]):
        # Emergency repair
        best_points = _greedy_remove(best_points, _PRIMES_UP_TO_N, random.Random(0))
    
    return best_points

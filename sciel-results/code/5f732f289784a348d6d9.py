"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from collections import defaultdict

N = 5000
DEADLINE = 320.0  # safety margin before 330s kill

# EVOLVE-BLOCK-START

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(sieve[p*p::p]))
    return [i for i in range(2, limit + 1) if sieve[i]]


def _is_admissible(points, primes_up_to_k):
    """Check if the set of points is admissible: for every prime p <= len(points),
    some residue class mod p is missed."""
    k = len(points)
    for p in primes_up_to_k:
        if p > k:
            break
        residues = set()
        for v in points:
            residues.add(v % p)
        if len(residues) == p:
            return False
    return True


def _greedy_construct(primes_needed):
    """
    Greedy construction: start with all integers [0, N], then for each prime
    in order, remove the smallest residue class if all classes are covered.
    This is a local heuristic that tries to maximize the remaining set size.
    
    primes_needed: list of primes that must be checked (primes <= final size)
    We iterate primes in increasing order, and for each prime where all
    residue classes are present, remove the least-populated class.
    """
    # Start with all points
    points = list(range(N + 1))
    
    # We need to handle primes up to len(points), which decreases as we remove
    # Strategy: repeatedly check all relevant primes and fix violations
    # Use a worklist approach: keep fixing until no violations remain
    
    # Precompute primes up to N+1 (max possible size)
    all_primes = _sieve_primes(N + 1)
    
    changed = True
    max_iter = 1000
    iter_count = 0
    while changed and iter_count < max_iter:
        changed = False
        iter_count += 1
        k = len(points)
        # Only need to check primes <= k
        for p in all_primes:
            if p > k:
                break
            # Count residues
            classes = defaultdict(list)
            for v in points:
                classes[v % p].append(v)
            if len(classes) == p:
                # All classes covered - remove the smallest class
                smallest = min(classes.values(), key=len)
                # Remove all points in this class
                to_remove = set(smallest)
                points = [v for v in points if v not in to_remove]
                changed = True
                break  # Restart checking from smallest prime
    
    return points


def _local_search_improve(points, primes_needed, time_budget):
    """
    Try to improve the admissible set by adding back removed points.
    For each point not in the current set (from 0 to N), check if adding it
    preserves admissibility. If so, add it.
    """
    points_set = set(points)
    start_time = time.time()
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_budget:
            break
        for v in range(N + 1):
            if v in points_set:
                continue
            # Try adding v
            new_points = points_set | {v}
            k = len(new_points)
            admissible = True
            for p in primes_needed:
                if p > k:
                    break
                residues = set()
                for x in new_points:
                    residues.add(x % p)
                if len(residues) == p:
                    admissible = False
                    break
            if admissible:
                points_set.add(v)
                improved = True
    return list(points_set)


def _multi_start_greedy(primes_needed, time_budget):
    """
    Try multiple greedy strategies:
    1. Standard greedy (remove smallest class for violating primes)
    2. Greedy with different prime ordering
    3. Greedy starting from a subset
    
    Returns the best (largest) admissible set found.
    """
    best = None
    best_size = 0
    start_time = time.time()
    
    all_primes = _sieve_primes(N + 1)
    
    # Strategy 1: Standard greedy from full set
    if time.time() - start_time < time_budget:
        points = _greedy_construct(all_primes)
        if len(points) > best_size:
            best = points
            best_size = len(points)
    
    # Strategy 2: Greedy with primes in reverse order (larger primes first)
    if time.time() - start_time < time_budget:
        points = list(range(N + 1))
        changed = True
        max_iter = 1000
        iter_count = 0
        while changed and iter_count < max_iter:
            changed = False
            iter_count += 1
            k = len(points)
            for p in reversed(all_primes):
                if p > k:
                    continue
                if p < 2:
                    break
                classes = defaultdict(list)
                for v in points:
                    classes[v % p].append(v)
                if len(classes) == p:
                    smallest = min(classes.values(), key=len)
                    to_remove = set(smallest)
                    points = [v for v in points if v not in to_remove]
                    changed = True
                    break
        if len(points) > best_size:
            best = points
            best_size = len(points)
    
    # Strategy 3: Greedy with random prime ordering
    import random
    rng = random.Random(42)
    for trial in range(5):
        if time.time() - start_time > time_budget:
            break
        points = list(range(N + 1))
        changed = True
        max_iter = 1000
        iter_count = 0
        while changed and iter_count < max_iter:
            changed = False
            iter_count += 1
            k = len(points)
            # Shuffle primes <= k
            relevant_primes = [p for p in all_primes if p <= k]
            rng.shuffle(relevant_primes)
            for p in relevant_primes:
                classes = defaultdict(list)
                for v in points:
                    classes[v % p].append(v)
                if len(classes) == p:
                    smallest = min(classes.values(), key=len)
                    to_remove = set(smallest)
                    points = [v for v in points if v not in to_remove]
                    changed = True
                    break
        if len(points) > best_size:
            best = points
            best_size = len(points)
    
    return best if best else []


def _polish(points, time_budget):
    """
    Final polish: try to add back any points that were removed,
    and also try local moves (swap one point for another).
    """
    points_set = set(points)
    start_time = time.time()
    all_primes = _sieve_primes(N + 1)
    
    # Phase 1: Add back any point that preserves admissibility
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_budget:
            break
        for v in range(N + 1):
            if v in points_set:
                continue
            new_set = points_set | {v}
            k = len(new_set)
            admissible = True
            for p in all_primes:
                if p > k:
                    break
                residues = set(x % p for x in new_set)
                if len(residues) == p:
                    admissible = False
                    break
            if admissible:
                points_set.add(v)
                improved = True
    
    # Phase 2: Try swaps - remove one point, add another
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_budget:
            break
        for v in list(points_set):
            if time.time() - start_time > time_budget:
                break
            without_v = points_set - {v}
            k = len(without_v)
            # Check if without_v is admissible
            admissible_without = True
            for p in all_primes:
                if p > k:
                    break
                residues = set(x % p for x in without_v)
                if len(residues) == p:
                    admissible_without = False
                    break
            if not admissible_without:
                continue
            # Try adding each point not in without_v
            for w in range(N + 1):
                if w in without_v or w == v:
                    continue
                new_set = without_v | {w}
                k2 = len(new_set)
                admissible = True
                for p in all_primes:
                    if p > k2:
                        break
                    residues = set(x % p for x in new_set)
                    if len(residues) == p:
                        admissible = False
                        break
                if admissible:
                    points_set = new_set
                    improved = True
                    break
    
    return list(points_set)


def construct():
    start_time = time.time()
    time_budget = DEADLINE - (time.time() - start_time) - 5.0  # 5s safety margin
    
    all_primes = _sieve_primes(N + 1)
    
    # Step 1: Multi-start greedy
    points = _multi_start_greedy(all_primes, time_budget * 0.4)
    
    if not points:
        points = list(range(N + 1))
    
    # Step 2: Local search to add back points
    remaining_time = DEADLINE - (time.time() - start_time) - 5.0
    if remaining_time > 1.0:
        points = _local_search_improve(points, all_primes, remaining_time * 0.3)
    
    # Step 3: Polish
    remaining_time = DEADLINE - (time.time() - start_time) - 5.0
    if remaining_time > 1.0:
        points = _polish(points, remaining_time * 0.5)
    
    # Ensure admissibility - final check and repair if needed
    k = len(points)
    for p in all_primes:
        if p > k:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            # Need to remove a class
            classes = defaultdict(list)
            for v in points:
                classes[v % p].append(v)
            smallest = min(classes.values(), key=len)
            points = [v for v in points if v not in set(smallest)]
            k = len(points)
    
    return points

# EVOLVE-BLOCK-END

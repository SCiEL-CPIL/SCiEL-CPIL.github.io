"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 5000

# EVOLVE-BLOCK-START

def _primes(limit):
    """Generate primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:limit + 1:step] = [False] * len(range(start, limit + 1, step))
    return [p for p, ok in enumerate(sieve) if ok]


def _count_residues(points, p):
    """Count occurrences of each residue class mod p for the given points."""
    counts = [0] * p
    for v in points:
        counts[v % p] += 1
    return counts


def _is_admissible(points, primes_up_to_n):
    """Check if points form an admissible tuple."""
    n = len(points)
    for p in primes_up_to_n:
        if p > n:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True


def _greedy_remove(points, primes, deadline):
    """
    Greedy algorithm: repeatedly find a prime where all residue classes are covered,
    and remove the least-populated class. Uses a priority based on how 'tight' the
    constraint is (larger primes first, as they have more residue classes to cover).
    """
    start_time = time.time()
    
    # Sort primes in descending order to handle larger primes first (more constraints)
    # Actually, let's think: for a tuple of size n, we need primes p <= n.
    # Removing a class mod p removes ~n/p points.
    # Larger p removes fewer points per removal, so it's "cheaper".
    # But larger p means more residue classes, so it's harder to miss one.
    # Strategy: process primes in order of p (ascending or descending?).
    # Let's try: for each prime p <= n, if all residues are present, remove the smallest class.
    # To maximize size, we want to remove as few points as possible.
    # Removing class mod p removes ~n/p points.
    # So we should prefer removing classes for larger p (smaller removal count) first?
    # Or smaller p (larger removal count) first to quickly satisfy constraints?
    # Actually, if we remove a class mod p, it might create empty classes for other primes q.
    # So order matters.
    # Let's try a simple greedy: iterate primes in ascending order.
    # If a prime p has all residues covered, remove the least populated class.
    # This is what the original did. Let's see if we can improve.
    
    # Alternative: Look ahead. For each prime p that is violated, calculate the cost of removing each class.
    # Cost = number of points removed.
    # Pick the removal with minimum cost.
    
    points = list(points)
    n = len(points)
    
    while True:
        if time.time() > deadline:
            break
        
        n = len(points)
        if n < 2:
            break
        
        relevant_primes = [p for p in primes if p <= n]
        
        # Find all violated primes and their best removal option
        best_removal = None  # (cost, p, target_class)
        
        for p in relevant_primes:
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            
            if 0 in counts:
                continue
            
            # All classes filled: find the least-populated class
            min_count = min(counts)
            if min_count == 0:
                continue
            
            # Pick the class with minimum count; tie-break by lowest residue
            target_class = counts.index(min_count)
            
            # Cost is min_count
            # We want to minimize cost
            if best_removal is None or min_count < best_removal[0]:
                best_removal = (min_count, p, target_class)
        
        if best_removal is None:
            # No violations
            break
        
        cost, p, target_class = best_removal
        points = [v for v in points if v % p != target_class]
    
    return points


def _local_search(points, primes, deadline):
    """
    Try to improve the solution by attempting to add back points that were removed,
    or by swapping points. This is a simple local search.
    """
    start_time = time.time()
    
    # First, get the greedy solution
    points = _greedy_remove(list(points), primes, deadline)
    
    # Now try to improve
    # We can try to add back points from the original set that were removed.
    # But we don't know which were removed. Let's keep track.
    
    # Actually, let's just do a simple hill-climbing on the set of points.
    # But the search space is huge.
    
    # Alternative: Use the greedy solution as a starting point.
    # Try to find a better solution by considering different removal orders.
    
    # For now, just return the greedy solution.
    # We can enhance this with more sophisticated search later.
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 290  # safety margin before 330s wall-clock limit

    points = list(range(N + 1))
    
    # Primes up to N+1
    all_primes = _primes(N + 1)
    
    # Use the improved greedy algorithm
    result = _greedy_remove(points, all_primes, deadline)
    
    # Final check: ensure admissibility
    n = len(result)
    for p in [p for p in all_primes if p <= n]:
        residues = set(v % p for v in result)
        if len(residues) == p:
            # Still a violation - repair
            counts = [0] * p
            for v in result:
                counts[v % p] += 1
            target_class = counts.index(min(counts))
            result = [v for v in result if v % p != target_class]
            break
    
    return result

# EVOLVE-BLOCK-END

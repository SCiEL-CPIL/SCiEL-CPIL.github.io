"""Combined: randomized greedy with multiple seeds + local search polish."""
import time
import math
import random
import numpy as np

N = 5000


def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _greedy_pass(primes, seed_val):
    """One greedy pass with random tie-breaking."""
    random.seed(seed_val)
    points = list(range(N + 1))
    
    for p in primes:
        current_size = len(points)
        if p > current_size:
            break
        
        # Count residues using a simple list (fast enough for N=5000)
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        
        # Check if all classes are covered
        if 0 in counts:
            continue
        
        # Find minimum count and candidates
        min_count = min(counts)
        candidates = [r for r in range(p) if counts[r] == min_count]
        
        # Randomly choose among tied minimums
        target_residue = candidates[random.randint(0, len(candidates) - 1)]
        
        # Remove all points in that residue class
        points = [v for v in points if v % p != target_residue]
    
    return points


def _local_search(points, primes, time_limit):
    """Try to add back removed points that maintain admissibility."""
    start_time = time.time()
    points_set = set(points)
    all_points_set = set(range(N + 1))
    removed = sorted(all_points_set - points_set)
    
    if not removed:
        return points
    
    points_arr = np.array(points, dtype=np.int64)
    
    # Precompute primes up to len(points)+1
    max_prime_needed = len(points) + 1
    relevant_primes = [p for p in primes if p <= max_prime_needed]
    
    # Try adding back points one by one
    for v in removed:
        if time.time() - start_time > time_limit:
            break
        
        # Check if adding v maintains admissibility
        new_len = len(points) + 1
        admissible = True
        for p in relevant_primes:
            if p > new_len:
                break
            # Count residues including v
            residues = set(x % p for x in points)
            residues.add(v % p)
            if len(residues) == p:
                admissible = False
                break
        
        if admissible:
            points.append(v)
            points_set.add(v)
            points_arr = np.concatenate([points_arr, [v]])
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 290  # 40s safety margin
    
    primes = _primes(N + 1)
    
    best_points = None
    best_size = -1
    
    # Run multiple randomized greedy passes
    seed_counter = 0
    while time.time() < deadline - 60:  # Leave 60s for local search
        current_points = _greedy_pass(primes, seed_counter)
        current_size = len(current_points)
        
        if current_size > best_size:
            best_size = current_size
            best_points = current_points
        
        seed_counter += 1
        
        # Early termination if we've found a very good solution
        if best_size >= N - 100:
            break
    
    if best_points is None:
        best_points = _greedy_pass(primes, 0)
        best_size = len(best_points)
    
    # Local search to polish the best solution
    remaining_time = deadline - 10 - time.time()
    if remaining_time > 5:
        best_points = _local_search(best_points, primes, remaining_time)
    
    return sorted(best_points)

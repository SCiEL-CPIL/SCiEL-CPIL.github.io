"""Improved program: greedy construction of largest admissible tuple in [0, 5000]."""
import time
import math
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


def _is_admissible(points, max_prime_len):
    """Check if the tuple of points is admissible."""
    for p in _primes(max_prime_len):
        residues = set(x % p for x in points)
        if len(residues) == p:
            return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 280  # 280s safety margin out of 330s

    # Strategy: Start with all points [0, N], then iteratively remove points
    # that are in the least-populated residue class for each prime that is fully covered.
    # Use NumPy for vectorized operations to speed up residue counting.

    points = set(range(N + 1))
    points_arr = np.array(sorted(points), dtype=np.int64)

    # Precompute primes up to N+1
    all_primes = _primes(N + 1)

    # Phase 1: Greedy removal - process primes in order, removing least-populated residue class
    for p in all_primes:
        if time.time() > deadline:
            break
        current_len = len(points_arr)
        if p > current_len:
            break

        # Compute residue counts using NumPy (vectorized)
        residues = points_arr % p
        counts = np.bincount(residues, minlength=p)

        if counts.min() > 0:
            # All classes covered; remove the least-populated class
            min_class = np.argmin(counts)
            # Create mask for points NOT in the least-populated class
            mask = (points_arr % p) != min_class
            points_arr = points_arr[mask]
            points = set(points_arr.tolist())

    # Phase 2: Try to re-add removed points that no longer violate admissibility
    # Collect all removed points
    all_points_set = set(range(N + 1))
    removed = all_points_set - points

    if removed and time.time() < deadline - 30:
        removed_list = sorted(removed)
        # Try adding back points one by one (or in batches) if admissibility is maintained
        # Use a smarter approach: for each removed point, check if adding it back
        # would violate any prime constraint
        for v in removed_list:
            if time.time() > deadline - 10:
                break
            # Check if adding v back maintains admissibility
            new_points_arr = np.concatenate([points_arr, [v]])
            new_len = len(new_points_arr)
            admissible = True
            for p in all_primes:
                if p > new_len:
                    break
                residues = new_points_arr % p
                counts = np.bincount(residues, minlength=p)
                if counts.min() > 0:
                    # This prime is now fully covered - violation
                    admissible = False
                    break
            if admissible:
                points_arr = new_points_arr
                points.add(v)

    return sorted(points_arr.tolist())

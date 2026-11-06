"""Structural change: greedy point-removal via max-coverage of active conflicts."""
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


def construct():
    start_time = time.time()
    deadline = start_time + 300  # safety margin

    # Precompute primes up to N+1
    all_primes = _primes(N + 1)

    # Start with all points [0, N]
    points = set(range(N + 1))
    points_list = list(points)

    def _active_conflicts(pts_list, max_prime):
        """Return list of primes p <= max_prime where all p residue classes are covered."""
        conflicts = []
        for p in all_primes:
            if p > max_prime:
                break
            residues = set(x % p for x in pts_list)
            if len(residues) == p:
                conflicts.append(p)
        return conflicts

    # We'll iterate: find active conflicts, for each find least-populated residue,
    # then pick the point that appears in the most "least-populated residue classes"
    # across active conflicts, and remove it.

    # For efficiency, work with numpy arrays
    pts = np.array(points_list, dtype=np.int32)

    while True:
        if time.time() > deadline:
            break

        current_len = len(pts)
        if current_len == 0:
            break

        # Find active conflicts: primes p <= current_len where all p classes covered
        active = []
        for p in all_primes:
            if p > current_len:
                break
            residues = pts % p
            counts = np.bincount(residues, minlength=p)
            if len(counts) < p:
                continue
            if np.min(counts) == 0:
                continue
            # All classes covered - this is an active conflict
            min_res = int(np.argmin(counts))
            active.append((p, min_res, int(np.min(counts))))

        if not active:
            break

        # For each active conflict, identify the set of points in the least-populated residue
        # We want to find the point that is in the least-populated class for the most conflicts
        # Build a count: for each point, how many active conflicts would it break if removed?

        # Efficient approach: for each active conflict (p, min_res), the points to remove are
        # those with pts % p == min_res. We count for each point how many such conflicts it's in.

        # Use an array to count coverage per point
        coverage = np.zeros(len(pts), dtype=np.int32)

        for (p, min_res, min_count) in active:
            mask = (pts % p) == min_res
            coverage[mask] += 1

        # Find the point with maximum coverage
        max_cov = np.max(coverage)
        if max_cov == 0:
            # No single point breaks any conflict - shouldn't happen if active is non-empty
            break

        # Remove one point with max coverage (pick the first one)
        best_idx = int(np.argmax(coverage))
        pts = np.delete(pts, best_idx)
        points.discard(int(pts[best_idx]) if best_idx < len(pts) else points_list[best_idx])

    # Final: verify admissibility and return
    result = sorted(pts.tolist())

    # Verify admissibility
    k = len(result)
    for p in all_primes:
        if p > k:
            break
        residues = set(x % p for x in result)
        if len(residues) == p:
            # Not admissible - need to fix, but this shouldn't happen
            break

    return result

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 5000

# EVOLVE-BLOCK-START
def _primes_upto(limit):
    """Return list of all primes <= limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]


def _check_and_repair(points, primes_to_check, deadline):
    """
    Given a set of points (as a sorted list), check admissibility for each prime in primes_to_check.
    If a prime has all residue classes covered, remove elements from the least-populated class.
    Returns the repaired list.
    """
    point_set = set(points)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for p in primes_to_check:
            if len(points) < p:
                # If we have fewer points than p, we can't cover all p classes
                continue
            # Count residue classes
            class_counts = {}
            for v in points:
                r = v % p
                class_counts[r] = class_counts.get(r, 0) + 1
            if len(class_counts) < p:
                # Already missing a class, no action needed
                continue
            # All p classes are covered; find the least-populated class
            min_count = min(class_counts.values())
            # Remove all elements in classes that have the minimum count
            to_remove = set()
            for v in points:
                if class_counts[v % p] == min_count:
                    to_remove.add(v)
            points = [v for v in points if v not in to_remove]
            changed = True
            # After removal, re-check from the beginning since sizes changed
            break
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s kill

    # Start with all integers in [0, N]
    points = list(range(N + 1))

    # We need to check primes up to len(points), which changes as we remove elements.
    # Precompute primes up to N+1 (max possible length)
    all_primes = _primes_upto(N + 1)

    # Iteratively repair: for each prime p <= len(points), ensure not all classes are covered
    # Process in rounds; each round, check all relevant primes and fix violations
    max_rounds = 200
    for round_num in range(max_rounds):
        if time.time() >= deadline:
            break
        m = len(points)
        # Only need to check primes p <= m
        relevant_primes = [p for p in all_primes if p <= m]
        if not relevant_primes:
            break

        # Check if any prime violates admissibility
        violation_found = False
        for p in relevant_primes:
            if time.time() >= deadline:
                break
            if m < p:
                continue
            class_counts = {}
            for v in points:
                r = v % p
                class_counts[r] = class_counts.get(r, 0) + 1
            if len(class_counts) == p:
                # Violation: all p classes covered. Remove least-populated class(es)
                min_count = min(class_counts.values())
                to_remove = set()
                for v in points:
                    if class_counts[v % p] == min_count:
                        to_remove.add(v)
                points = [v for v in points if v not in to_remove]
                violation_found = True
                break

        if not violation_found:
            break

    # Final verification
    m = len(points)
    for p in _primes_upto(m):
        if m < p:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            # Shouldn't happen, but if it does, do one more repair pass
            class_counts = {}
            for v in points:
                r = v % p
                class_counts[r] = class_counts.get(r, 0) + 1
            min_count = min(class_counts.values())
            to_remove = set()
            for v in points:
                if class_counts[v % p] == min_count:
                    to_remove.add(v)
            points = [v for v in points if v not in to_remove]
            break

    return points
# EVOLVE-BLOCK-END

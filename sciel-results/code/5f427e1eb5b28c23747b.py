"""Improved program: greedy construction of largest admissible tuple in [0, 5000]."""
import time
import math

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
    # This is a greedy approach to maximize the remaining set size.

    # Actually, a better strategy: start with a large initial set and repair.
    # The seed starts with primes > sqrt(N) and 1, which is small.
    # Let's try starting with all of [0, N] and removing points greedily.

    # But removing from [0, N] where N=5000 means we have 5001 points.
    # For each prime p <= 5001, we need to miss at least one residue class mod p.
    # The number of primes <= 5001 is about 669.

    # Greedy removal: while the set is not admissible, find a prime p that is fully covered,
    # and remove the least-populated residue class mod p.

    points = set(range(N + 1))

    # Precompute primes up to N+1
    all_primes = _primes(N + 1)

    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        # We only need to check primes p <= len(points)
        current_len = len(points)
        for p in all_primes:
            if p > current_len:
                break
            # Check if all p residue classes mod p are covered
            residues = {}
            for v in points:
                r = v % p
                if r not in residues:
                    residues[r] = 0
                residues[r] += 1

            if len(residues) == p:
                # All classes covered; remove the least-populated class
                min_class = min(residues, key=residues.get)
                # Remove all points in this residue class
                to_remove = [v for v in points if v % p == min_class]
                for v in to_remove:
                    points.discard(v)
                changed = True
                break

    return sorted(points)

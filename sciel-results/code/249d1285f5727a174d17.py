"""Narrow admissible tuple: greedy maximum independent set on conflict graph."""
import math
from collections import defaultdict

N = 5000


def _primes(limit):
    """Return list of primes up to limit (inclusive)."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(math.isqrt(limit)) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _build_conflict_edges(primes, k, n):
    """
    Build a conflict graph where vertices are 0..n.
    An edge exists between u and v if there exists a prime p <= k
    such that including both u and v would force all p residue classes
    to be covered in the final set.
    
    For efficiency, we approximate: two vertices u, v conflict for prime p
    if u % p != v % p and the current set would otherwise be missing exactly
    the residues {u%p, v%p} - but that's state-dependent.
    
    Instead, use a simpler sufficient condition: u and v are in conflict
    if there exists p <= k such that u % p != v % p. This is too aggressive.
    
    Better approach: For each prime p <= k, partition [0, n] by residue mod p.
    The constraint is that at least one residue class mod p must be entirely absent.
    
    We model this as: for each prime p, we must choose at least one residue r_p
    such that no selected vertex has value ≡ r_p (mod p).
    
    This is a complex constraint satisfaction. For the greedy approach, we use
    a different strategy: iteratively add vertices, checking admissibility.
    """
    pass


def construct():
    n = N
    # For the tuple of size m, we need primes p <= m.
    # We don't know m in advance, so we work with all primes up to n.
    # Strategy: Greedy vertex selection. Start with all vertices, then remove
    # vertices that cause conflicts.
    
    # A vertex x is "bad" if there exists a prime p such that all p residue
    # classes are present in the current set. We need to remove enough vertices
    # so that for every prime p <= |set|, some residue class is missing.
    
    # Greedy approach: Start with all numbers 0..n. Iteratively, for each prime
    # p, if all p residue classes are covered, remove the vertex from the least
    # populated residue class. Repeat until admissible.
    
    # But we want to MAXIMIZE the tuple size. The above approach minimizes removals
    # for a given starting set, which is good.
    
    # However, the starting set matters. Let's try starting with all [0, n] and
    # removing minimally.
    
    points = list(range(n + 1))
    
    # We need to check primes up to len(points). Initially len = n+1 = 5001.
    # Primes up to 5001.
    all_primes = _primes(n + 1)
    
    changed = True
    while changed:
        changed = False
        # Check primes in order. For efficiency, check smaller primes first
        # since they have more residue classes and are more likely to be fully covered.
        for p in all_primes:
            if p > len(points):
                break
            # Count residues
            residue_counts = defaultdict(int)
            residue_members = defaultdict(list)
            for v in points:
                r = v % p
                residue_counts[r] += 1
                residue_members[r].append(v)
            
            if len(residue_counts) == p:
                # All p residues are covered. Remove one member from the least populated.
                min_residue = min(residue_counts.keys(), key=lambda r: residue_counts[r])
                # Remove the first member of that residue
                to_remove = residue_members[min_residue][0]
                points.remove(to_remove)
                changed = True
                break  # Restart from beginning since set changed
    
    # Final check: ensure admissibility
    final_primes = _primes(len(points))
    for p in final_primes:
        residues = set(v % p for v in points)
        if len(residues) == p:
            # Still not admissible - this shouldn't happen but handle it
            residue_counts = defaultdict(int)
            residue_members = defaultdict(list)
            for v in points:
                r = v % p
                residue_counts[r] += 1
                residue_members[r].append(v)
            min_residue = min(residue_counts.keys(), key=lambda r: residue_counts[r])
            to_remove = residue_members[min_residue][0]
            points.remove(to_remove)
    
    return points

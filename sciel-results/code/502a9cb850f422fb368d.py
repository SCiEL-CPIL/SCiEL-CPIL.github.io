"""Rewritten program: Greedy MIS-inspired maximization with dynamic degree-based heuristic for admissible tuples in [0, 5000]."""

import numpy as np

N = 5000


def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]


def construct():
    primes = _primes(N + 1)
    prime_set = set(primes)
    
    # Start with the full range [0, N]
    points = list(range(N + 1))
    n = len(points)
    
    # Initialize counts
    counts = {}
    is_full = {}
    for p in primes:
        c = [0] * p
        for x in points:
            c[x % p] += 1
        counts[p] = c
        is_full[p] = (min(c) > 0)

    # Top M candidates for detailed evaluation
    M = 50 

    while True:
        n = len(points)
        if n == 0:
            break
            
        # Identify full primes p <= n
        full_primes = [p for p in primes if p <= n and is_full[p]]
        
        if not full_primes:
            break
        
        # Calculate min counts for full primes
        # To optimize, we only need this for candidates
        candidates = []
        for p in full_primes:
            min_c = min(counts[p])
            # min_c must be > 0 because is_full[p] is True
            candidates.append((min_c, p, counts[p].index(min_c)))
            
        # Sort by min count (cheapest first)
        candidates.sort(key=lambda x: x[0])
        
        # Take top M candidates for "Degree Centrality" (dropped primes) evaluation
        top_candidates = candidates[:M]
        
        best_score = float('inf')
        best_p = None
        best_r = None
        best_k = None
        
        for k, p, r in top_candidates:
            n_new = n - k
            # Count full primes in (n_new, n]
            # These are primes q such that n_new < q <= n and is_full[q] is True
            # Note: q must be a prime.
            dropped_count = 0
            for q in full_primes:
                if n_new < q <= n:
                    dropped_count += 1
            
            # Score: Minimize (points removed - benefit of dropping full primes)
            # We use a weight for dropped primes. 
            # Each dropped prime saves us from having to fix it later.
            # Let's weight dropped primes heavily.
            score = k - dropped_count * 10 
            
            if score < best_score:
                best_score = score
                best_p = p
                best_r = r
                best_k = k
            elif score == best_score:
                # Tie breaker: prefer smaller k (less immediate loss)
                if k < best_k:
                    best_p = p
                    best_r = r
                    best_k = k

        if best_p is None:
            break
        
        # Remove all points x such that x % best_p == best_r
        to_remove = [x for x in points if x % best_p == best_r]
        
        # Update points
        remove_set = set(to_remove)
        points = [x for x in points if x not in remove_set]
        
        new_n = len(points)
        
        # Update counts and is_full
        # We only need to update primes q <= new_n? 
        # Actually, if q > new_n, we don't care if it's full or not anymore.
        # But we might care if new_n increases? No, n only decreases.
        # So we can ignore primes q > new_n.
        
        for x in to_remove:
            for q in primes:
                if q > new_n:
                    break
                idx = x % q
                counts[q][idx] -= 1
                if counts[q][idx] == 0 and is_full[q]:
                    is_full[q] = False

    return points

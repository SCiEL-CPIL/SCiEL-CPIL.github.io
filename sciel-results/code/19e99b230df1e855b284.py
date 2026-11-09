import time

N = 5000
_DEADLINE = time.time() + 320.0  # 330s wall-clock, 10s safety margin

def _primes(limit):
    """Return list of primes <= limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]


def construct():
    # Start from the full range [0, N] and repair admissibility.
    # Key improvement: among all primes that violate admissibility,
    # process the one where removing the least-populated class costs the fewest
    # elements first. This preserves more flexibility for subsequent primes.
    
    in_set = [True] * (N + 1)
    current_len = N + 1
    primes = _primes(N + 1)
    
    while time.time() < _DEADLINE:
        k = current_len
        if k < 2:
            break
        
        # Find all primes p <= k where all residues are covered
        # For each, compute the minimum residue count
        # Pick the prime where min_count is smallest (cheapest to fix)
        best_p = None
        best_min_count = float('inf')
        best_residue = None
        
        for p in primes:
            if p > k:
                break
            
            counts = [0] * p
            for v in range(N + 1):
                if in_set[v]:
                    counts[v % p] += 1
            
            min_count = min(counts)
            if min_count > 0:
                # All residues covered - this prime violates admissibility
                # We need to remove one residue class
                # Track the "cheapest" violation
                if min_count < best_min_count:
                    best_min_count = min_count
                    best_p = p
                    best_residue = counts.index(min_count)
        
        if best_p is None:
            # No violations - tuple is admissible
            break
        
        # Remove the least-populated residue class for the cheapest violating prime
        for v in range(best_residue, N + 1, best_p):
            if in_set[v]:
                in_set[v] = False
                current_len -= 1
    
    result = [v for v in range(N + 1) if in_set[v]]
    return result

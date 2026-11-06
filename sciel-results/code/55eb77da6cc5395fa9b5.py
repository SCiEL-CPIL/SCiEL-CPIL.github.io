import numpy as np

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    primes = _primes(N)
    # Only primes p <= N can constrain the set [0, N]
    # Actually, if p > N, the constraint is trivially satisfied for any subset of size <= N+1?
    # The condition is: "for every prime p <= len(tuple) some residue class mod p is missed".
    # If we aim for a tuple of size K, we need to satisfy constraints for p <= K.
    # Since K <= N+1, we only need to consider primes p <= N+1.
    # My primes list goes up to N. This is sufficient as K <= N+1, so max p is <= N+1.
    # If N+1 is prime, we need it. _primes(N) might miss N+1 if N+1 is prime.
    # Let's extend primes to N+1.
    primes = _primes(N + 1)
    
    # Initialize residues randomly or to 0
    # Random initialization often helps escape local minima
    residues = {p: 0 for p in primes}
    
    # We will perform local search
    # To speed up, we can precompute the indices for each prime's residues
    # But N is small enough (5000) that direct loops are okay.
    
    def get_score(res):
        """Calculate size of admissible tuple for given residues."""
        removed = [False] * (N + 1)
        for p in primes:
            r = res[p]
            # Mark elements in residue r mod p
            for x in range(r, N + 1, p):
                removed[x] = True
        return N + 1 - sum(removed)
    
    current_score = get_score(residues)
    
    # Local Search Loop
    improved = True
    max_iterations = 50
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for p in primes:
            # Calculate coverage by all other primes
            # We can do this by temporarily removing p's contribution
            # But it's easier to just compute the "others" mask
            others_removed = [False] * (N + 1)
            for q in primes:
                if q == p:
                    continue
                r = residues[q]
                for x in range(r, N + 1, q):
                    others_removed[x] = True
            
            # For each possible residue r for p, calculate the number of NEW elements removed
            # New elements removed = | {x in C_p(r) | not others_removed[x] } |
            # We want to MINIMIZE this number.
            best_r = residues[p]
            min_new_removed = None
            for r in range(p):
                new_removed = 0
                for x in range(r, N + 1, p):
                    if not others_removed[x]:
                        new_removed += 1
                if min_new_removed is None or new_removed < min_new_removed:
                    min_new_removed = new_removed
                    best_r = r
            
            if best_r != residues[p]:
                residues[p] = best_r
                improved = True
                # Recalculate global score to track it, though local improvement guarantees global improvement
                # current_score = get_score(residues) # Can skip if we trust the local logic
                # Actually, let's just update the score incrementally or recalc at end of sweep
    
    # Final Score Calculation
    final_score = get_score(residues)
    
    # Construct the actual list
    removed = [False] * (N + 1)
    for p in primes:
        r = residues[p]
        for x in range(r, N + 1, p):
            removed[x] = True
            
    result = [x for x in range(N + 1) if not removed[x]]
    return result

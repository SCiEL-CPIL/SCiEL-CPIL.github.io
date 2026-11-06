import time
import math

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
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    primes = _primes(N)
    # Map prime to index for faster access? Or just use prime value as key in dict.
    # Using list of lists indexed by prime value is wasteful (5000 lists).
    # But 5000 is small. Let's use a dict: prime -> list of counts.
    
    # Initialize active set
    active = [True] * (N + 1)
    
    # Initialize counts
    # counts[p] = [count_0, count_1, ..., count_{p-1}]
    counts = {}
    for p in primes:
        c = [0] * p
        for i in range(N + 1):
            c[i % p] += 1
        counts[p] = c
        
    # Precompute residues for each prime to speed up updates?
    # residues[p][i] = i % p
    residues = {}
    for p in primes:
        residues[p] = [i % p for i in range(N + 1)]
        
    while True:
        if time.time() > deadline:
            break
            
        # Calculate current size
        k = sum(active)
        
        # Find violated primes p <= k
        # Violated if min(counts[p]) > 0
        best_p = None
        best_r = None
        min_count = float('inf')
        
        for p in primes:
            if p > k:
                break
            # Check if p is violated
            # If all residues are covered, min count > 0
            # We want to find the residue with the MINIMUM count among violated primes
            # to remove the fewest elements.
            
            # Optimization: if min count is 0, it's not violated.
            # We need to check if min(counts[p]) > 0.
            # And if so, what is that min?
            
            # We can just find min(counts[p])
            m = min(counts[p])
            if m > 0:
                if m < min_count:
                    min_count = m
                    best_p = p
                    # Find the residue r with this count
                    # There might be multiple, pick any.
                    best_r = counts[p].index(m)
        
        if best_p is None:
            # No violated primes, tuple is admissible
            break
            
        # Remove all x in S such that x % best_p == best_r
        to_remove = []
        for x in range(N + 1):
            if active[x] and residues[best_p][x] == best_r:
                to_remove.append(x)
                
        # Update active
        for x in to_remove:
            active[x] = False
            
        # Update counts for all primes
        for x in to_remove:
            for p in primes:
                counts[p][residues[p][x]] -= 1
                
        # Optional: check if we should stop early if k becomes very small?
        # No, we want the largest possible.
        
    # Construct result
    result = [i for i in range(N + 1) if active[i]]
    return result

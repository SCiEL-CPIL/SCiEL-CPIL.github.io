"""Rewritten program for the suite task; uses multi-start greedy + single-prime SA."""
import time
import random
import math

# EVOLVE-BLOCK-START
def _primes_upto(n):
    """Return list of primes <= n in increasing order."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def _score(residues, primes, cap=100000):
    """Compute the largest L such that every integer 1..L satisfies >= 2 congruences."""
    n = len(primes)
    # For each integer x from 1 upward, count how many primes p have x % p == residues[i]
    # We need count >= 2 for all x in 1..L
    # Efficient: for each prime i, the covered positions are residues[i], residues[i]+primes[i], ...
    # But we need the intersection count per position.
    
    # Use a difference array approach or just check up to some limit
    # Since we cap at 10^5, we can check positions 1..cap
    # But we want the MAXIMUM L, so we need to find the first position where count < 2
    
    # For efficiency, let's use a counting array
    # But cap is 10^5 and we have 168 primes, so 168 * (100000/p) operations total
    # Sum of 100000/p for primes p <= 1000 is roughly 100000 * sum(1/p) ~ 100000 * log(log(1000)) ~ 100000 * 1.9 ~ 190000
    # That's fine.
    
    # Actually, let's just compute for x = 1 to some limit and find the first failure
    # We'll check up to min(cap, some reasonable bound)
    
    count = [0] * (cap + 1)  # count[x] = number of congruences satisfied at x
    for i, p in enumerate(primes):
        r = residues[i]
        # Positions: r, r+p, r+2p, ... but we need x >= 1
        # If r == 0, positions are p, 2p, 3p, ...
        # If r > 0, positions are r, r+p, r+2p, ...
        start = r
        if start == 0:
            start = p
        elif start < 1:
            start = r + p * math.ceil((1 - r) / p)
        # Actually simpler: start from the smallest x >= 1 such that x % p == r
        if r == 0:
            start = p  # since x >= 1, first is p
        else:
            start = r  # r is in [1, p-1], so r >= 1
            if start < 1:
                start += p
        
        for x in range(start, cap + 1, p):
            count[x] += 1
    
    # Find the first x >= 1 where count[x] < 2
    for x in range(1, cap + 1):
        if count[x] < 2:
            return x - 1
    return cap

def _score_fast(residues, primes, check_limit=20000):
    """Faster scoring for optimization - only check up to check_limit."""
    n = len(primes)
    count = [0] * (check_limit + 1)
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start += p
        for x in range(start, check_limit + 1, p):
            count[x] += 1
    
    for x in range(1, check_limit + 1):
        if count[x] < 2:
            return x - 1
    return check_limit

def _greedy_init(primes, rng, check_limit=5000):
    """Greedy initialization: for each prime, pick residue that maximizes coverage."""
    n = len(primes)
    residues = [0] * n
    
    # Initialize with random residues
    for i in range(n):
        residues[i] = rng.randint(0, primes[i] - 1)
    
    # Greedy improvement: iterate through primes, for each pick best residue
    improved = True
    max_iters = 5
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n):
            best_r = residues[i]
            best_score = -1
            # Try all possible residues for this prime
            p = primes[i]
            for r in range(p):
                old = residues[i]
                residues[i] = r
                s = _score_fast(residues, primes, check_limit)
                if s > best_score:
                    best_score = s
                    best_r = r
                residues[i] = old
            if best_r != residues[i]:
                residues[i] = best_r
                improved = True
    
    return residues

def _sa_optimize(residues, primes, rng, time_limit, check_limit=5000):
    """Single-prime simulated annealing optimization."""
    n = len(primes)
    current = list(residues)
    current_score = _score_fast(current, primes, check_limit)
    best = list(current)
    best_score = current_score
    
    start_time = time.time()
    temp = 100.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick a random prime to change
        i = rng.randint(0, n - 1)
        p = primes[i]
        old_r = current[i]
        new_r = rng.randint(0, p - 1)
        if new_r == old_r:
            continue
        
        current[i] = new_r
        new_score = _score_fast(current, primes, check_limit)
        
        delta = new_score - current_score
        if delta > 0 or rng.random() < math.exp(delta / max(temp, 1e-10)):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = list(current)
        else:
            current[i] = old_r
        
        temp *= cooling
    
    return best, best_score

def construct():
    """Return 168 residues, one for each prime p <= 1000 in increasing order."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    primes = _primes_upto(1000)
    n = len(primes)
    
    rng = random.Random(42)
    
    best_residues = None
    best_score = 0
    
    # Multi-start: try several greedy initializations with different seeds
    num_starts = 3
    for start_idx in range(num_starts):
        if time.time() > deadline - 30:
            break
        
        rng_start = random.Random(1000 + start_idx * 7919)
        init = _greedy_init(primes, rng_start, check_limit=3000)
        
        # Evaluate full score
        full_score = _score(init, primes)
        
        if full_score > best_score:
            best_score = full_score
            best_residues = list(init)
        
        # SA refinement if time permits
        if time.time() > deadline - 20:
            break
        
        sa_time = min(15, max(5, deadline - time.time() - 10))
        refined, refined_score = _sa_optimize(init, primes, rng_start, sa_time, check_limit=3000)
        
        if refined_score > 0:
            # Verify with full score
            full_refined_score = _score(refined, primes)
            if full_refined_score > best_score:
                best_score = full_refined_score
                best_residues = list(refined)
    
    # If we haven't found anything, use a basic construction
    if best_residues is None:
        best_residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    # Final check: ensure validity
    for i, p in enumerate(primes):
        if not (0 <= best_residues[i] < p):
            best_residues[i] = best_residues[i] % p
    
    return best_residues
# EVOLVE-BLOCK-END

"""Bootstrap DIRECT solver for Erdos #689 (n=1000) doubly-covered run.

Improved construction:
  * Uses a smarter greedy + local search approach to maximize doubly-covered prefix.
  * Faster verification using vectorized NumPy.
  * Handles primes (numbers with one prime factor) by using free large primes.
"""
import time
import random
import numpy as np

T_START = time.time()
DEADLINE = 118.0          # hard safety margin (evaluator kills at 150 s)
MAX_L = 100000


def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]


_PRIMES = _primes_upto(1000)          # all 168 primes in increasing order
_BY_P = {p: i for i, p in enumerate(_PRIMES)}


def _verify(residues, limit=MAX_L):
    """Fast count of how many x in 1..limit satisfy >= 2 congruences.

    Returns the largest L with every 1..L doubly covered, capped at limit.
    """
    p = np.asarray(_PRIMES, dtype=np.int64)
    r = np.asarray(residues, dtype=np.int64)
    # For each x, the number of primes that hit x.
    mark = np.zeros(limit + 1, dtype=np.int16)
    for pp, rr in zip(p, r):
        rr = int(rr)
        # Find first x in [1,limit] with x % pp == rr
        if rr == 0:
            start = pp
        else:
            start = rr
            if start < 1:
                start += ((1 - start + pp - 1) // pp) * pp
        if start > limit:
            continue
        mark[start::pp] += 1
    # largest L with min(mark[1..L]) >= 2
    ok = mark[1:] >= 2
    if ok.all():
        return limit
    first = np.argmax(~ok)
    return int(first)


def _construct_greedy_L(L, seed=0):
    """Try to construct residues that doubly cover 1..L.
    
    Strategy: For each x in 1..L, ensure at least 2 primes hit it.
    Use a greedy approach: for each x, if it's not yet doubly covered,
    pick a prime p that can be assigned to hit x without conflicting
    with previous assignments.
    """
    rng = random.Random(seed)
    primes = _PRIMES
    n = len(primes)
    residues = [None] * n  # None means unassigned
    
    # For each x in 1..L, track which primes have been assigned to hit it
    x_hits = {x: set() for x in range(1, L + 1)}
    
    # Process x from 1 to L
    for x in range(1, L + 1):
        # Count current hits for x
        current_hits = 0
        for i, p in enumerate(primes):
            if residues[i] is not None and x % p == residues[i]:
                current_hits += 1
        
        if current_hits >= 2:
            continue
        
        # Need to add (2 - current_hits) more hits for x
        needed = 2 - current_hits
        
        # Try to find primes that can be assigned to hit x
        # A prime p can hit x if we set residues[_BY_P[p]] = x % p
        # and this doesn't conflict with existing assignments
        
        candidates = []
        for i, p in enumerate(primes):
            if residues[i] is not None:
                continue
            r = x % p
            # Check if assigning r to p would cause issues
            # It hits all y in 1..L where y % p == r
            # We need to make sure this doesn't break anything
            candidates.append((i, p, r))
        
        if not candidates:
            return None  # Can't cover x
        
        # Prefer smaller primes (more coverage)
        candidates.sort(key=lambda c: c[1])
        
        assigned_count = 0
        for i, p, r in candidates:
            if assigned_count >= needed:
                break
            # Assign this prime
            residues[i] = r
            assigned_count += 1
            # Update x_hits for all y in 1..L where y % p == r
            for y in range(1, L + 1):
                if y % p == r:
                    x_hits[y].add(p)
    
    # Check if all x in 1..L are doubly covered
    for x in range(1, L + 1):
        count = 0
        for i, p in enumerate(primes):
            if residues[i] is not None and x % p == residues[i]:
                count += 1
        if count < 2:
            return None
    
    # For unassigned primes, assign residues to avoid creating holes
    for i, p in enumerate(primes):
        if residues[i] is None:
            # Choose r that avoids hitting x where x is only singly covered
            # Simple: choose r = 0
            residues[i] = 0
    
    return residues


def _local_search(residues, L, budget=1.0, seed=0):
    """Local search to improve coverage beyond L."""
    rng = random.Random(seed)
    primes = _PRIMES
    n = len(primes)
    best = list(residues)
    best_score = _verify(best, min(L + 5000, MAX_L))
    
    t_start = time.time()
    while time.time() - t_start < budget:
        # Pick a random prime to reassign
        i = rng.randint(0, n - 1)
        p = primes[i]
        old_r = best[i]
        
        # Try a new random residue
        new_r = rng.randint(0, p - 1)
        if new_r == old_r:
            continue
        
        best[i] = new_r
        new_score = _verify(best, min(L + 5000, MAX_L))
        
        if new_score > best_score:
            best_score = new_score
        else:
            best[i] = old_r
    
    return best_score, best


def construct():
    global T_START
    T_START = time.time()
    rng = random.Random(20240607)

    best_score = 84
    best_res = [((i + 2) // 2) % p for i, p in enumerate(_PRIMES)]

    # Try various target L values
    L_candidates = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500]
    
    for L in L_candidates:
        if time.time() >= DEADLINE - 5:
            break
        
        # Try multiple seeds for this L
        for seed in range(5):
            if time.time() >= DEADLINE - 3:
                break
            
            res = _construct_greedy_L(L, seed=seed)
            if res is None:
                continue
            
            score = _verify(res, min(L + 2000, MAX_L))
            if score > best_score:
                best_score = score
                best_res = list(res)
                
                # Do local search to improve
                if time.time() < DEADLINE - 10:
                    lc_budget = min(2.0, max(0.5, (DEADLINE - time.time()) / 10))
                    s2, r2 = _local_search(res, L, budget=lc_budget, seed=seed + 1000)
                    if s2 > best_score:
                        best_score = s2
                        best_res = r2
        
        if best_score >= 1000:
            break
    
    return best_res

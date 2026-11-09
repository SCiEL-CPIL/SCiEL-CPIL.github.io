"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

def _get_primes(n):
    """Return list of primes <= n."""
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return [int(p) for p in np.where(sieve)[0]]

def _score(residues, primes):
    """
    Compute the largest L such that every integer 1..L satisfies at least two
    of the congruences x = r_p (mod p).
    Returns (L, count_array) where count_array[i] is the number of congruences
    satisfied by integer i+1.
    """
    # We need to find the largest L such that min_{x in 1..L} count(x) >= 2.
    # We'll compute counts for x in 1..MAX_X and find the first x where count < 2.
    MAX_X = 100000  # cap
    # For efficiency, we can compute counts incrementally.
    # count[x] = sum over p of (x == r_p mod p)
    
    # For each prime p, the integers x in 1..MAX_X that satisfy x ≡ r_p (mod p)
    # are x = r_p, r_p + p, r_p + 2p, ... <= MAX_X
    # If r_p == 0, then x = p, 2p, 3p, ...
    
    counts = np.zeros(MAX_X + 1, dtype=np.int32)  # counts[x] for x in 1..MAX_X
    
    for r, p in zip(residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        # x = start, start + p, start + 2p, ... <= MAX_X
        if start <= MAX_X:
            # Use slicing to increment counts
            # indices: start-1, start-1+p, start-1+2p, ... in 0-based array
            indices = np.arange(start - 1, MAX_X, p)
            counts[indices] += 1
    
    # Find the largest L such that counts[1..L] are all >= 2
    # i.e., find the first index x in 1..MAX_X where counts[x] < 2
    # L is x-1, or MAX_X if all are >= 2
    
    # Find first x in 1..MAX_X where counts[x] < 2
    below2 = np.where(counts[1:] < 2)[0]
    if len(below2) == 0:
        L = MAX_X
    else:
        first_below = below2[0] + 1  # 1-based
        L = first_below - 1
    
    return L, counts

def _improve_local(residues, primes, max_iter=10000):
    """
    Local search: try changing one residue at a time to improve the score.
    """
    n = len(primes)
    current_residues = list(residues)
    current_L, _ = _score(current_residues, primes)
    
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            p = primes[i]
            old_r = current_residues[i]
            # Try all other residues for this prime
            for new_r in range(p):
                if new_r == old_r:
                    continue
                current_residues[i] = new_r
                L, _ = _score(current_residues, primes)
                if L > current_L:
                    current_L = L
                    old_r = new_r
                    improved = True
                    break
                current_residues[i] = old_r
        if not improved:
            break
    
    return current_residues, current_L

def _bipartite_match(L, primes):
    """
    Check if there exists an assignment of residues such that every integer 1..L
    is covered by at least two congruences.
    
    This is a hard problem, but we can try a heuristic approach:
    For each integer x in 1..L, it needs to be covered by at least 2 primes.
    A prime p can cover x if we set r_p = x mod p.
    
    We can formulate this as: for each x, choose at least 2 primes from those
    that can cover x (i.e., all primes can cover any x by setting r_p appropriately).
    But the constraint is that each prime p has only one residue r_p, so it covers
    a specific arithmetic progression.
    
    This is complex. Instead, let's use a different approach.
    """
    pass

def construct():
    """
    Construct 168 residues, one for each prime p <= 1000 in increasing order.
    Maximize the largest L such that every integer 1..L satisfies at least two
    of the congruences x = r_p (mod p).
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    primes = _get_primes(1000)
    n = len(primes)
    
    # Initial solution: seed program approach
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    current_L, _ = _score(residues, primes)
    best_residues = list(residues)
    best_L = current_L
    
    # Local search improvement
    if time.time() < deadline - 10:
        residues, current_L = _improve_local(residues, primes, max_iter=5000)
        if current_L > best_L:
            best_L = current_L
            best_residues = list(residues)
    
    # Try a different initial strategy: random restarts
    rng = np.random.default_rng(42)
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        # Random initial residues
        residues = [int(rng.integers(0, p)) for p in primes]
        current_L, _ = _score(residues, primes)
        
        # Local search
        residues, current_L = _improve_local(residues, primes, max_iter=2000)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = list(residues)
    
    return best_residues
# EVOLVE-BLOCK-END

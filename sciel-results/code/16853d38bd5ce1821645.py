"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _get_primes():
    """Get all primes up to 1000."""
    sieve = [True] * 1001
    sieve[0] = sieve[1] = False
    for i in range(2, int(1000**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, 1001, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

PRIMES = _get_primes()
N_PRIMES = len(PRIMES)  # 168
MAX_L = 100000

def _build_counts(residues, limit=MAX_L):
    """Build count array for given residues up to limit."""
    counts = [0] * (limit + 1)
    for i, p in enumerate(PRIMES):
        r = residues[i]
        start = r if r >= 1 else p
        for x in range(start, limit + 1, p):
            counts[x] += 1
    return counts

def _find_L(counts, limit=MAX_L):
    """Find longest prefix where all counts >= 2."""
    L = 0
    for x in range(1, limit + 1):
        if counts[x] >= 2:
            L = x
        else:
            break
    return L

def _score(residues, limit=MAX_L):
    """Score = longest L where every integer 1..L is doubly covered."""
    counts = _build_counts(residues, limit)
    return _find_L(counts, limit)

def _initial_assignment():
    """Create a good initial assignment using a structured approach."""
    residues = [0] * N_PRIMES
    # Try to cover early integers densely
    # For each prime, choose residue to help cover integers in a target range
    # Simple: cycle through residues to spread coverage
    for i, p in enumerate(PRIMES):
        # Cover integer around position i+1
        residues[i] = (i + 1) % p
    return residues

def _hill_climb(residues, time_limit, rng, limit=MAX_L):
    """Simple hill climbing with restarts."""
    counts = _build_counts(residues, limit)
    current_L = _find_L(counts, limit)
    best_residues = residues[:]
    best_L = current_L
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    while time.time() < deadline:
        # Pick random prime
        i = rng.randint(0, N_PRIMES - 1)
        p = PRIMES[i]
        old_r = residues[i]
        new_r = rng.randint(0, p - 1)
        if new_r == old_r:
            continue
        
        # Update counts
        old_start = old_r if old_r >= 1 else p
        for x in range(old_start, limit + 1, p):
            counts[x] -= 1
        new_start = new_r if new_r >= 1 else p
        for x in range(new_start, limit + 1, p):
            counts[x] += 1
        
        # Find new L efficiently
        # Check if current L is still valid
        new_L = current_L
        # Extend
        while new_L < limit and counts[new_L + 1] >= 2:
            new_L += 1
        # If we can't extend, check if L broke
        if new_L < current_L:
            # Re-scan from 1
            new_L = 0
            for x in range(1, limit + 1):
                if counts[x] >= 2:
                    new_L = x
                else:
                    break
        
        if new_L >= current_L:
            current_L = new_L
            residues[i] = new_r
            if new_L > best_L:
                best_L = new_L
                best_residues = residues[:]
        else:
            # Undo
            for x in range(new_start, limit + 1, p):
                counts[x] -= 1
            for x in range(old_start, limit + 1, p):
                counts[x] += 1
            residues[i] = old_r
    
    return best_residues, best_L

def construct():
    """Construct 168 residues, one for each prime p <= 1000."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    rng = random.Random(12345)
    
    # Start with initial assignment
    residues = _initial_assignment()
    best_residues = residues[:]
    best_score = _score(best_residues)
    
    # Hill climb on initial
    remaining = deadline - time.time()
    if remaining > 5:
        hc_best, hc_score = _hill_climb(best_residues, min(remaining * 0.4, 20), rng)
        if hc_score > best_score:
            best_score = hc_score
            best_residues = hc_best[:]
    
    # Multiple restarts
    while time.time() < deadline - 3:
        # Perturb from best
        perturbed = best_residues[:]
        n_perturb = rng.randint(5, 25)
        for _ in range(n_perturb):
            i = rng.randint(0, N_PRIMES - 1)
            p = PRIMES[i]
            perturbed[i] = rng.randint(0, p - 1)
        
        remaining_time = deadline - time.time() - 3
        if remaining_time < 2:
            break
        
        search_time = min(remaining_time * 0.5, 10)
        hc_best, hc_score = _hill_climb(perturbed, search_time, rng)
        
        if hc_score > best_score:
            best_score = hc_score
            best_residues = hc_best[:]
    
    return best_residues
# EVOLVE-BLOCK-END

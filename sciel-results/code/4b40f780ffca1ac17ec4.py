"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _primes_up_to(n):
    """Return list of primes <= n."""
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _build_counts(residues, primes, L=100000):
    """
    Build coverage count array for integers 1..L.
    residues[i] is the residue for primes[i].
    Returns numpy array of shape (L+1,) where counts[x] = number of congruences satisfied by x.
    """
    L = min(L, 100000)
    counts = np.zeros(L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        # x ≡ r (mod p) for x in 1..L
        # First x >= 1 with x ≡ r (mod p):
        if r == 0:
            start = p
        else:
            start = r
        # Adjust to be >= 1
        if start < 1:
            start += p * ((1 - start + p - 1) // p)
        if start > L:
            continue
        counts[start::p] += 1
    return counts

def _compute_L(counts):
    """
    Given coverage counts for 1..L, find the largest L such that every integer 1..L has count >= 2.
    """
    # Find first index where count < 2
    if len(counts) == 0:
        return 0
    # counts[0] is unused (we track 1..L)
    bad = np.where(counts[1:] < 2)[0]
    if len(bad) == 0:
        return len(counts) - 1
    return bad[0]  # first bad index in 1..L is bad[0]+1, so L = bad[0]

def _score(residues, primes, L_limit=100000):
    """Compute the score: largest L such that every integer 1..L has count >= 2."""
    counts = _build_counts(residues, primes, L_limit)
    return _compute_L(counts)

def construct():
    """
    Return 168 residues, one for each prime p <= 1000 in increasing order, with 0 <= r_p < p.
    Score is the largest L such that every integer 1..L satisfies at least two congruences.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    primes = _primes_up_to(1000)
    n_primes = len(primes)  # 168
    
    # Initial solution: use the seed approach
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    best_residues = residues[:]
    best_L = _score(residues, primes)
    
    # Simulated annealing / local search
    # Objective: maximize L (the prefix length where all have count >= 2)
    # We can also track the "vulnerable" set: integers with count exactly 1
    
    rng = random.Random(42)
    
    # Build initial counts
    counts = _build_counts(residues, primes, 100000)
    current_L = _compute_L(counts)
    
    # Temperature schedule
    T = 10.0
    T_min = 0.1
    cooling = 0.9999
    
    # Precompute for fast evaluation
    # For each prime p and residue r, we know which positions in 1..L are covered
    # Changing residue from r_old to r_new for prime p:
    #   - decrement counts at positions congruent to r_old mod p
    #   - increment counts at positions congruent to r_new mod p
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Pick a random prime to modify
        i = rng.randrange(n_primes)
        p = primes[i]
        old_r = residues[i]
        
        # Pick a new residue
        new_r = rng.randrange(p)
        if new_r == old_r:
            continue
        
        # Compute the effect of this change on the coverage
        # Positions affected: those congruent to old_r mod p (decrement) and new_r mod p (increment)
        
        # Find positions where old_r mod p holds in 1..L
        # And positions where new_r mod p holds
        
        # For efficiency, we only care about the region up to some horizon
        # Let's track the current L and a bit beyond
        
        # Get the range of interest: from 1 to max(current_L + 1000, 10000)
        horizon = min(100000, max(current_L + 2000, 10000))
        
        # Find affected positions
        # Positions with old_r mod p in [1, horizon]
        if old_r == 0:
            old_start = p
        else:
            old_start = old_r
        if old_start < 1:
            old_start += p * ((1 - old_start + p - 1) // p)
        
        old_positions = []
        if old_start <= horizon:
            old_positions = list(range(old_start, horizon + 1, p))
        
        # Positions with new_r mod p in [1, horizon]
        if new_r == 0:
            new_start = p
        else:
            new_start = new_r
        if new_start < 1:
            new_start += p * ((1 - new_start + p - 1) // p)
        
        new_positions = []
        if new_start <= horizon:
            new_positions = list(range(new_start, horizon + 1, p))
        
        # Apply the change tentatively
        # Decrement old positions, increment new positions
        for pos in old_positions:
            counts[pos] -= 1
        for pos in new_positions:
            counts[pos] += 1
        
        # Compute new L
        new_L = _compute_L(counts)
        
        # Acceptance criterion
        if new_L >= current_L:
            # Accept: improved or equal
            current_L = new_L
            residues[i] = new_r
            if current_L > best_L:
                best_L = current_L
                best_residues = residues[:]
        else:
            # Reject: revert
            for pos in new_positions:
                counts[pos] -= 1
            for pos in old_positions:
                counts[pos] += 1
            
            # Simulated annealing: accept with probability exp(-delta/T)
            delta = current_L - new_L
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    # Accept anyway
                    for pos in old_positions:
                        counts[pos] -= 1
                    for pos in new_positions:
                        counts[pos] += 1
                    current_L = new_L
                    residues[i] = new_r
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues[:]
        
        # Cool down
        T = max(T_min, T * cooling)
        
        # Periodic restart if stuck
        if iterations % 5000 == 0 and current_L < best_L:
            # Reset to best
            residues = best_residues[:]
            counts = _build_counts(residues, primes, 100000)
            current_L = _compute_L(counts)
            T = max(T, 5.0)
    
    return best_residues

# EVOLVE-BLOCK-END

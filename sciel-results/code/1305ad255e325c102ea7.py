"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _primes_up_to(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _compute_L(residues, primes, max_check=100000):
    """Compute the largest L such that every integer 1..L is covered by at least 2 congruences."""
    n_primes = len(primes)
    # Build coverage array for 1..max_check
    cov = np.zeros(max_check + 1, dtype=np.int32)
    for idx, (r, p) in enumerate(zip(residues, primes)):
        # Positions x in [1, max_check] with x ≡ r (mod p)
        # First position >= 1: if r == 0, first is p; else first is r if r >= 1
        first = r if r >= 1 else p
        if first > max_check:
            continue
        # Generate all positions: first, first+p, first+2p, ... <= max_check
        positions = np.arange(first, max_check + 1, p)
        cov[positions] += 1
    # Find largest L such that cov[1..L] >= 2
    # cov[i] for i in 1..max_check
    if cov[1] < 2:
        return 0
    # Find first index where cov < 2
    below = np.where(cov[1:max_check + 1] < 2)[0]
    if len(below) == 0:
        return max_check
    return below[0]

def _greedy_init(primes, search_range=2000):
    """Greedy: for each prime, pick residue maximizing newly-doubled coverage in frontier."""
    n_primes = len(primes)
    residues = [0] * n_primes
    # Track coverage counts for positions 1..search_range
    cov = np.zeros(search_range + 1, dtype=np.int32)
    for idx, p in enumerate(primes):
        best_r = 0
        best_score = -1
        for r in range(p):
            # Compute how many positions in [1, search_range] would go from 1 to 2+ coverage
            first = r if r >= 1 else p
            if first > search_range:
                gain = 0
            else:
                positions = np.arange(first, search_range + 1, p)
                # Gain = number of positions with cov == 1
                gain = np.sum(cov[positions] == 1)
            # Also penalize if it breaks double coverage (cov goes from 2+ to 1)
            if gain > best_score or (gain == best_score and r < best_r):
                best_score = gain
                best_r = r
        residues[idx] = best_r
        # Update coverage
        first = best_r if best_r >= 1 else p
        if first <= search_range:
            positions = np.arange(first, search_range + 1, p)
            cov[positions] += 1
    return residues

def _local_search(residues, primes, max_iter=5000, time_limit=100):
    """Hill-climbing local search: try changing one residue at a time."""
    n_primes = len(primes)
    start_time = time.time()
    current_L = _compute_L(residues, primes)
    best_residues = list(residues)
    best_L = current_L
    no_improve = 0
    max_no_improve = 50
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        if no_improve >= max_no_improve:
            break
        
        # Pick a random prime index to try changing
        idx = np.random.randint(0, n_primes)
        p = primes[idx]
        old_r = residues[idx]
        
        # Try a few random residues
        tried = set()
        improved = False
        for _ in range(5):
            if time.time() - start_time > time_limit:
                break
            new_r = np.random.randint(0, p)
            if new_r == old_r or new_r in tried:
                continue
            tried.add(new_r)
            
            residues[idx] = new_r
            new_L = _compute_L(residues, primes)
            if new_L > current_L:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_residues = list(residues)
                improved = True
                no_improve = 0
                break
            else:
                residues[idx] = old_r
        
        if not improved:
            no_improve += 1
    
    return best_residues, best_L

def construct():
    start_time = time.time()
    deadline = 110  # safety margin before 120s limit
    
    primes = _primes_up_to(1000)
    n_primes = len(primes)
    
    # Phase 1: Greedy initialization
    residues = _greedy_init(primes, search_range=2000)
    current_L = _compute_L(residues, primes)
    best_residues = list(residues)
    best_L = current_L
    
    # Phase 2: Local search
    if time.time() - start_time < deadline:
        best_residues, best_L = _local_search(residues, primes, 
                                               max_iter=3000, 
                                               time_limit=deadline - (time.time() - start_time))
    
    # Phase 3: If time remains, do another round of local search from best
    if time.time() - start_time < deadline - 10:
        remaining = deadline - (time.time() - start_time)
        best_residues2, best_L2 = _local_search(best_residues, primes,
                                                 max_iter=2000,
                                                 time_limit=remaining)
        if best_L2 > best_L:
            best_residues = best_residues2
            best_L = best_L2
    
    return best_residues
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def _primes_up_to(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0:2] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return [int(p) for p in np.where(sieve)[0]]

def _compute_counts(residues, primes, L):
    """Compute coverage counts for integers 1..L using vectorized approach."""
    counts = np.zeros(L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        # Find all x in [1, L] such that x ≡ r (mod p)
        # x = r + k*p for k >= 0, and 1 <= x <= L
        if r == 0:
            # x = p, 2p, 3p, ...
            start = p
        else:
            start = r
            if start < 1:
                start += p * ((1 - start + p - 1) // p)
        if start > L:
            continue
        # Generate x values: start, start+p, start+2p, ... <= L
        k_max = (L - start) // p
        x_vals = start + np.arange(k_max + 1) * p
        counts[x_vals] += 1
    return counts

def _score(residues, primes, L_max=100000):
    """Find the largest L such that every integer 1..L has count >= 2."""
    # Start with a reasonable upper bound
    # We can binary search, but first let's try a direct approach
    # For efficiency, compute counts up to some limit
    L = 0
    # Try progressively larger L
    for L in range(1, min(L_max + 1, 2000)):
        counts = _compute_counts(residues, primes, L)
        if np.all(counts[1:] >= 2):
            continue
        else:
            return L - 1
    # If we made it to L_max, check further
    return min(L_max, 2000)

def _greedy_construct(primes):
    """Greedy: for each prime, pick residue that maximizes coverage of prefix."""
    n = len(primes)
    residues = [0] * n
    # Start with all zeros
    for i in range(n):
        p = primes[i]
        # For each possible residue, compute how many integers 1..L are covered
        # by this residue class, given current assignments for primes < i
        # We want to maximize the doubly-covered prefix
        best_r = 0
        best_score = -1
        # Limit search to residues that might help
        for r in range(p):
            residues[i] = r
            # Compute score - this is expensive, so limit
            # Use a smaller L for greedy
            L_test = 200
            counts = _compute_counts(residues, primes, L_test)
            # Find largest prefix with count >= 2
            prefix = 0
            for x in range(1, L_test + 1):
                if counts[x] >= 2:
                    prefix = x
                else:
                    break
            if prefix > best_score:
                best_score = prefix
                best_r = r
        residues[i] = best_r
    return residues

def _local_search(residues, primes, time_limit=100.0):
    """Local search: try changing one residue at a time to improve score."""
    n = len(primes)
    start_time = time.time()
    best_residues = residues[:]
    best_score = _score(best_residues, primes)
    
    # Precompute primes list
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit - 10:
                break
            p = primes[i]
            old_r = best_residues[i]
            # Try a subset of residues (not all, for speed)
            candidates = [old_r]
            # Add some random candidates
            for _ in range(5):
                candidates.append(np.random.randint(0, p))
            # Also try residues near the "target" integer
            # For prime i, the target integer is roughly i/2 + 1
            target = (i + 2) // 2
            for dr in [-2, -1, 0, 1, 2]:
                candidates.append((target + dr) % p)
            
            for r in candidates:
                if r == old_r:
                    continue
                best_residues[i] = r
                score = _score(best_residues, primes)
                if score > best_score:
                    best_score = score
                    old_r = r
                    improved = True
                else:
                    best_residues[i] = old_r
        
        if not improved:
            # Try random restarts within local search
            for i in range(n):
                if time.time() - start_time > time_limit - 10:
                    break
                best_residues[i] = np.random.randint(0, primes[i])
            best_score = _score(best_residues, primes)
    
    return best_residues, best_score

def construct():
    primes = _primes_up_to(1000)
    n = len(primes)
    
    # Start with greedy construction
    residues = _greedy_construct(primes)
    score = _score(residues, primes)
    
    # Local search with time limit
    residues, score = _local_search(residues, primes, time_limit=100.0)
    
    return residues
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve_primes(limit):
    """Return list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _score(residues, primes, max_check=100000):
    """Compute the largest L such that every integer 1..L satisfies at least two congruences."""
    n = len(primes)
    # Build a coverage array for 1..max_check
    # For each prime p with residue r, it covers numbers r, r+p, r+2p, ... in [1, max_check]
    # We need count >= 2 for all x in 1..L
    
    # Efficient approach: create an array of zeros, then for each prime add 1 at positions it covers
    # But 168 * 100000 = 16.8M operations, which is fine with numpy
    
    coverage = np.zeros(max_check + 1, dtype=np.int32)
    for i, (p, r) in enumerate(zip(primes, residues)):
        # Find all x in [1, max_check] such that x % p == r
        # First x >= 1: if r == 0, first is p; else first is r (if r >= 1) or p (if r == 0)
        if r == 0:
            first = p
        else:
            first = r
        if first > max_check:
            continue
        # Mark all positions first, first+p, first+2p, ...
        positions = np.arange(first, max_check + 1, p)
        coverage[positions] += 1
    
    # Find the largest L such that coverage[1..L] >= 2
    # Find first position where coverage < 2
    below_two = np.where(coverage[1:] < 2)[0]
    if len(below_two) == 0:
        return max_check
    return below_two[0]  # This is 0-indexed offset from 1, so L = below_two[0]

def _score_fast(residues, primes, max_check=100000):
    """Faster scoring using vectorized approach."""
    n = len(primes)
    coverage = np.zeros(max_check + 1, dtype=np.int32)
    
    for i in range(n):
        p = primes[i]
        r = residues[i]
        if r == 0:
            first = p
        else:
            first = r
        if first > max_check:
            continue
        # Use slice assignment for speed
        coverage[first::p] += 1
    
    # Find first position >= 1 where coverage < 2
    # We want the smallest x in [1, max_check] with coverage[x] < 2
    # If none, L = max_check
    mask = coverage[1:] < 2
    if not mask.any():
        return max_check
    first_fail = np.argmax(mask)
    return first_fail  # L = first_fail (since mask is 0-indexed from position 1)

def _initial_residues(primes):
    """Generate initial residues using a heuristic."""
    # Strategy: try to spread coverage evenly
    # Use a greedy approach: for each prime, pick residue that maximizes coverage of early integers
    n = len(primes)
    residues = [0] * n
    
    # Start with a simple initialization
    for i, p in enumerate(primes):
        residues[i] = (i + 1) % p
    
    return residues

def _find_first_gap(residues, primes, max_check=100000):
    """Find the first integer >= 1 with coverage < 2."""
    coverage = np.zeros(max_check + 1, dtype=np.int32)
    for i in range(len(primes)):
        p = primes[i]
        r = residues[i]
        if r == 0:
            first = p
        else:
            first = r
        if first > max_check:
            continue
        coverage[first::p] += 1
    
    mask = coverage[1:] < 2
    if not mask.any():
        return -1, coverage
    first_fail = np.argmax(mask)
    return first_fail, coverage

def _try_improve(residues, primes, target_x, max_check=100000):
    """Try to change a residue to cover target_x. Returns new residues if improved, else None."""
    n = len(primes)
    best_residues = None
    best_score = -1
    
    # For each prime, try setting its residue to target_x % p
    # This would make that prime cover target_x
    # We want to find a change that maximizes the score
    
    # First, compute current coverage
    coverage = np.zeros(max_check + 1, dtype=np.int32)
    for i in range(n):
        p = primes[i]
        r = residues[i]
        if r == 0:
            first = p
        else:
            first = r
        if first > max_check:
            continue
        coverage[first::p] += 1
    
    current_score = _score_fast(residues, primes, max_check)
    
    # Try changing each prime's residue to target_x % p
    # But only if that prime currently doesn't cover target_x
    for i in range(n):
        p = primes[i]
        new_r = target_x % p
        if new_r == residues[i]:
            continue  # Already covers it
        
        # Check if this prime currently covers target_x
        if (target_x - residues[i]) % p == 0:
            continue  # Already covers it, no point changing
        
        # Try this change
        old_r = residues[i]
        residues[i] = new_r
        new_score = _score_fast(residues, primes, max_check)
        
        if new_score > best_score:
            best_score = new_score
            best_residues = residues.copy()
        
        residues[i] = old_r  # Restore
    
    if best_residues is not None and best_score > current_score:
        return best_residues, best_score
    return None, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    primes = _sieve_primes(1000)
    n = len(primes)  # 168
    
    # Initial residues
    residues = _initial_residues(primes)
    current_score = _score_fast(residues, primes, 100000)
    
    best_residues = residues.copy()
    best_score = current_score
    
    # Hill climbing
    while time.time() < deadline:
        # Find first gap
        gap_pos, coverage = _find_first_gap(residues, primes, 100000)
        
        if gap_pos == -1:
            break  # No gap found, we've reached max_check
        
        target_x = gap_pos + 1  # Convert to actual integer
        
        # Try to improve
        new_residues, new_score = _try_improve(residues, primes, target_x, 100000)
        
        if new_residues is not None:
            residues = new_residues
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_residues = residues.copy()
        else:
            # No improvement found, try a different strategy
            # Random restart: perturb some residues
            improved = False
            for _ in range(10):
                if time.time() >= deadline:
                    break
                # Randomly change 1-3 residues
                num_changes = np.random.randint(1, 4)
                indices = np.random.choice(n, size=num_changes, replace=False)
                trial = residues.copy()
                for idx in indices:
                    trial[idx] = np.random.randint(0, primes[idx])
                
                trial_score = _score_fast(trial, primes, 100000)
                if trial_score > current_score:
                    residues = trial
                    current_score = trial_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = residues.copy()
                    break
            
            if not improved:
                # Try more aggressive random search
                for _ in range(5):
                    if time.time() >= deadline:
                        break
                    trial = residues.copy()
                    num_changes = np.random.randint(3, 10)
                    indices = np.random.choice(n, size=num_changes, replace=False)
                    for idx in indices:
                        trial[idx] = np.random.randint(0, primes[idx])
                    
                    trial_score = _score_fast(trial, primes, 100000)
                    if trial_score > current_score:
                        residues = trial
                        current_score = trial_score
                        if current_score > best_score:
                            best_score = current_score
                            best_residues = residues.copy()
                        break
    
    return best_residues
# EVOLVE-BLOCK-END

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
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_check
    return int(below_two[0])

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    
    # Initialize: use the seed strategy as starting point
    residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int32)
    
    best_residues = residues.copy()
    best_score = _score(residues, primes)
    
    # Local search: hill-climbing on individual residues
    # Strategy: for each prime, try all residues and keep the best improvement
    # Iterate multiple passes until no improvement or deadline
    
    max_passes = 50
    
    for iteration in range(max_passes):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try to improve by adjusting each prime's residue
        for i in range(n_primes):
            if time.time() > deadline:
                break
            
            p = primes[i]
            current_r = residues[i]
            best_r = current_r
            best_local_score = best_score
            
            # Try a subset of residues (all if p is small, sample if large)
            if p <= 100:
                candidates = list(range(p))
            else:
                # For larger primes, try current and some random candidates
                candidates = [current_r]
                if p <= 500:
                    candidates.extend(np.random.randint(0, p, size=min(20, p)).tolist())
                else:
                    candidates.extend(np.random.randint(0, p, size=10).tolist())
            
            for r in candidates:
                if r == current_r:
                    continue
                
                # Temporarily change residue
                old_r = residues[i]
                residues[i] = r
                
                new_score = _score(residues, primes)
                
                residues[i] = old_r
                
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_r = r
            
            if best_r != current_r:
                residues[i] = best_r
                best_score = best_local_score
                best_residues = residues.copy()
                improved = True
        
        if not improved:
            break
    
    # Final check
    final_score = _score(best_residues, primes)
    if final_score > best_score:
        best_score = final_score
    
    return best_residues.tolist()
# EVOLVE-BLOCK-END

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
    
    # Find the first x where counts[x] < 2
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_check
    return below_two[0]

def _score_fast(residues, primes, current_L):
    """Fast score: only check if we can extend beyond current_L."""
    max_check = 100000
    # First check if current_L is still valid
    counts = np.zeros(current_L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= current_L:
            counts[start::p] += 1
    
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) > 0:
        return below_two[0]
    
    # Current_L is valid, try to extend
    # Extend in chunks
    L = current_L
    while L < max_check:
        new_L = min(L + 1000, max_check)
        counts = np.zeros(new_L + 1, dtype=np.int32)
        for r, p in zip(residues, primes):
            start = r if r > 0 else p
            if start <= new_L:
                counts[start::p] += 1
        below_two = np.where(counts[1:] < 2)[0]
        if len(below_two) > 0:
            return below_two[0]
        L = new_L
    return L

def construct():
    primes = _sieve_primes(1000)
    n = len(primes)
    
    # Deadline guard - leave safety margin
    deadline = time.time() + 145
    
    # Smart initialization: for each prime, try to cover early numbers
    # Strategy: for small primes, choose residues that cover small numbers
    # For prime p, if we choose r=0, it covers p, 2p, 3p, ...
    # If we choose r=k (0<k<p), it covers k, k+p, k+2p, ...
    
    # Initial strategy: greedily assign residues to maximize coverage of 1..L
    # Start with all residues = 0, then improve
    best_residues = [0] * n
    best_score = _score(best_residues, primes)
    
    # Greedy improvement: for each position, try all residues and pick best
    # But that's 168 * avg_p ~ 168 * 500 = 84000 evaluations, too many
    # Instead, do multiple passes with random sampling
    
    np.random.seed(42)
    
    # Phase 1: Greedy local search with full residue tries for small primes
    for pass_num in range(5):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            p = primes[i]
            current_r = best_residues[i]
            
            # For small primes, try all residues
            # For large primes, try a subset
            if p <= 50:
                candidates = list(range(p))
            else:
                # Try current, neighbors, and random
                candidates = [current_r]
                for delta in range(-5, 6):
                    new_r = (current_r + delta) % p
                    if new_r not in candidates:
                        candidates.append(new_r)
                for _ in range(10):
                    new_r = np.random.randint(0, p)
                    if new_r not in candidates:
                        candidates.append(new_r)
            
            best_r = current_r
            best_s = best_score
            
            for new_r in candidates:
                if new_r == current_r:
                    continue
                
                test_residues = list(best_residues)
                test_residues[i] = new_r
                s = _score(test_residues, primes)
                
                if s > best_s:
                    best_s = s
                    best_r = new_r
            
            if best_r != current_r:
                best_residues[i] = best_r
                best_score = best_s
                improved = True
        
        if not improved:
            break
    
    # Phase 2: More aggressive random restarts and perturbations
    while time.time() < deadline - 5:
        # Try random perturbations of multiple residues
        num_perturb = np.random.randint(1, 6)
        test_residues = list(best_residues)
        
        for _ in range(num_perturb):
            i = np.random.randint(0, n)
            p = primes[i]
            test_residues[i] = np.random.randint(0, p)
        
        s = _score(test_residues, primes)
        
        if s > best_score:
            best_residues = test_residues
            best_score = s
    
    # Phase 3: Final local search refinement
    if time.time() < deadline - 2:
        for i in range(n):
            if time.time() > deadline:
                break
            
            p = primes[i]
            current_r = best_residues[i]
            
            # Try all residues for small primes, subset for large
            if p <= 30:
                candidates = list(range(p))
            else:
                candidates = [current_r]
                for delta in range(-10, 11):
                    new_r = (current_r + delta) % p
                    if new_r not in candidates:
                        candidates.append(new_r)
                for _ in range(20):
                    new_r = np.random.randint(0, p)
                    if new_r not in candidates:
                        candidates.append(new_r)
            
            best_r = current_r
            best_s = best_score
            
            for new_r in candidates:
                if new_r == current_r:
                    continue
                
                test_residues = list(best_residues)
                test_residues[i] = new_r
                s = _score(test_residues, primes)
                
                if s > best_s:
                    best_s = s
                    best_r = new_r
            
            if best_r != current_r:
                best_residues[i] = best_r
                best_score = best_s
    
    return best_residues
# EVOLVE-BLOCK-END

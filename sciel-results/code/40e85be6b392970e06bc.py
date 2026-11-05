"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _get_primes():
    """Return list of primes <= 1000."""
    sieve = [True] * 1001
    sieve[0] = sieve[1] = False
    for i in range(2, int(1000**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(2, 1001) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    primes = _get_primes()
    n = len(primes)
    
    # Use MAX_L = 10000 for speed; the target is L >= 1000
    MAX_L = 10000
    
    def build_coverage(residues):
        """Build the coverage array from scratch."""
        cov = np.zeros(MAX_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start <= MAX_L:
                cov[start:MAX_L+1:p] += 1
        return cov
    
    def find_first_gap(cov):
        """Find the largest L such that all 1..L have coverage >= 2."""
        bad_indices = np.where(cov[1:MAX_L+1] < 2)[0]
        if len(bad_indices) == 0:
            return MAX_L
        first_bad = bad_indices[0] + 1
        return first_bad - 1
    
    def update_coverage(cov, idx, old_r, new_r):
        """Update coverage in-place when changing residue at idx."""
        p = primes[idx]
        if old_r == 0:
            old_start = p
        else:
            old_start = old_r
        if old_start <= MAX_L:
            cov[old_start:MAX_L+1:p] -= 1
        if new_r == 0:
            new_start = p
        else:
            new_start = new_r
        if new_start <= MAX_L:
            cov[new_start:MAX_L+1:p] += 1
    
    best_L_global = 0
    best_residues_global = [0] * n
    
    # Multiple initializations to explore different basins
    def init_seed():
        return [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    def init_random():
        return [random.randint(0, p - 1) for p in primes]
    
    def init_perturbed():
        r = init_seed()
        for _ in range(10):
            idx = random.randint(0, n - 1)
            r[idx] = random.randint(0, primes[idx] - 1)
        return r
    
    initializers = [init_seed, init_perturbed, init_random, init_seed, init_perturbed]
    
    for init_func in initializers:
        if time.time() > deadline - 5:
            break
        
        residues = init_func()
        coverage = build_coverage(residues)
        current_L = find_first_gap(coverage)
        
        if current_L > best_L_global:
            best_L_global = current_L
            best_residues_global = residues[:]
        
        # Local search with shuffled order to avoid bias
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            prime_indices = list(range(n))
            random.shuffle(prime_indices)
            
            for idx in prime_indices:
                if time.time() > deadline - 5:
                    break
                
                p = primes[idx]
                old_r = residues[idx]
                best_r_for_idx = old_r
                best_L_for_idx = current_L
                
                # Generate targeted candidates
                candidates = set()
                candidates.add(old_r)
                
                # Target the current bottleneck: cover current_L + 1
                target = current_L + 1
                if target <= MAX_L:
                    target_r = target % p
                    candidates.add(target_r)
                
                # Try to fix the first gap by covering it
                # Also try residues that would cover positions near the gap
                for offset in range(-2, 3):
                    t = current_L + 1 + offset
                    if 1 <= t <= MAX_L:
                        candidates.add(t % p)
                
                # Add neighbors of current residue
                candidates.add((old_r + 1) % p)
                candidates.add((old_r - 1) % p)
                
                # Add random candidates
                num_random = min(20, p)
                for _ in range(num_random):
                    candidates.add(random.randint(0, p - 1))
                
                for r in candidates:
                    if r == old_r:
                        continue
                    if time.time() > deadline - 5:
                        break
                    
                    # Try this candidate
                    update_coverage(coverage, idx, old_r, r)
                    new_L = find_first_gap(coverage)
                    # Revert
                    update_coverage(coverage, idx, r, old_r)
                    
                    if new_L > best_L_for_idx:
                        best_L_for_idx = new_L
                        best_r_for_idx = r
                
                # Commit best change if found
                if best_r_for_idx != old_r:
                    update_coverage(coverage, idx, old_r, best_r_for_idx)
                    residues[idx] = best_r_for_idx
                    current_L = best_L_for_idx
                    improved = True
                    if current_L > best_L_global:
                        best_L_global = current_L
                        best_residues_global = residues[:]
        
        # End of local search for this initialization
    
    return [int(r) for r in best_residues_global]
# EVOLVE-BLOCK-END

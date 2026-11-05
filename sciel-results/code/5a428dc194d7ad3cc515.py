import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Generate primes up to 1000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    
    # We search for L up to a maximum limit. 
    # The problem asks for L >= 1000. 
    # Setting MAX_L too high slows down the 'find_first_gap' step.
    # 10,000 is a reasonable upper bound for this problem size (168 primes).
    MAX_L = 10000
    coverage = np.zeros(MAX_L + 2, dtype=np.int32)
    
    def build_coverage(residues):
        """Build the coverage array from scratch."""
        cov = np.zeros(MAX_L + 2, dtype=np.int32)
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
        """Find the largest L such that all 1..L have >= 2 hits."""
        # We want the first index i >= 1 where cov[i] < 2
        # cov[1:] corresponds to indices 1..MAX_L+1
        # np.where returns indices in the sliced array, so we add 1
        bad_indices = np.where(cov[1:MAX_L+1] < 2)[0]
        if len(bad_indices) == 0:
            return MAX_L
        first_bad = bad_indices[0] + 1
        return first_bad - 1

    def update_coverage(cov, idx, old_r, new_r):
        """
        Update coverage array in-place when changing residue at idx from old_r to new_r.
        Returns None.
        """
        p = primes[idx]
        
        # Remove old
        if old_r == 0:
            old_start = p
        else:
            old_start = old_r
        if old_start <= MAX_L:
            cov[old_start:MAX_L+1:p] -= 1
            
        # Add new
        if new_r == 0:
            new_start = p
        else:
            new_start = new_r
        if new_start <= MAX_L:
            cov[new_start:MAX_L+1:p] += 1

    best_L_global = 0
    best_residues_global = [0] * n_primes
    
    # Define initializations
    # 1. The "seed" heuristic from inspiration 1
    def init_seed():
        return [((i + 2) // 2) % p for i, p in enumerate(primes)]
        
    # 2. Random
    def init_random():
        return [random.randint(0, p-1) for p in primes]

    # 3. Perturbed seed
    def init_perturbed():
        r = init_seed()
        # Perturb a few
        for _ in range(10):
            idx = random.randint(0, n_primes-1)
            r[idx] = random.randint(0, primes[idx]-1)
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
            
        # Local Search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle prime indices to avoid bias
            prime_indices = list(range(n_primes))
            random.shuffle(prime_indices)
            
            for idx in prime_indices:
                if time.time() > deadline - 5:
                    break
                
                p = primes[idx]
                old_r = residues[idx]
                best_r_for_idx = old_r
                best_L_for_idx = current_L
                
                # Generate candidates
                # 1. Try to cover the current bottleneck (current_L + 1)
                # If current_L + 1 is covered by old_r, we don't need to change for that specific reason,
                # but changing might help later.
                # Let's try a mix of:
                # - Residue that covers current_L + 1
                # - Random residues
                # - Neighbors
                
                candidates = set()
                candidates.add(old_r)
                
                # Target the next gap
                target = current_L + 1
                if target <= MAX_L:
                    target_r = target % p
                    candidates.add(target_r)
                    
                # Add random candidates
                num_random = min(20, p)
                for _ in range(num_random):
                    candidates.add(random.randint(0, p-1))
                    
                # Add neighbors
                candidates.add((old_r + 1) % p)
                candidates.add((old_r - 1) % p)
                
                for r in candidates:
                    if r == old_r:
                        continue
                    if time.time() > deadline - 5:
                        break
                        
                    # Apply change
                    update_coverage(coverage, idx, old_r, r)
                    
                    # Evaluate
                    new_L = find_first_gap(coverage)
                    
                    if new_L > best_L_for_idx:
                        best_L_for_idx = new_L
                        best_r_for_idx = r
                        # Keep the change for now
                        # residues[idx] = r 
                    else:
                        # Revert
                        update_coverage(coverage, idx, r, old_r)
                    
                # Commit best change for this index if found
                if best_r_for_idx != old_r:
                    residues[idx] = best_r_for_idx
                    # Note: coverage was already updated in the loop for the best candidate?
                    # No, in the loop, if new_L > best_L_for_idx, we kept the change.
                    # But if we reverted a worse one, the coverage is back to old_r.
                    # If we found a better one, the coverage is currently at r_best.
                    # We need to ensure residues[idx] matches coverage.
                    # If best_r_for_idx != old_r, the last successful update left coverage at best_r_for_idx.
                    # If no change was better, coverage is at old_r.
                    # So we just need to update residues[idx] if we improved.
                    current_L = best_L_for_idx
                    improved = True
                    if current_L > best_L_global:
                        best_L_global = current_L
                        best_residues_global = residues[:]
                        
        # End of local search for this initialization
        
    return best_residues_global

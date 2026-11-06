import numpy as np
import time
import math

def construct():
    # Generate primes <= 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    
    # Search range for L
    # Using a larger N allows finding higher scores, but slows down.
    # 10000 is a good balance.
    N = 10000
    
    best_residues = None
    best_score = 0
    
    start_time = time.time()
    max_time = 115 # seconds
    
    # Precompute indices
    indices = np.arange(N)
    
    def get_score(counts):
        # Check 1..N-1
        failures = np.nonzero(counts[1:N] < 2)[0]
        if len(failures) == 0:
            return N
        return failures[0] + 1

    def find_best_residue(counts, i, p, residues):
        # Remove current contribution of prime i
        r_old = residues[i]
        mask_old = (indices % p) == r_old
        base_counts = counts - mask_old
        
        best_r = r_old
        best_s = -1
        
        # Vectorized evaluation of all residues for this prime
        # masks shape: (N, p)
        masks = (indices[:, None] % p) == np.arange(p)[None, :]
        
        # new_counts shape: (N, p)
        new_counts = base_counts[:, None] + masks
        
        # Calculate scores for each column
        failure_mask = new_counts[1:N, :] < 2
        any_failure = np.any(failure_mask, axis=0)
        
        first_fail_indices = np.argmax(failure_mask, axis=0)
        
        scores = np.where(any_failure, first_fail_indices + 1, N)
        
        best_idx = np.argmax(scores)
        best_s = scores[best_idx]
        best_r = best_idx
        
        return best_r, best_s

    # Main Loop: Multi-Start Coordinate Descent
    for restart in range(30):
        if time.time() - start_time > max_time:
            break
            
        # Initialize
        if restart == 0:
            residues = np.array([np.random.randint(0, p) for p in primes])
        else:
            # Perturb best solution
            residues = best_residues.copy()
            # Randomly change 10% of residues
            num_changes = max(1, n_primes // 10)
            idxs = np.random.choice(n_primes, num_changes, replace=False)
            for idx in idxs:
                p = primes[idx]
                residues[idx] = np.random.randint(0, p)
        
        # Calculate initial counts
        counts = np.zeros(N, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            counts[indices % p == r] += 1
            
        current_score = get_score(counts)
        
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
        
        # Coordinate Descent
        no_improve_count = 0
        max_no_improve = 5
        
        while time.time() - start_time < max_time and no_improve_count < max_no_improve:
            improved = False
            # Shuffle order of primes to avoid bias
            order = np.random.permutation(n_primes)
            
            for i in order:
                if time.time() - start_time > max_time:
                    break
                
                p = primes[i]
                r_old = residues[i]
                
                best_r, best_s = find_best_residue(counts, i, p, residues)
                
                if best_r != r_old:
                    # Update counts
                    counts[indices % p == r_old] -= 1
                    counts[indices % p == best_r] += 1
                    residues[i] = best_r
                    current_score = best_s
                    improved = True
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = residues.copy()
                
            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0
                
    return best_residues.tolist()

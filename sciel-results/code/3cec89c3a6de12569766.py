import time
import random
import numpy as np

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_p in enumerate(sieve) if is_p]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # Evaluation window size
    # If we can score > 3.5, we want L > 3500.
    # Let's use N=50000 to be safe and allow for large jumps.
    N = 50000
    
    x_vals = np.arange(N + 1)
    
    def _init_coverage(residues):
        cov = np.zeros(N + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            start = r if r > 0 else p
            if start <= N:
                idx = np.arange(start, N + 1, p)
                cov[idx] += 1
        return cov

    def _get_score(coverage):
        # Find first index in 1..N where coverage < 2
        # coverage[1:] corresponds to x=1..N
        gaps = np.where(coverage[1:] < 2)[0]
        if len(gaps) == 0:
            return N
        return gaps[0]

    # Multi-start initialization
    best_residues = None
    best_score = -1
    best_coverage = None
    
    for seed in [42, 123, 999, 2024, 31337, 777]:
        if time.time() > deadline - 5:
            break
        random.seed(seed)
        residues = [random.randint(0, p - 1) for p in primes]
        cov = _init_coverage(residues)
        score = _get_score(cov)
        if score > best_score:
            best_score = score
            best_residues = residues[:]
            best_coverage = cov.copy()
            
    # Simulated Annealing with Directed Repair
    current_residues = best_residues[:]
    current_coverage = best_coverage.copy()
    current_score = best_score
    
    T = 50.0
    T_min = 0.5
    cooling = 0.99
    
    stagnant = 0
    max_stagnant = 1000
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Directed Repair Logic
        # Find first gap
        gaps = np.where(current_coverage[1:] < 2)[0]
        
        if len(gaps) > 0:
            first_gap_idx = gaps[0]
            gap_x = first_gap_idx + 1
            
            # Bias towards primes that do NOT cover gap_x
            # and bias towards residue that DOES cover gap_x
            
            # Select prime index
            if random.random() < 0.6:
                # Try to pick a prime not covering the gap
                # We can sample a few random indices and check
                i = -1
                for _ in range(5):
                    cand_i = random.randint(0, n_primes - 1)
                    if current_residues[cand_i] != gap_x % primes[cand_i]:
                        i = cand_i
                        break
                if i == -1:
                    i = random.randint(0, n_primes - 1)
            else:
                i = random.randint(0, n_primes - 1)
                
            p = primes[i]
            old_r = current_residues[i]
            
            # Select new residue
            if random.random() < 0.4:
                # Target the gap
                new_r = gap_x % p
            else:
                new_r = random.randint(0, p - 1)
                
            if new_r == old_r:
                T *= cooling
                continue
                
            # Evaluate change
            # Compute masks
            # Note: x_vals is 0..N. 
            # We only care about 1..N for score, but coverage[0] is 0.
            # Masks include 0.
            old_mask = (x_vals % p == old_r)
            new_mask = (x_vals % p == new_r)
            
            # Update coverage
            # int32 arithmetic
            current_coverage += (new_mask - old_mask).astype(np.int32)
            
            new_score = _get_score(current_coverage)
            
            # Acceptance
            delta = new_score - current_score
            if delta > 0 or (T > 0 and random.random() < np.exp(delta / T)):
                # Accept
                current_residues[i] = new_r
                current_score = new_score
                
                if new_score > best_score:
                    best_score = new_score
                    best_residues = current_residues[:]
                    best_coverage = current_coverage.copy()
                    stagnant = 0
                else:
                    stagnant += 1
            else:
                # Reject
                current_coverage -= (new_mask - old_mask).astype(np.int32)
                stagnant += 1
                
            T *= cooling
            
            if stagnant > max_stagnant:
                # Restart from best
                current_residues = best_residues[:]
                current_coverage = best_coverage.copy()
                current_score = best_score
                T = 50.0
                stagnant = 0

    return best_residues

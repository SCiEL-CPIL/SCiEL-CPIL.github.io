import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i:limit + 1:i] = [False] * len(sieve[i * i:limit + 1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000
    
    # Helper to calculate counts for a given residue set
    def get_counts(residues):
        counts = np.zeros(MAX_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            start = r if r > 0 else p
            if start <= MAX_L:
                counts[start::p] += 1
        return counts

    def find_L(counts, check_limit=None):
        if check_limit is None:
            check_limit = MAX_L
        check_limit = min(check_limit, MAX_L)
        if np.all(counts[1:check_limit+1] >= 2):
            return check_limit
        bad_idx = np.where(counts[1:check_limit+1] < 2)[0]
        if len(bad_idx) == 0:
            return check_limit
        return bad_idx[0]

    # --- Phase 1: Multiple Greedy Initializations ---
    best_init_L = 0
    best_init_res = [0] * n_primes
    best_init_counts = np.zeros(MAX_L + 1, dtype=np.int32)
    
    # Strategy: For each prime, choose residue to maximize coverage in a sliding window
    for attempt in range(5):
        residues = [0] * n_primes
        counts = np.zeros(MAX_L + 1, dtype=np.int32)
        
        for idx in range(n_primes):
            p = primes[idx]
            best_r = 0
            best_score = -1
            
            # For small primes, try all; for large, sample strategically
            if p <= 30:
                candidates = list(range(p))
            else:
                # Focus on residues that cover positions near the current frontier
                # Try residues that hit positions in 1..min(2000, MAX_L)
                candidates = set()
                for target in range(1, min(2000, MAX_L + 1)):
                    r = target % p
                    candidates.add(r)
                # Also add some random ones
                for _ in range(min(50, p)):
                    candidates.add(np.random.randint(p))
                candidates = list(candidates)
            
            for r in candidates:
                start = r if r > 0 else p
                if start > MAX_L:
                    continue
                # Count how many positions in 1..MAX_L with count==1 would become count==2
                pos = np.arange(start, MAX_L + 1, p)
                current_counts = counts[pos]
                gained = np.sum(current_counts == 1)
                # Also penalize if we're wasting coverage on already-2+ positions
                wasted = np.sum(current_counts >= 2)
                score = gained - 0.1 * wasted
                if score > best_score:
                    best_score = score
                    best_r = r
            
            residues[idx] = best_r
            r = best_r
            start = r if r > 0 else p
            if start <= MAX_L:
                counts[start::p] += 1
        
        L = find_L(counts)
        if L > best_init_L:
            best_init_L = L
            best_init_res = residues.copy()
            best_init_counts = counts.copy()

    # Start from the best greedy result
    residues = best_init_res
    counts = best_init_counts
    current_L = best_init_L
    best_L = current_L
    best_residues = residues.copy()
    
    # --- Phase 2: Simulated Annealing with multiple restarts ---
    T = 10.0
    cooling_rate = 0.99995
    min_T = 0.05
    K_moves = 3
    
    prime_indices = np.arange(n_primes)
    
    # Run SA for a portion of the time, then restart
    sa_deadline = start_time + 110.0
    restart_count = 0
    max_restarts = 3
    
    while time.time() < sa_deadline and restart_count < max_restarts:
        # Reset temperature for each restart (except first)
        if restart_count > 0:
            # Start from best found so far with some perturbation
            residues = best_residues.copy()
            counts = get_counts(residues)
            current_L = best_L
            T = 5.0
            # Perturb some residues
            for _ in range(10):
                idx = np.random.randint(n_primes)
                p = primes[idx]
                residues[idx] = np.random.randint(p)
                old_r = best_residues[idx]
                start_old = old_r if old_r > 0 else p
                start_new = residues[idx] if residues[idx] > 0 else p
                if start_old <= MAX_L:
                    counts[start_old::p] -= 1
                if start_new <= MAX_L:
                    counts[start_new::p] += 1
            current_L = find_L(counts)
        
        while time.time() < sa_deadline:
            # Choose K random primes to perturb
            if n_primes < K_moves:
                indices_to_change = prime_indices
                K_actual = n_primes
            else:
                indices_to_change = np.random.choice(prime_indices, K_moves, replace=False)
                K_actual = K_moves
                
            old_residues = [residues[i] for i in indices_to_change]
            
            # Generate new random residues for these primes
            new_residues = []
            for i in indices_to_change:
                p = primes[i]
                new_residues.append(np.random.randint(p))
            
            # Apply changes to counts
            for j, idx in enumerate(indices_to_change):
                p = primes[idx]
                old_r = residues[idx]
                new_r = new_residues[j]
                
                if old_r == new_r:
                    continue
                    
                # Remove old
                start_old = old_r if old_r > 0 else p
                if start_old <= MAX_L:
                    counts[start_old::p] -= 1
                
                # Add new
                start_new = new_r if new_r > 0 else p
                if start_new <= MAX_L:
                    counts[start_new::p] += 1
                
                residues[idx] = new_r
            
            # Calculate new L - only check up to current best + some margin for speed
            check_limit = min(current_L + 5000, MAX_L)
            new_L = find_L(counts, check_limit)
            
            # Acceptance
            if new_L >= current_L:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
            else:
                delta = new_L - current_L
                if delta > -1000:
                    prob = np.exp(delta / T)
                    if np.random.random() < prob:
                        current_L = new_L
                    else:
                        # Revert
                        for j, idx in enumerate(indices_to_change):
                            p = primes[idx]
                            new_r = new_residues[j]
                            old_r = old_residues[j]
                            
                            start_new = new_r if new_r > 0 else p
                            if start_new <= MAX_L:
                                counts[start_new::p] -= 1
                            
                            start_old = old_r if old_r > 0 else p
                            if start_old <= MAX_L:
                                counts[start_old::p] += 1
                            
                            residues[idx] = old_r
                else:
                    # Revert
                    for j, idx in enumerate(indices_to_change):
                        p = primes[idx]
                        new_r = new_residues[j]
                        old_r = old_residues[j]
                        
                        start_new = new_r if new_r > 0 else p
                        if start_new <= MAX_L:
                            counts[start_new::p] -= 1
                        
                        start_old = old_r if old_r > 0 else p
                        if start_old <= MAX_L:
                            counts[start_old::p] += 1
                        
                        residues[idx] = old_r
            
            T *= cooling_rate
            if T < min_T:
                T = min_T
                # If temperature is at minimum, restart
                restart_count += 1
                break
    
    return best_residues

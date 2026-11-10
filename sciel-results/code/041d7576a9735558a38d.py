import numpy as np
import random
import time

# EVOLVE-BLOCK-START
def construct():
    # Sieve for primes up to 1000
    sieve = [True] * 1001
    sieve[0] = sieve[1] = False
    for i in range(2, int(1000 ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    primes = [i for i in range(2, 1001) if sieve[i]]
    n_primes = len(primes)
    
    # Helper: build count array for given residues up to limit
    def build_counts(residues, limit):
        counts = np.zeros(limit + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                counts[p:limit+1:p] += 1
            else:
                counts[r:limit+1:p] += 1
        return counts
    
    # Helper: find largest L such that all 1..L have count >= 2
    def get_L(counts, limit):
        if limit < 1:
            return 0
        below = np.where(counts[1:limit+1] < 2)[0]
        if len(below) == 0:
            return limit
        return int(below[0])
    
    # Differentiable surrogate: sum of squared deficits below 2
    def surrogate_deficit(counts, limit):
        """Minimize this: sum of (2 - count[x])^2 for x where count[x] < 2"""
        below = counts[1:limit+1]
        deficit = np.maximum(2 - below, 0)
        return float(np.sum(deficit * deficit))
    
    # Smart initialization: greedy for small primes
    residues = [0] * n_primes
    L_INIT = 500
    count_init = np.zeros(L_INIT + 1, dtype=np.int32)
    
    # For small primes, greedily pick residue that minimizes deficit
    small_prime_count = min(40, n_primes)
    for i in range(small_prime_count):
        p = primes[i]
        best_r = 0
        best_deficit = float('inf')
        for r in range(p):
            if r == 0:
                positions = np.arange(p, L_INIT + 1, p)
            else:
                if r > L_INIT:
                    continue
                positions = np.arange(r, L_INIT + 1, p)
            if len(positions) == 0:
                deficit = float(np.sum(np.maximum(2 - count_init[1:L_INIT+1], 0)**2))
            else:
                temp = count_init.copy()
                temp[positions] += 1
                deficit = float(np.sum(np.maximum(2 - temp[1:L_INIT+1], 0)**2))
            if deficit < best_deficit:
                best_deficit = deficit
                best_r = r
        residues[i] = best_r
        r = best_r
        if r == 0:
            count_init[p:L_INIT+1:p] += 1
        else:
            count_init[r:L_INIT+1:p] += 1
    
    # For remaining primes, use deficit-minimizing greedy
    for i in range(small_prime_count, n_primes):
        p = primes[i]
        best_r = 0
        best_deficit = float('inf')
        # Sample residues for large primes
        if p > 100:
            candidates = [0] + [random.randint(0, p-1) for _ in range(20)]
        else:
            candidates = list(range(p))
        for r in candidates:
            if r == 0:
                positions = np.arange(p, L_INIT + 1, p)
            else:
                if r > L_INIT:
                    continue
                positions = np.arange(r, L_INIT + 1, p)
            if len(positions) == 0:
                deficit = float(np.sum(np.maximum(2 - count_init[1:L_INIT+1], 0)**2))
            else:
                temp = count_init.copy()
                temp[positions] += 1
                deficit = float(np.sum(np.maximum(2 - temp[1:L_INIT+1], 0)**2))
            if deficit < best_deficit:
                best_deficit = deficit
                best_r = r
        residues[i] = best_r
        r = best_r
        if r == 0:
            count_init[p:L_INIT+1:p] += 1
        else:
            count_init[r:L_INIT+1:p] += 1
    
    # Initial M and counts
    M = 5000
    max_M = 100000
    counts = build_counts(residues, M)
    current_L = get_L(counts, M)
    best_L = current_L
    best_residues = residues[:]
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Coordinate descent with deficit-based scoring
    # For each prime, find the residue that minimizes local deficit impact
    
    stagnation_count = 0
    max_stagnation = 30
    
    # Precompute position masks for efficiency
    def get_positions(p, r, limit):
        if r == 0:
            return np.arange(p, limit+1, p)
        else:
            if r > limit:
                return np.array([], dtype=np.int64)
            return np.arange(r, limit+1, p)
    
    while time.time() < deadline - 5:
        # Expand M if close to boundary
        if current_L >= M - 500:
            new_M = min(M * 2, max_M)
            if new_M > M:
                counts = build_counts(residues, new_M)
                M = new_M
                current_L = get_L(counts, M)
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
                continue
        
        # Coordinate descent: try to improve each prime's residue
        improved = False
        # Process primes in order of impact (smaller primes first)
        for i in range(n_primes):
            if time.time() > deadline - 5:
                break
            
            p = primes[i]
            old_r = residues[i]
            
            # Find best residue for this prime by evaluating deficit change
            best_r = old_r
            best_delta = 0  # We want to minimize deficit (maximize delta = old - new)
            
            # Sample candidate residues
            if p <= 50:
                candidates = list(range(p))
            else:
                # Include old_r and some smart candidates
                candidates = [old_r]
                # Residue that covers the first gap
                if current_L + 1 <= M:
                    candidates.append((current_L + 1) % p)
                # Random candidates
                for _ in range(min(30, p // 2)):
                    candidates.append(random.randint(0, p - 1))
                # Deduplicate
                candidates = list(set(candidates))
            
            for new_r in candidates:
                if new_r == old_r:
                    continue
                
                # Compute deficit change efficiently
                old_positions = get_positions(p, old_r, M)
                new_positions = get_positions(p, new_r, M)
                
                # Remove old coverage, add new coverage
                temp_counts = counts.copy()
                if len(old_positions) > 0:
                    temp_counts[old_positions] -= 1
                if len(new_positions) > 0:
                    temp_counts[new_positions] += 1
                
                new_deficit = float(np.sum(np.maximum(2 - temp_counts[1:M+1], 0)**2))
                old_deficit = float(np.sum(np.maximum(2 - counts[1:M+1], 0)**2))
                delta = old_deficit - new_deficit  # Positive means improvement
                
                if delta > best_delta:
                    best_delta = delta
                    best_r = new_r
            
            if best_r != old_r and best_delta > 0:
                # Apply the change
                old_positions = get_positions(p, old_r, M)
                new_positions = get_positions(p, best_r, M)
                
                if len(old_positions) > 0:
                    counts[old_positions] -= 1
                if len(new_positions) > 0:
                    counts[new_positions] += 1
                residues[i] = best_r
                
                new_L = get_L(counts, M)
                if new_L > current_L:
                    current_L = new_L
                    improved = True
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues[:]
        
        if improved:
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        # When stagnant, do a burst of random perturbations
        if stagnation_count > max_stagnation:
            burst_start = time.time()
            for _ in range(200):
                if time.time() > burst_start + 5.0 or time.time() > deadline - 5:
                    break
                
                idx = random.randint(0, n_primes - 1)
                p = primes[idx]
                old_r = residues[idx]
                new_r = random.randint(0, p - 1)
                
                if new_r == old_r:
                    continue
                
                old_positions = get_positions(p, old_r, M)
                new_positions = get_positions(p, new_r, M)
                
                if len(old_positions) > 0:
                    counts[old_positions] -= 1
                if len(new_positions) > 0:
                    counts[new_positions] += 1
                residues[idx] = new_r
                
                new_L = get_L(counts, M)
                if new_L > current_L:
                    current_L = new_L
                    stagnation_count = 0
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues[:]
                else:
                    # Revert
                    if len(new_positions) > 0:
                        counts[new_positions] -= 1
                    if len(old_positions) > 0:
                        counts[old_positions] += 1
                    residues[idx] = old_r
            
            stagnation_count = 0
    
    return best_residues
# EVOLVE-BLOCK-END

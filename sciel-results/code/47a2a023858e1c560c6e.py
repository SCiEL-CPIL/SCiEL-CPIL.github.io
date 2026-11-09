import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    num_primes = len(primes)
    
    N = 1000
    x = np.arange(1, N + 1, dtype=np.int32)
    
    # Precompute x % p for all primes to speed up mask generation
    # x_mod_p[i] is x % primes[i]
    x_mod_p = np.zeros((num_primes, N), dtype=np.int32)
    for i, p in enumerate(primes):
        x_mod_p[i] = x % p
        
    def compute_counts(residues):
        counts = np.zeros(N, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            counts += (x_mod_p[i] == r)
        return counts
        
    def compute_L(counts):
        # Find first index where count < 2
        # np.argmax returns 0 if all are >= 2, so we check that case
        fail_idx = np.argmax(counts < 2)
        if counts[fail_idx] >= 2:
            return N
        return fail_idx

    # 1. Greedy Initialization
    # Start with 0 coverage, add primes one by one choosing residue that maximizes min coverage in 1..N
    # This is a heuristic to get a good starting point.
    residues = np.zeros(num_primes, dtype=np.int32)
    counts = np.zeros(N, dtype=np.int32)
    
    # We want to maximize the minimum count. 
    # Greedy: for each prime, pick r that maximizes the minimum count of the resulting vector.
    # This is O(N * num_primes * N) which is too slow for 168 primes * 1000 N * 1000 N.
    # Instead, a simpler greedy: 
    # Sort primes? No, order doesn't matter much for final state, but for greedy it does.
    # Let's just use a few random restarts of greedy or simple random + hill climb.
    
    # Better Init: Random, but then run a fast "Fix Bottlenecks" pass.
    
    best_residues = None
    best_L = -1
    best_counts = None
    
    # Try a few different random seeds for initialization
    for seed in range(10):
        random.seed(seed)
        residues = np.array([random.randint(0, p-1) for p in primes], dtype=np.int32)
        counts = compute_counts(residues)
        current_L = compute_L(counts)
        
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
            best_counts = counts.copy()
            
        # Local Optimization: Bottleneck Driven
        # Loop until no improvement or time check
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            current_L = compute_L(counts)
            
            if current_L == N:
                break
                
            # Find the first failing integer
            fail_idx = np.argmax(counts < 2)
            if counts[fail_idx] >= 2:
                break # All covered
            
            target_x = fail_idx # 0-indexed, so value is fail_idx + 1
            
            # Find primes that currently do NOT cover target_x
            # We want to change r_p to (target_x % p) for some p
            # But we must ensure we don't break coverage of other x in 1..current_L
            
            # Candidates: Primes where changing r_p to target_r increases count of target_x
            # Since target_x is not covered by p currently (count < 2 implies it might be 0 or 1),
            # actually we just want to increase count of target_x.
            
            # Let's pick a random subset of primes to try to avoid O(P) overhead per step if P is large
            # But P=168 is small.
            
            # Strategy: Try all primes that don't cover target_x? 
            # If p covers target_x, changing r_p might decrease its count or keep it.
            # If p does NOT cover target_x, changing r_p to target_x % p increases count by 1.
            
            # We need to be careful: increasing count of target_x might decrease count of other x's.
            # We only accept if new_L > current_L.
            
            # To speed up, we can prioritize primes where the "loss" elsewhere is minimal.
            # But simple random sampling of candidates is robust.
            
            # Get indices of primes
            indices = np.arange(num_primes)
            random.shuffle(indices)
            
            for i in indices:
                p = primes[i]
                old_r = residues[i]
                new_r = target_x % p # Note: x is 1..N, so x[fail_idx] = fail_idx+1. 
                # Wait, x = np.arange(1, N+1). So x[fail_idx] is the value.
                target_val = x[fail_idx]
                new_r = target_val % p
                
                if new_r == old_r:
                    continue
                    
                # Calculate delta
                # Old mask: x_mod_p[i] == old_r
                # New mask: x_mod_p[i] == new_r
                
                # We update counts temporarily
                counts -= (x_mod_p[i] == old_r)
                counts += (x_mod_p[i] == new_r)
                
                new_L = compute_L(counts)
                
                if new_L > current_L:
                    # Accept
                    residues[i] = new_r
                    current_L = new_L
                    improved = True
                    # Continue loop to find next bottleneck
                    break # Break out of prime loop to re-evaluate bottleneck
                else:
                    # Revert
                    counts -= (x_mod_p[i] == new_r)
                    counts += (x_mod_p[i] == old_r)
                    
            # If we tried all primes and didn't improve, we are stuck in a local minimum
            # for single-coordinate moves targeting the first bottleneck.
            # Try a "kick": change a random residue to escape, or try a 2-move.
            
            if not improved:
                # Try a random kick to escape local min
                # Pick a random prime and change it to a random residue
                # Accept if it doesn't reduce L too much? Or just accept and hope?
                # Better: Try to fix the bottleneck by changing TWO primes.
                # This is complex. Let's just do a random perturbation and hill climb again.
                
                # Perturb 3 random residues
                for _ in range(3):
                    i = random.randint(0, num_primes - 1)
                    p = primes[i]
                    old_r = residues[i]
                    new_r = random.randint(0, p - 1)
                    if new_r != old_r:
                        counts -= (x_mod_p[i] == old_r)
                        counts += (x_mod_p[i] == new_r)
                        residues[i] = new_r
                
                # Re-evaluate L
                current_L = compute_L(counts)
                # If L dropped significantly, we might revert to best?
                # For now, we continue searching from this perturbed state.
                
    # Final Polish: Local search on best_residues
    # Ensure we have the best found so far
    if best_residues is None:
        # Fallback
        residues = np.array([random.randint(0, p-1) for p in primes], dtype=np.int32)
        best_residues = residues
    else:
        residues = best_residues
        counts = compute_counts(residues)
        current_L = compute_L(counts)
        
        # Run a tighter local search on the best candidate
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_L = compute_L(counts)
            if current_L == N: break
            
            fail_idx = np.argmax(counts < 2)
            if counts[fail_idx] >= 2: break
            
            target_val = x[fail_idx]
            
            # Try all primes again, maybe we missed something due to order
            # Or try to improve by looking at primes that cover target_val but maybe we can shift?
            # No, if count < 2, we need to ADD coverage.
            
            indices = np.arange(num_primes)
            random.shuffle(indices)
            
            for i in indices:
                p = primes[i]
                old_r = residues[i]
                new_r = target_val % p
                if new_r == old_r: continue
                
                counts -= (x_mod_p[i] == old_r)
                counts += (x_mod_p[i] == new_r)
                
                new_L = compute_L(counts)
                
                if new_L > current_L:
                    residues[i] = new_r
                    current_L = new_L
                    improved = True
                    break
                else:
                    counts -= (x_mod_p[i] == new_r)
                    counts += (x_mod_p[i] == old_r)
                    
            if not improved:
                # Stuck. Try a 2-move: change prime i and prime j
                # This is expensive. Just break and rely on the earlier perturbations.
                break

    return [int(r) for r in residues]

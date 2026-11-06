import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin
    
    # 1. Generate Primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    primes_np = np.array(primes)
    
    MAX_CHECK = 2000
    x_values = np.arange(1, MAX_CHECK + 1)
    
    # Precompute x mod p
    x_mod_p = x_values[:, None] % primes_np[None, :]
    
    # Build covers structure
    # covers[j][r] = list of x_indices
    # covers_set[j][r] = set of x_indices for fast lookup
    covers = [None] * n_primes
    covers_set = [None] * n_primes
    
    for j, p in enumerate(primes):
        covers[j] = [[] for _ in range(p)]
        covers_set[j] = [set() for _ in range(p)]
        
        for x_idx, x in enumerate(x_values):
            r = x % p
            covers[j][r].append(x_idx)
            covers_set[j][r].add(x_idx)
            
    # Initial Solution: r_p = 1
    current_residues = np.ones(n_primes, dtype=np.int64)
    
    # Initial counts
    counts = np.zeros(MAX_CHECK, dtype=np.int32)
    for j in range(n_primes):
        r = current_residues[j]
        if covers[j][r]:
             idxs = np.array(covers[j][r])
             counts[idxs] += 1
             
    def get_first_gap(counts):
        invalid = counts < 2
        if not np.any(invalid):
            return -1
        first_invalid_idx = np.argmax(invalid)
        return int(first_invalid_idx)
        
    current_gap_idx = get_first_gap(counts)
    best_score = current_gap_idx
    best_residues = current_residues.copy()
    
    # If we start with a gap at 0 (x=1 not covered by 2), we need to fix it.
    # With r_p=1, x=1 is covered by all. x=2 is covered by 0 (as analyzed before).
    # So current_gap_idx will be 1 (index for x=2).
    
    # Greedy Repair Loop
    while time.time() < deadline:
        if current_gap_idx == -1:
            break
            
        target_x_idx = current_gap_idx
        target_x = x_values[target_x_idx]
        
        found_move = False
        perm = np.random.permutation(n_primes)
        
        for j in perm:
            p = primes[j]
            old_r = current_residues[j]
            new_r = target_x % p
            
            if new_r == old_r:
                continue
                
            # Check validity
            # We must not break any x in [0, target_x_idx) that has count == 2.
            # A move breaks x if x is in covers[j][old_r] and NOT in covers[j][new_r].
            
            valid = True
            # Iterate through the lost set
            # Optimization: if the lost set is large, this might be slow.
            # But MAX_CHECK is 2000, so max size is 2000.
            # 168 primes * 2000 checks = 336k ops per gap.
            # Acceptable.
            
            for x_idx in covers[j][old_r]:
                if x_idx >= target_x_idx:
                    continue
                if counts[x_idx] == 2:
                    # Check if x_idx is in the new set
                    if x_idx not in covers_set[j][new_r]:
                        valid = False
                        break
            
            if valid:
                # Apply move
                # Update counts
                for x_idx in covers[j][old_r]:
                    counts[x_idx] -= 1
                for x_idx in covers[j][new_r]:
                    counts[x_idx] += 1
                    
                current_residues[j] = new_r
                
                # Find new gap
                current_gap_idx = get_first_gap(counts)
                
                if current_gap_idx > best_score:
                    best_score = current_gap_idx
                    best_residues = current_residues.copy()
                    
                found_move = True
                break # Next gap
        
        if not found_move:
            # Stuck. 
            # Try a random restart from best? 
            # Or just break.
            # Given the deterministic nature, breaking is fine.
            # But maybe we can try a few random moves to escape local minima?
            # Let's try 100 random moves.
            escaped = False
            for _ in range(100):
                # Pick random prime
                j = np.random.randint(0, n_primes)
                p = primes[j]
                old_r = current_residues[j]
                new_r = np.random.randint(0, p)
                
                if new_r == old_r:
                    continue
                    
                # Apply
                for x_idx in covers[j][old_r]:
                    counts[x_idx] -= 1
                for x_idx in covers[j][new_r]:
                    counts[x_idx] += 1
                current_residues[j] = new_r
                
                # Check if we improved the gap
                new_gap = get_first_gap(counts)
                if new_gap > current_gap_idx:
                    current_gap_idx = new_gap
                    if current_gap_idx > best_score:
                        best_score = current_gap_idx
                        best_residues = current_residues.copy()
                    escaped = True
                    break
                else:
                    # Revert
                    for x_idx in covers[j][new_r]:
                        counts[x_idx] -= 1
                    for x_idx in covers[j][old_r]:
                        counts[x_idx] += 1
                    current_residues[j] = old_r
                    
            if not escaped:
                break

    return [int(r) for r in best_residues]

np.random.seed(42)

if __name__ == "__main__":
    res = construct()
    print(len(res))

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
    p_arr = np.array(primes, dtype=np.int32)
    
    N = 1000
    x = np.arange(1, N + 1, dtype=np.int32)
    
    # Precompute for each prime p, the index array where x % p == r for each r
    # This is expensive to precompute fully, so we'll compute on the fly but vectorized
    
    # Initial residues: try to cover small numbers densely
    # Strategy: for each prime p, pick r_p such that many small numbers are covered
    # Start with r_p = 1 for all (covers x=1, 1+p, 1+2p, ...)
    residues = np.ones(num_primes, dtype=np.int32)
    
    def compute_L(res):
        """Compute the largest L such that every integer 1..L has count >= 2."""
        counts = np.zeros(N, dtype=np.int32)
        for i in range(num_primes):
            p = p_arr[i]
            r = res[i]
            counts += (x % p == r)
        # Find first index where count < 2
        fail_mask = counts < 2
        if not np.any(fail_mask):
            return N
        first_fail = np.argmax(fail_mask)
        return first_fail
    
    current_L = compute_L(residues)
    best_L = current_L
    best_res = residues.copy()
    
    # Simulated annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.9995
    
    # Precompute masks for efficiency
    # For each prime index i, precompute the pattern of positions covered by each residue
    # This is too large (168 * p * N), so we'll compute incrementally
    
    iteration = 0
    while time.time() < deadline and T > T_min:
        if time.time() > deadline - 5.0:
            break
            
        # Choose a random prime index to modify
        i = random.randint(0, num_primes - 1)
        p = p_arr[i]
        old_r = residues[i]
        
        # Choose a new random residue
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
        
        # Compute the change in counts
        # Remove coverage from old_r, add coverage from new_r
        mask_old = (x % p == old_r)
        mask_new = (x % p == new_r)
        
        # Apply temporarily
        residues[i] = new_r
        
        # Compute new L
        new_L = compute_L(residues)
        
        delta = new_L - current_L
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_res = residues.copy()
        else:
            # Revert
            residues[i] = old_r
        
        T *= cooling
        iteration += 1
        
        # Periodic greedy improvement
        if iteration % 100 == 0:
            # Try to fix the first failure greedily
            fail_mask = np.zeros(N, dtype=bool)
            counts_tmp = np.zeros(N, dtype=np.int32)
            for j in range(num_primes):
                counts_tmp += (x % p_arr[j] == residues[j])
            fail_mask = counts_tmp < 2
            if np.any(fail_mask):
                first_fail_idx = np.argmax(fail_mask)
                target_x = first_fail_idx + 1
                # Try each prime to see if changing it can cover target_x
                for j in range(num_primes):
                    p_j = p_arr[j]
                    if (target_x % p_j) != residues[j]:
                        # Check if changing would help without breaking prefix
                        old_r_j = residues[j]
                        new_r_j = target_x % p_j
                        mask_old_j = (x % p_j == old_r_j)
                        # Check if any x in 1..first_fail_idx has count exactly 2 and is covered by old_r_j
                        if first_fail_idx > 0:
                            prefix_danger = (x[:first_fail_idx] % p_j == old_r_j) & (counts_tmp[:first_fail_idx] == 2)
                            if np.any(prefix_danger):
                                continue
                        # Apply and check
                        residues[j] = new_r_j
                        new_L2 = compute_L(residues)
                        if new_L2 > current_L:
                            current_L = new_L2
                            if new_L2 > best_L:
                                best_L = new_L2
                                best_res = residues.copy()
                            break
                        else:
                            residues[j] = old_r_j
    
    # Final greedy pass
    for _ in range(10):
        improved = False
        counts_tmp = np.zeros(N, dtype=np.int32)
        for j in range(num_primes):
            counts_tmp += (x % p_arr[j] == best_res[j])
        fail_mask = counts_tmp < 2
        if not np.any(fail_mask):
            break
        first_fail_idx = np.argmax(fail_mask)
        target_x = first_fail_idx + 1
        for j in range(num_primes):
            p_j = p_arr[j]
            if (target_x % p_j) != best_res[j]:
                old_r_j = best_res[j]
                new_r_j = target_x % p_j
                mask_old_j = (x % p_j == old_r_j)
                if first_fail_idx > 0:
                    prefix_danger = (x[:first_fail_idx] % p_j == old_r_j) & (counts_tmp[:first_fail_idx] == 2)
                    if np.any(prefix_danger):
                        continue
                best_res[j] = new_r_j
                # Recompute
                counts_tmp2 = np.zeros(N, dtype=np.int32)
                for k in range(num_primes):
                    counts_tmp2 += (x % p_arr[k] == best_res[k])
                new_L = compute_L(best_res)
                if new_L > best_L:
                    best_L = new_L
                    improved = True
                    break
                else:
                    best_res[j] = old_r_j
        if not improved:
            break
    
    return [int(r) for r in best_res]

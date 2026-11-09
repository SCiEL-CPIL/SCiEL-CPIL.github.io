import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 118.0  # Safety margin for 120s limit

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    
    # We will search for L up to M. 
    M = 2000
    counts = np.zeros(M + 1, dtype=np.int64)
    
    # Initialize residues
    residues = np.zeros(n_primes, dtype=np.int64)
    for i, p in enumerate(primes):
        residues[i] = random.randint(0, p - 1)
        counts[residues[i]::p] += 1
        
    # --- Greedy Repair Initialization ---
    # Try to extend L by fixing uncovered points sequentially
    # Use large primes to minimize disruption
    x = 1
    # Pre-sort prime indices by prime value descending
    prime_indices_desc = sorted(range(n_primes), key=lambda i: primes[i], reverse=True)
    
    while x <= M:
        if counts[x] >= 2:
            x += 1
            continue
            
        # counts[x] < 2
        # Find a prime p that does not cover x
        # Try large primes first to minimize impact on previous points
        fixed = False
        for i in prime_indices_desc:
            p = primes[i]
            if residues[i] != x % p:
                # Check if this move breaks any y < x?
                # We can't check all y easily.
                # But we can check if the old residue r_old covers any y < x 
                # that has count == 2 (critical).
                # If so, skip this prime.
                
                r_old = residues[i]
                r_new = x % p
                
                # Check critical points
                # The points affected are those y such that y % p == r_old
                # And y < x.
                # For each such y, if counts[y] == 2, this move would make it 1.
                # We want to avoid that.
                
                # Iterate y = r_old, r_old+p, ... < x
                # This is O(x/p). For large p, this is small.
                
                is_safe = True
                # Start from the largest y < x that is congruent to r_old mod p
                # y_max = r_old + floor((x-1-r_old)/p)*p
                if r_old < x:
                    k = (x - 1 - r_old) // p
                    if k >= 0:
                        y = r_old + k * p
                        while y >= 1:
                            if counts[y] == 2:
                                # This move would break y
                                is_safe = False
                                break
                            y -= p
                            
                if is_safe:
                    # Apply move
                    counts[r_old::p] -= 1
                    counts[r_new::p] += 1
                    residues[i] = r_new
                    fixed = True
                    break
        
        if not fixed:
            # No safe move found. Just pick the largest prime that doesn't cover x
            # and hope for the best (SA will fix)
            for i in prime_indices_desc:
                p = primes[i]
                if residues[i] != x % p:
                    r_old = residues[i]
                    r_new = x % p
                    counts[r_old::p] -= 1
                    counts[r_new::p] += 1
                    residues[i] = r_new
                    break
        x += 1

    def calc_L(counts, M):
        # Returns the largest L such that counts[1..L] >= 2
        if counts[1] < 2:
            return 0
        # Find first index >= 1 where count < 2
        mask = counts[1:] < 2
        if not np.any(mask):
            return M
        first_false_idx = np.argmax(mask)
        return first_false_idx
    
    def find_first_uncovered(counts, start, M):
        """Find first index >= start where count < 2, or M if none"""
        if start > M:
            return M
        if start <= 0:
            start = 1
        segment = counts[start:M+1]
        if len(segment) == 0:
            return M
        mask = segment < 2
        if not np.any(mask):
            return M
        return start + np.argmax(mask)

    current_L = calc_L(counts, M)
    best_L = current_L
    best_residues = residues.copy()
    
    # Helper to update counts and return delta or new L
    # We will do this inline for speed
    
    # Optimization loop
    # We use a mix of directed and random moves
    T = 10.0 # Temperature for accepting worse moves (start high for SA)
    T_min = 0.1
    cooling_rate = 0.9995
    
    while time.time() - start_time < deadline:
        # Pick a random prime index
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        r_old = residues[i]
        
        # Determine target residue
        # If L < M, target is L+1
        # We want to cover x = current_L + 1
        x_target = current_L + 1
        r_target = x_target % p
        
        # With probability 0.7, pick r_target, else random
        # This balances exploration (random) and exploitation (targeted)
        if random.random() < 0.7:
            r_new = r_target
        else:
            r_new = random.randint(0, p - 1)
            
        if r_new == r_old:
            continue
            
        # Apply move
        counts[r_old::p] -= 1
        counts[r_new::p] += 1
        
        new_L = calc_L(counts, M)
        
        # If we hit the boundary M, we might have a better L.
        # For now, just use new_L.
        
        if new_L >= current_L:
            current_L = new_L
            residues[i] = r_new
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
                
                # Optional: Extend M if we are close to the limit
                if best_L >= M - 10:
                    # Extend counts and M
                    new_M = M * 2
                    old_counts = counts
                    counts = np.zeros(new_M + 1, dtype=np.int64)
                    # Rebuild counts from residues? Or copy?
                    # Copying and updating is faster if we just append zeros, 
                    # but the stride changes? No, stride is fixed by p.
                    # We need to fill the new part of counts.
                    # counts[new_part] += contributions from primes that reach there.
                    # Easier: just recalculate counts for the new M? 
                    # No, we can just extend the array and add the missing parts.
                    # The missing parts are indices from M+1 to new_M.
                    # For each prime, add 1 to counts[k::p] for k in [M+1, new_M].
                    # This is efficient.
                    for j, pj in enumerate(primes):
                        rj = residues[j]
                        # Find first multiple >= M+1 that is congruent to rj mod pj
                        # k = rj + m*pj >= M+1
                        # m >= (M+1 - rj) / pj
                        start_m = max(0, (M + 1 - rj + pj - 1) // pj)
                        first_idx = rj + start_m * pj
                        if first_idx <= new_M:
                            counts[first_idx::pj] += 1
                    M = new_M
                    # Recalculate current_L and best_L? 
                    # current_L was M (old), so it is at least M.
                    # We should recalc to see if it extended further.
                    current_L = calc_L(counts, M)
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
                        
        else:
            # Simulated Annealing acceptance
            delta = current_L - new_L
            # prob = np.exp(-delta / T)
            # Using log for stability if T is very small, but T_min is 0.1 so exp is fine
            prob = np.exp(-delta / T)
            if random.random() < prob:
                current_L = new_L
                residues[i] = r_new
            else:
                # Revert
                counts[r_new::p] -= 1
                counts[r_old::p] += 1
        
        # Cool down
        T = max(T_min, T * cooling_rate)

    return [int(r) for r in best_residues]

"""Improved with Constraint-Driven Greedy Repair and Window Expansion."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # Initial window size
    MAX_WINDOW = 1000
    counts = np.zeros(MAX_WINDOW + 1, dtype=np.int32)
    
    # Initial residues: r_p = 1
    residues = [1 % p for p in primes]

    def add_contributions(counts_arr, start_idx, end_idx, residues_list):
        """Add contributions for indices in [start_idx, end_idx] for all primes."""
        for i in range(n_primes):
            p = primes[i]
            r = residues_list[i]
            # Find first x in [start_idx, end_idx] such that x % p == r
            # x = k*p + r
            # k*p >= start_idx - r
            # k >= ceil((start_idx - r) / p)
            
            if r == 0:
                # x is multiple of p
                # First multiple >= start_idx
                k_min = (start_idx + p - 1) // p
                first_x = k_min * p
            else:
                # x = k*p + r
                # k*p >= start_idx - r
                k_min = (start_idx - r + p - 1) // p
                if k_min < 0:
                    k_min = 0
                first_x = k_min * p + r
                if first_x < start_idx:
                    first_x += p

            if first_x <= end_idx:
                idxs = np.arange(first_x, end_idx + 1, p, dtype=np.int64)
                counts_arr[idxs] += 1

    # Initialize counts for [1, MAX_WINDOW]
    add_contributions(counts, 1, MAX_WINDOW, residues)

    def get_L(counts_arr):
        """Find largest L such that counts[1..L] >= 2."""
        mask = counts_arr[1:] < 2
        if not mask.any():
            return len(counts_arr) - 1
        first_gap_idx = np.argmax(mask)
        return first_gap_idx

    def try_fix_gap(gap_x, counts_arr, residues_list):
        """
        Try to fix the gap at gap_x by changing one prime's residue.
        Returns (success, new_residue_idx, new_residue_val) or (False, -1, -1)
        """
        x_range = np.arange(1, gap_x + 1, dtype=np.int64)
        current_counts_prefix = counts_arr[1:gap_x+1]
        
        for i in range(n_primes):
            p = primes[i]
            r_old = residues_list[i]
            r_new = gap_x % p
            
            if r_new == r_old:
                continue
                
            mods = x_range % p
            mask_old = (mods == r_old)
            mask_new = (mods == r_new)
            
            new_counts = current_counts_prefix.copy()
            new_counts[mask_old] -= 1
            new_counts[mask_new] += 1
            
            if np.min(new_counts) >= 2:
                return True, i, r_new
                
        return False, -1, -1

    def update_counts_for_prime(i, old_r, new_r, counts_arr):
        """Update counts array when residue of prime i changes from old_r to new_r."""
        p = primes[i]
        
        # Remove old contributions in [1, MAX_WINDOW]
        if old_r == 0:
            start_old = p
        else:
            start_old = old_r
        if start_old < 1:
            start_old += p
        if start_old <= MAX_WINDOW:
            idxs = np.arange(start_old, MAX_WINDOW + 1, p, dtype=np.int64)
            counts_arr[idxs] -= 1
            
        # Add new contributions in [1, MAX_WINDOW]
        if new_r == 0:
            start_new = p
        else:
            start_new = new_r
        if start_new < 1:
            start_new += p
        if start_new <= MAX_WINDOW:
            idxs = np.arange(start_new, MAX_WINDOW + 1, p, dtype=np.int64)
            counts_arr[idxs] += 1

    # Main Loop
    stuck_counter = 0
    max_stuck = 50
    
    while time.time() < deadline - 5.0:
        L = get_L(counts)
        
        if L >= MAX_WINDOW:
            # Expand window
            new_max = MAX_WINDOW * 2
            if new_max > 100000:
                new_max = 100000
            if new_max == MAX_WINDOW:
                break
                
            old_counts = counts
            counts = np.zeros(new_max + 1, dtype=np.int32)
            # Copy old counts
            counts[1:MAX_WINDOW+1] = old_counts[1:MAX_WINDOW+1]
            # Add contributions for new range
            add_contributions(counts, MAX_WINDOW + 1, new_max, residues)
            MAX_WINDOW = new_max
            continue

        gap_x = L + 1
        
        success, i, r_new = try_fix_gap(gap_x, counts, residues)
        
        if success:
            old_r = residues[i]
            residues[i] = r_new
            update_counts_for_prime(i, old_r, r_new, counts)
            stuck_counter = 0
        else:
            stuck_counter += 1
            
            if stuck_counter > max_stuck:
                # Try a random kick on a small prime
                i_kick = random.randint(0, 10)
                p_kick = primes[i_kick]
                old_r_kick = residues[i_kick]
                new_r_kick = random.randint(0, p_kick - 1)
                
                if new_r_kick != old_r_kick:
                    residues[i_kick] = new_r_kick
                    update_counts_for_prime(i_kick, old_r_kick, new_r_kick, counts)
                    stuck_counter = 0

    return residues

# EVOLVE-BLOCK-END

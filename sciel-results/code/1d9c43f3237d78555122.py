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

    current_L = calc_L(counts, M)
    best_L = current_L
    best_residues = residues.copy()
    
    # Optimization loop
    T = 10.0 # Temperature for accepting worse moves
    T_min = 0.1
    cooling_rate = 0.9995
    
    # Helper to apply a single move and return new L
    def apply_move(i, r_new):
        p = primes[i]
        r_old = residues[i]
        if r_new == r_old:
            return current_L
        counts[r_old::p] -= 1
        counts[r_new::p] += 1
        residues[i] = r_new
        return calc_L(counts, M)

    # Helper to revert a single move
    def revert_move(i, r_new):
        p = primes[i]
        r_old = residues[i]
        counts[r_new::p] -= 1
        counts[r_old::p] += 1
        residues[i] = r_old
        return calc_L(counts, M)

    while time.time() - start_time < deadline:
        # Determine move type
        move_type = random.random()
        if move_type < 0.7:
            # 1-move: Targeted at L+1
            x_target = current_L + 1
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            r_new = x_target % p
            r_old = residues[i]
            if r_new == r_old:
                continue
            # Apply
            counts[r_old::p] -= 1
            counts[r_new::p] += 1
            residues[i] = r_new
            new_L = calc_L(counts, M)
            
            # Acceptance
            if new_L >= current_L:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
                    if best_L >= M - 10:
                        # Extend M
                        new_M = M * 2
                        old_counts = counts
                        counts = np.zeros(new_M + 1, dtype=np.int64)
                        for j, pj in enumerate(primes):
                            rj = residues[j]
                            start_m = max(0, (M + 1 - rj + pj - 1) // pj)
                            first_idx = rj + start_m * pj
                            if first_idx <= new_M:
                                counts[first_idx::pj] += 1
                        M = new_M
                        current_L = calc_L(counts, M)
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = residues.copy()
            else:
                delta = current_L - new_L
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_L = new_L
                else:
                    # Revert
                    counts[r_new::p] -= 1
                    counts[r_old::p] += 1
                    residues[i] = r_old

        elif move_type < 0.8:
            # 1-move: Random
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            r_old = residues[i]
            r_new = random.randint(0, p - 1)
            if r_new == r_old:
                continue
            counts[r_old::p] -= 1
            counts[r_new::p] += 1
            residues[i] = r_new
            new_L = calc_L(counts, M)
            
            if new_L >= current_L:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
                    if best_L >= M - 10:
                        new_M = M * 2
                        counts = np.zeros(new_M + 1, dtype=np.int64)
                        for j, pj in enumerate(primes):
                            rj = residues[j]
                            start_m = max(0, (M + 1 - rj + pj - 1) // pj)
                            first_idx = rj + start_m * pj
                            if first_idx <= new_M:
                                counts[first_idx::pj] += 1
                        M = new_M
                        current_L = calc_L(counts, M)
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = residues.copy()
            else:
                delta = current_L - new_L
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_L = new_L
                else:
                    counts[r_new::p] -= 1
                    counts[r_old::p] += 1
                    residues[i] = r_old

        else:
            # 2-move: Targeted at L+1, then repair
            x_target = current_L + 1
            i = random.randint(0, n_primes - 1)
            p1 = primes[i]
            r1_old = residues[i]
            r1_new = x_target % p1
            if r1_new == r1_old:
                continue
            
            # Apply first move
            counts[r1_old::p1] -= 1
            counts[r1_new::p1] += 1
            residues[i] = r1_new
            
            L_after_1 = calc_L(counts, M)
            
            if L_after_1 < current_L:
                # We broke something. Find the first broken index.
                # It must be < x_target (since x_target is now covered, but maybe something before it broke)
                # Actually, if L decreased, the first index with count < 2 is the new L.
                # We want to fix it.
                broken_x = L_after_1
                # Find a prime j != i to fix broken_x
                # We need r_j such that broken_x % p_j is the new residue.
                # And we hope this doesn't break anything else.
                # Pick random j != i
                j = random.randint(0, n_primes - 1)
                if j == i:
                    j = (j + 1) % n_primes
                p2 = primes[j]
                r2_old = residues[j]
                r2_new = broken_x % p2
                
                if r2_new != r2_old:
                    # Apply second move
                    counts[r2_old::p2] -= 1
                    counts[r2_new::p2] += 1
                    residues[j] = r2_new
                    
                    L_after_2 = calc_L(counts, M)
                    
                    if L_after_2 >= current_L:
                        current_L = L_after_2
                        if L_after_2 > best_L:
                            best_L = L_after_2
                            best_residues = residues.copy()
                            if best_L >= M - 10:
                                new_M = M * 2
                                counts = np.zeros(new_M + 1, dtype=np.int64)
                                for k, pk in enumerate(primes):
                                    rk = residues[k]
                                    start_m = max(0, (M + 1 - rk + pk - 1) // pk)
                                    first_idx = rk + start_m * pk
                                    if first_idx <= new_M:
                                        counts[first_idx::pk] += 1
                                M = new_M
                                current_L = calc_L(counts, M)
                                if current_L > best_L:
                                    best_L = current_L
                                    best_residues = residues.copy()
                    else:
                        # Revert both
                        counts[r2_new::p2] -= 1
                        counts[r2_old::p2] += 1
                        residues[j] = r2_old
                        
                        counts[r1_new::p1] -= 1
                        counts[r1_old::p1] += 1
                        residues[i] = r1_old
                else:
                    # Revert first
                    counts[r1_new::p1] -= 1
                    counts[r1_old::p1] += 1
                    residues[i] = r1_old
            else:
                # L_after_1 >= current_L, accept
                current_L = L_after_1
                if L_after_1 > best_L:
                    best_L = L_after_1
                    best_residues = residues.copy()
                    if best_L >= M - 10:
                        new_M = M * 2
                        counts = np.zeros(new_M + 1, dtype=np.int64)
                        for k, pk in enumerate(primes):
                            rk = residues[k]
                            start_m = max(0, (M + 1 - rk + pk - 1) // pk)
                            first_idx = rk + start_m * pk
                            if first_idx <= new_M:
                                counts[first_idx::pk] += 1
                        M = new_M
                        current_L = calc_L(counts, M)
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = residues.copy()

        # Cool down
        T = max(T_min, T * cooling_rate)

    return [int(r) for r in best_residues]

import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin (150s kill, 5s buffer)

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

    # We will search for L up to M. Start with M=2000 and extend if needed.
    M = 2000
    counts = np.zeros(M + 1, dtype=np.int64)

    # Initialize residues randomly
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

    def extend_M(new_M):
        nonlocal M, counts
        old_M = M
        old_counts = counts
        counts = np.zeros(new_M + 1, dtype=np.int64)
        # Copy old counts
        counts[:old_M + 1] = old_counts
        # For each prime, add contributions to the new range
        for j, pj in enumerate(primes):
            rj = residues[j]
            # Find first index >= old_M + 1 that is congruent to rj mod pj
            start_m = max(0, (old_M + 1 - rj + pj - 1) // pj)
            first_idx = rj + start_m * pj
            if first_idx <= new_M:
                counts[first_idx::pj] += 1
        M = new_M

    # Phase 1: Greedy initialization - try to improve coverage for each prime
    for i in range(n_primes):
        if time.time() - start_time > deadline - 30:
            break
        p = primes[i]
        r_old = residues[i]
        best_r = r_old
        best_local_L = current_L

        # Try all possible residues for this prime
        for r_candidate in range(p):
            if r_candidate == r_old:
                continue

            # Apply candidate
            counts[r_old::p] -= 1
            counts[r_candidate::p] += 1

            local_L = calc_L(counts, M)

            if local_L > best_local_L:
                best_local_L = local_L
                best_r = r_candidate
            else:
                # Revert
                counts[r_candidate::p] -= 1
                counts[r_old::p] += 1

        # Apply best
        if best_r != r_old:
            counts[r_old::p] -= 1
            counts[best_r::p] += 1
            residues[i] = best_r
            current_L = best_local_L
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
                if best_L >= M - 10:
                    extend_M(M * 2)
                    current_L = calc_L(counts, M)
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()

    # Phase 2: Main search loop - directed moves targeting the bottleneck
    while time.time() - start_time < deadline - 25:
        # Pick a random prime index
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        r_old = residues[i]

        # Determine target residue
        # If L < M, target is L+1
        x_target = current_L + 1
        r_target = x_target % p

        # With probability 0.7, pick r_target, else random
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

        if new_L > current_L:
            current_L = new_L
            residues[i] = r_new
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
                # Optional: Extend M if we are close to the limit
                if best_L >= M - 10:
                    extend_M(M * 2)
                    current_L = calc_L(counts, M)
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
        elif new_L == current_L:
            # Accept with small probability to explore
            if random.random() < 0.1:
                residues[i] = r_new
            else:
                # Revert
                counts[r_new::p] -= 1
                counts[r_old::p] += 1
        else:
            # Revert
            counts[r_new::p] -= 1
            counts[r_old::p] += 1

    # Phase 3: Local optimization - try all residues for each prime
    for i in range(n_primes):
        if time.time() - start_time > deadline - 10:
            break
        p = primes[i]
        r_old = residues[i]
        best_r = r_old
        best_local_L = current_L

        for r_candidate in range(p):
            if r_candidate == r_old:
                continue

            # Apply candidate
            counts[r_old::p] -= 1
            counts[r_candidate::p] += 1

            local_L = calc_L(counts, M)

            if local_L > best_local_L:
                best_local_L = local_L
                best_r = r_candidate
            else:
                # Revert
                counts[r_candidate::p] -= 1
                counts[r_old::p] += 1

        # Apply best
        if best_r != r_old:
            counts[r_old::p] -= 1
            counts[best_r::p] += 1
            residues[i] = best_r
            current_L = best_local_L
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
                if best_L >= M - 10:
                    extend_M(M * 2)
                    current_L = calc_L(counts, M)
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()

    # Phase 4: Final random polish
    while time.time() - start_time < deadline - 2:
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        r_old = residues[i]
        r_new = random.randint(0, p - 1)

        if r_new == r_old:
            continue

        counts[r_old::p] -= 1
        counts[r_new::p] += 1

        new_L = calc_L(counts, M)

        if new_L >= current_L:
            residues[i] = r_new
            if new_L > current_L:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
                    if best_L >= M - 10:
                        extend_M(M * 2)
                        current_L = calc_L(counts, M)
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = residues.copy()
        else:
            counts[r_new::p] -= 1
            counts[r_old::p] += 1

    return [int(r) for r in best_residues]

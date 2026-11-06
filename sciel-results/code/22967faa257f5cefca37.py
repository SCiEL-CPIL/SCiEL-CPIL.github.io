"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Generate primes up to 1000 using sieve
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

    # Precompute coverage masks: for each prime i and residue r, which positions in [1, MAX_L] are covered
    MAX_L = 5000
    cover = np.zeros((n_primes, 1000, MAX_L + 1), dtype=np.bool_)
    for i, p in enumerate(primes):
        for r in range(p):
            if r == 0:
                first = p
            else:
                first = r
            if 1 <= first <= MAX_L:
                cover[i, r, first::p] = True

    def get_counts(residues):
        """Compute coverage count array for given residues."""
        counts = np.zeros(MAX_L + 1, dtype=np.int32)
        for i in range(n_primes):
            counts += cover[i, residues[i], :].astype(np.int32)
        return counts

    def find_L_from_counts(counts):
        """Find largest L such that counts[1..L] >= 2."""
        gap_mask = (counts[1:MAX_L+1] < 2)
        if not gap_mask.any():
            return MAX_L
        first_gap = int(np.argmax(gap_mask))
        return first_gap  # 0-indexed, so L = first_gap

    def evaluate(residues):
        counts = get_counts(residues)
        return find_L_from_counts(counts)

    # Initial assignment: try several strategies
    best_residues = None
    best_L = 0

    # Strategy 1: r_p = 1 for all (each prime covers x ≡ 1 mod p)
    residues_init = [1 % p for p in primes]
    L_init = evaluate(residues_init)
    if L_init > best_L:
        best_L = L_init
        best_residues = residues_init[:]

    # Strategy 2: r_p = 2 for all
    residues_init2 = [2 % p for p in primes]
    L_init2 = evaluate(residues_init2)
    if L_init2 > best_L:
        best_L = L_init2
        best_residues = residues_init2[:]

    # Strategy 3: Random initializations
    random.seed(42)
    for trial in range(15):
        residues_rand = [random.randint(0, p - 1) for p in primes]
        Lr = evaluate(residues_rand)
        if Lr > best_L:
            best_L = Lr
            best_residues = residues_rand[:]

    # Gap-directed local search
    if best_residues is not None:
        current = best_residues[:]
        current_L = best_L
        counts = get_counts(current)

        random.seed(123)
        no_improve = 0
        max_no_improve = 3000

        while time.time() < deadline - 8.0 and no_improve < max_no_improve:
            # Identify the first gap position (current_L + 1)
            gap_pos = current_L + 1
            if gap_pos > MAX_L:
                break

            # Find primes that could cover gap_pos: those where (gap_pos % p) can be matched
            # i.e., any prime can cover any position by choosing r = gap_pos % p
            # But we want to find primes whose current residue does NOT cover gap_pos,
            # and changing them to cover gap_pos might help

            # Strategy: pick a random prime, try to set its residue to cover the gap position
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = current[i]
            target_r = gap_pos % p  # residue that would cover gap_pos

            if target_r == old_r:
                # Already covers the gap; try a different approach
                no_improve += 1
                continue

            # Try setting prime i to target_r
            counts -= cover[i, old_r, :].astype(np.int32)
            counts += cover[i, target_r, :].astype(np.int32)
            current[i] = target_r

            new_L = find_L_from_counts(counts)
            if new_L > current_L:
                current_L = new_L
                no_improve = 0
                if current_L > best_L:
                    best_L = current_L
                    best_residues = current[:]
            else:
                # Revert
                counts -= cover[i, target_r, :].astype(np.int32)
                counts += cover[i, old_r, :].astype(np.int32)
                current[i] = old_r
                no_improve += 1

            # Also try random perturbations occasionally
            if no_improve % 10 == 0:
                i2 = random.randint(0, n_primes - 1)
                p2 = primes[i2]
                old_r2 = current[i2]
                new_r2 = random.randint(0, p2 - 1)
                if new_r2 != old_r2:
                    counts -= cover[i2, old_r2, :].astype(np.int32)
                    counts += cover[i2, new_r2, :].astype(np.int32)
                    current[i2] = new_r2
                    new_L2 = find_L_from_counts(counts)
                    if new_L2 > current_L:
                        current_L = new_L2
                        no_improve = 0
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = current[:]
                    else:
                        counts -= cover[i2, new_r2, :].astype(np.int32)
                        counts += cover[i2, old_r2, :].astype(np.int32)
                        current[i2] = old_r2

        # Final pass: try all residues for small primes to squeeze out more
        if time.time() < deadline - 5.0:
            counts = get_counts(current)
            current_L = find_L_from_counts(counts)
            if current_L > best_L:
                best_L = current_L
                best_residues = current[:]

            for i in range(min(30, n_primes)):  # Small primes first
                if time.time() > deadline - 5.0:
                    break
                p = primes[i]
                old_r = current[i]
                best_r = old_r
                best_local_L = current_L

                for new_r in range(p):
                    if new_r == old_r:
                        continue
                    counts -= cover[i, old_r, :].astype(np.int32)
                    counts += cover[i, new_r, :].astype(np.int32)
                    current[i] = new_r
                    new_L = find_L_from_counts(counts)
                    if new_L > best_local_L:
                        best_local_L = new_L
                        best_r = new_r
                    else:
                        counts -= cover[i, new_r, :].astype(np.int32)
                        counts += cover[i, old_r, :].astype(np.int32)
                        current[i] = old_r

                if best_r != old_r:
                    counts -= cover[i, old_r, :].astype(np.int32)
                    counts += cover[i, best_r, :].astype(np.int32)
                    current[i] = best_r
                    current_L = best_local_L
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = current[:]

    return best_residues if best_residues is not None else [1 % p for p in primes]
# EVOLVE-BLOCK-END

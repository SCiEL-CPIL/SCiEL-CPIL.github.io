"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

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
    primes_arr = np.array(primes, dtype=np.int64)

    def compute_counts(residues, bound=2000):
        """Vectorized computation of coverage counts for integers 1..bound."""
        counts = np.zeros(bound + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes[i]
            r = residues[i]
            # Find first integer >= 1 that is ≡ r (mod p)
            if r == 0:
                first = p
            else:
                first = r
            if first < 1:
                first += p
            # Add 1 to all positions first, first+p, first+2p, ...
            positions = np.arange(first, bound + 1, p)
            if len(positions) > 0:
                counts[positions] += 1
        return counts

    def get_L_from_counts(counts, bound=2000):
        """Find largest L such that counts[1..L] >= 2."""
        # Find first position where count < 2
        below = np.where(counts[1:bound+1] < 2)[0]
        if len(below) == 0:
            return bound
        return below[0]  # 0-indexed, so position below[0]+1 is the gap, L = below[0]

    def evaluate(residues, bound=2000):
        """Compute L for given residues."""
        counts = compute_counts(residues, bound)
        return get_L_from_counts(counts, bound)

    # Initial assignments
    best_residues = None
    best_L = 0

    # Strategy 1: Try several initializations
    initial_strategies = [
        [0] * n_primes,
        [1 % p for p in primes],
        [((i + 2) // 2) % p for i, p in enumerate(primes)],
        [random.randint(0, p - 1) for p in primes],
        [random.randint(0, p - 1) for p in primes],
        [random.randint(0, p - 1) for p in primes],
    ]

    for residues in initial_strategies:
        L = evaluate(residues)
        if L > best_L:
            best_L = L
            best_residues = residues[:]

    if best_residues is None:
        best_residues = [0] * n_primes
        best_L = evaluate(best_residues)

    # Gap-filling local search
    current = best_residues[:]
    current_L = best_L
    random.seed(42)

    def try_fix_gap(current, current_L, bound=2000):
        """Try to fix the first gap by changing one residue."""
        if current_L >= bound:
            return current, current_L, False

        counts = compute_counts(current, bound)
        # Find first gap
        below = np.where(counts[1:bound+1] < 2)[0]
        if len(below) == 0:
            return current, bound, True

        gap_pos = below[0] + 1  # 1-indexed position of the gap
        gap_count = counts[gap_pos]

        # For each prime, check if we can cover gap_pos by changing its residue
        # gap_pos ≡ r (mod p) means r = gap_pos % p
        # But we need to make sure this doesn't break other positions

        best_new_L = current_L
        best_new_residues = None

        # Try primes in order of increasing size (smaller primes cover more)
        order = sorted(range(n_primes), key=lambda i: primes[i])

        for i in order[:50]:  # Try first 50 primes
            p = primes[i]
            target_r = gap_pos % p
            if target_r == 0:
                target_r = p  # No, residue must be in [0, p-1]
            target_r = gap_pos % p

            old_r = current[i]
            if target_r == old_r:
                continue

            # Try changing this residue
            current[i] = target_r
            new_counts = compute_counts(current, bound)
            new_L = get_L_from_counts(new_counts, bound)

            if new_L > best_new_L:
                best_new_L = new_L
                best_new_residues = current[:]

            current[i] = old_r

            if time.time() > deadline - 10.0:
                break

        if best_new_residues is not None:
            return best_new_residues, best_new_L, True
        return current, current_L, False

    # Iterative gap-filling
    no_improve = 0
    max_no_improve = 200

    while time.time() < deadline - 10.0 and no_improve < max_no_improve:
        if current_L >= 2000:
            break

        new_current, new_L, improved = try_fix_gap(current, current_L)

        if improved and new_L > current_L:
            current = new_current
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_residues = current[:]
            no_improve = 0
        else:
            no_improve += 1

            # If stuck, try random perturbations
            if no_improve % 10 == 0:
                # Randomly change a few residues
                for _ in range(3):
                    i = random.randint(0, n_primes - 1)
                    current[i] = random.randint(0, primes[i] - 1)

                new_counts = compute_counts(current, 2000)
                new_L = get_L_from_counts(new_counts, 2000)

                if new_L > current_L:
                    current_L = new_L
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = current[:]
                    no_improve = 0
                else:
                    # Revert to best
                    current = best_residues[:]
                    current_L = best_L

    # Final local search with small perturbations
    if time.time() < deadline - 5.0:
        current = best_residues[:]
        current_L = best_L
        no_improve = 0

        while time.time() < deadline - 5.0 and no_improve < 500:
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = current[i]

            # Try a few random values
            found_improvement = False
            for _ in range(5):
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
                current[i] = new_r
                new_L = evaluate(current, 2000)
                if new_L > current_L:
                    current_L = new_L
                    found_improvement = True
                    break
                else:
                    current[i] = old_r

            if found_improvement:
                no_improve = 0
                if current_L > best_L:
                    best_L = current_L
                    best_residues = current[:]
            else:
                no_improve += 1

    return best_residues if best_residues is not None else [0] * n_primes
# EVOLVE-BLOCK-END

import numpy as np
import time
import random
from math import isqrt

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(limit) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(1000)
    n_primes = len(primes)
    M = 100000  # max L to check

    # We'll work with a window size for efficiency
    # But we need to track the full prefix for the score
    # Let's use a different approach: track coverage counts for a growing window

    # Strategy: Greedy + local search
    # Key insight: assign residues to maximize coverage of early integers
    # Small primes should be assigned residues that hit many early numbers

    # For each prime p, residue r covers numbers r, r+p, r+2p, ...
    # To cover early numbers, we want r to be small (0 or 1) for large primes
    # But we need at least 2 covers per number

    # Better approach: think of it as a scheduling problem
    # We want to maximize the prefix where min coverage >= 2

    # Let's try a constructive approach:
    # 1. Start with all residues = 0
    # 2. Find the first gap (number with < 2 coverage)
    # 3. Fix it by adjusting residues of primes that could cover that number
    # 4. Repeat until no more gaps or time runs out

    # Initialize: all residues = 0
    residues = np.zeros(n_primes, dtype=np.int64)
    for i, p in enumerate(primes):
        residues[i] = 0

    # Compute coverage counts for 1..M
    counts = np.zeros(M + 1, dtype=np.int64)
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            counts[p::p] += 1  # covers p, 2p, 3p, ...
            # Note: 0 is not in 1..M, so we start at p
        else:
            counts[r::p] += 1

    def calc_L(counts):
        if counts[1] < 2:
            return 0
        mask = ~(counts[1:] >= 2)
        if not np.any(mask):
            return M
        first_false_idx = np.argmax(mask)
        return first_false_idx

    current_L = calc_L(counts)
    best_L = current_L
    best_residues = residues.copy()

    def try_improve():
        nonlocal current_L, best_L, best_residues, counts
        # Find the first gap
        if current_L >= M:
            return True
        gap_pos = current_L + 1  # first position with < 2 coverage
        if counts[gap_pos] >= 2:
            # Recalculate L
            current_L = calc_L(counts)
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
            return current_L >= M

        # Find primes that could cover gap_pos
        # Prime p can cover gap_pos if we set residue to gap_pos % p
        candidates = []
        for i, p in enumerate(primes):
            if p > gap_pos:
                # p can cover gap_pos only if residue = gap_pos (but residue < p, so if gap_pos < p, residue = gap_pos)
                # Actually residue must be in [0, p-1], so if gap_pos < p, we can set r = gap_pos
                if gap_pos < p:
                    candidates.append((i, p, gap_pos))
            else:
                r_needed = gap_pos % p
                candidates.append((i, p, r_needed))

        # Try each candidate: change residue to cover gap_pos
        # But we need to check if this doesn't break other positions
        # For efficiency, try candidates in order of prime size (smaller primes first, as they affect more positions)
        candidates.sort(key=lambda x: x[1])

        improved = False
        for i, p, r_new in candidates:
            r_old = residues[i]
            if r_new == r_old:
                continue

            # Update counts
            if r_old == 0:
                counts[p::p] -= 1
            else:
                counts[r_old::p] -= 1

            if r_new == 0:
                counts[p::p] += 1
            else:
                counts[r_new::p] += 1

            new_L = calc_L(counts)

            if new_L > current_L:
                current_L = new_L
                residues[i] = r_new
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
                improved = True
                break
            else:
                # Revert
                if r_new == 0:
                    counts[p::p] -= 1
                else:
                    counts[r_new::p] -= 1

                if r_old == 0:
                    counts[p::p] += 1
                else:
                    counts[r_old::p] += 1

                residues[i] = r_old

        if not improved:
            # Try local search: random perturbation
            if time.time() - start_time > deadline - 5:
                return False

            # Pick a random prime and change its residue
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            r_old = residues[i]
            r_new = random.randint(0, p - 1)
            if r_new == r_old:
                return try_improve()

            # Update
            if r_old == 0:
                counts[p::p] -= 1
            else:
                counts[r_old::p] -= 1

            if r_new == 0:
                counts[p::p] += 1
            else:
                counts[r_new::p] += 1

            new_L = calc_L(counts)

            # Accept if it doesn't make things worse, or with some probability
            if new_L >= current_L:
                current_L = new_L
                residues[i] = r_new
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
                return try_improve()
            else:
                # Revert
                if r_new == 0:
                    counts[p::p] -= 1
                else:
                    counts[r_new::p] -= 1

                if r_old == 0:
                    counts[p::p] += 1
                else:
                    counts[r_old::p] += 1

                residues[i] = r_old
                return try_improve()

    # Main loop
    while time.time() - start_time < deadline - 5:
        if not try_improve():
            break
        if best_L >= M:
            break

    return [int(r) for r in best_residues]

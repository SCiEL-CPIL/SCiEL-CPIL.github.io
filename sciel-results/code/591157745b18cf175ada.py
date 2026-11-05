import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i:limit + 1:i] = [False] * len(sieve[i * i:limit + 1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000

    # Precompute step arrays for each prime (positions in [1, MAX_L])
    # For prime p with residue r, positions are r, r+p, r+2p, ...
    # We'll use this for fast count updates

    def build_counts(residues, window):
        counts = np.zeros(window + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            start = r if r > 0 else p
            if start <= window:
                counts[start::p] += 1
        return counts

    def find_L(counts, current_L, max_L):
        """Find the largest L such that all positions 1..L have count >= 2."""
        if current_L >= max_L:
            return max_L
        if current_L > 0:
            # Check if current prefix is still valid
            if np.min(counts[1:current_L + 1]) < 2:
                bad = np.where(counts[1:current_L + 1] < 2)[0]
                return bad[0]
            # Try to extend
            L = current_L
            rest = counts[L + 1:]
            bad_idx = np.where(rest < 2)[0]
            if len(bad_idx) == 0:
                return max_L
            else:
                return L + bad_idx[0]
        else:
            bad_idx = np.where(counts[1:] < 2)[0]
            if len(bad_idx) == 0:
                return max_L
            return bad_idx[0]

    # --- Phase 1: Multi-start Greedy with look-ahead ---
    best_init_L = 0
    best_init_res = [0] * n_primes
    best_init_counts = None

    GREEDY_WINDOW = 8000

    for attempt in range(8):
        residues = [0] * n_primes
        counts_greedy = np.zeros(GREEDY_WINDOW + 1, dtype=np.int32)

        for idx in range(n_primes):
            p = primes[idx]
            best_r = 0
            best_score = -1

            # For small primes, try all residues; for large primes, sample strategically
            if p <= 200:
                candidates = range(p)
            else:
                # Sample residues that target the current frontier
                # Find where the current coverage is weakest
                frontier = 1
                for pos in range(1, min(GREEDY_WINDOW + 1, current_L + 100)):
                    if counts_greedy[pos] < 2:
                        frontier = pos
                        break
                
                # Generate candidates around the frontier
                candidates = set()
                # Residues that cover the frontier
                candidates.add(frontier % p)
                # Neighbors
                for offset in range(-5, 6):
                    candidates.add((frontier + offset) % p)
                # Random samples
                if p > 50:
                    rand_samples = np.random.choice(p, size=min(p, 60), replace=False)
                    candidates.update(rand_samples.tolist())
                candidates = list(candidates)

            for r in candidates:
                start = r if r > 0 else p
                if start > GREEDY_WINDOW:
                    continue
                positions = np.arange(start, GREEDY_WINDOW + 1, p)
                current_counts = counts_greedy[positions]
                deficient = np.sum(current_counts < 2)
                already_covered = np.sum(current_counts >= 2)
                # Weight by how close to the frontier
                pos_weight = np.arange(len(positions)) * 0.01
                score = deficient - 0.2 * already_covered + np.sum(pos_weight * (current_counts < 2))
                if score > best_score:
                    best_score = score
                    best_r = r

            residues[idx] = best_r
            r = best_r
            start = r if r > 0 else p
            if start <= GREEDY_WINDOW:
                counts_greedy[start::p] += 1

        # Evaluate full length
        full_counts = build_counts(residues, MAX_L)
        L = find_L(full_counts, 0, MAX_L)
        if L > best_init_L:
            best_init_L = L
            best_init_res = residues.copy()
            best_init_counts = full_counts

    # Start refinement from the best greedy result
    residues = best_init_res
    counts = best_init_counts
    current_L = best_init_L
    best_L = current_L
    best_residues = residues.copy()

    # --- Phase 2: Simulated Annealing with adaptive parameters ---
    T = 8.0
    cooling_rate = 0.99997
    min_T = 0.1

    # Track stagnation for adaptive cooling
    stagnation_counter = 0
    stagnation_threshold = 500

    while time.time() < deadline:
        # Pick a random prime to perturb
        i = np.random.randint(n_primes)
        p = primes[i]
        old_r = residues[i]
        
        # Generate new residue - bias towards covering the frontier
        if np.random.random() < 0.3 and current_L > 0:
            # Targeted perturbation: try residues that cover positions near the frontier
            target_pos = current_L + np.random.randint(1, min(20, MAX_L - current_L + 1))
            new_r = target_pos % p
        else:
            new_r = np.random.randint(p)

        if new_r == old_r:
            continue

        # Apply change
        start_old = old_r if old_r > 0 else p
        if start_old <= MAX_L:
            counts[start_old::p] -= 1

        start_new = new_r if new_r > 0 else p
        if start_new <= MAX_L:
            counts[start_new::p] += 1

        # Calculate new L
        new_L = find_L(counts, current_L, MAX_L)

        # Acceptance
        if new_L >= current_L:
            # Accept
            residues[i] = new_r
            current_L = new_L
            stagnation_counter = 0
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            # Reject or Accept with probability
            delta = new_L - current_L
            if delta > -200:
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    residues[i] = new_r
                    current_L = new_L
                    stagnation_counter += 1
                else:
                    # Revert
                    residues[i] = old_r
                    if start_new <= MAX_L:
                        counts[start_new::p] -= 1
                    if start_old <= MAX_L:
                        counts[start_old::p] += 1
                    stagnation_counter += 1
            else:
                # Revert
                residues[i] = old_r
                if start_new <= MAX_L:
                    counts[start_new::p] -= 1
                if start_old <= MAX_L:
                    counts[start_old::p] += 1
                stagnation_counter += 1

        # Adaptive cooling
        T *= cooling_rate
        if T < min_T:
            T = min_T

        # If stagnating, occasionally reheat
        if stagnation_counter > stagnation_threshold:
            T = max(T, 2.0)
            stagnation_counter = 0

    return best_residues

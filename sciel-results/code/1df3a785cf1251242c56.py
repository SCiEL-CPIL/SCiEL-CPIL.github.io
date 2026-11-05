"""Rewritten program: greedy submodular maximization for double-coverage."""
import time
import random
import numpy as np

_DEADLINE = 145.0  # safety margin before 150s wall-clock limit


def _primes_up_to(n):
    """Sieve of Eratosthenes for primes up to n."""
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(n + 1) if sieve[i]]


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE

    primes = _primes_up_to(1000)
    n = len(primes)
    MAX_L = 100000

    def build_coverage(residues):
        """Build the coverage array from scratch."""
        cov = np.zeros(MAX_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            start = r if r != 0 else p
            if start <= MAX_L:
                cov[start::p] += 1
        return cov

    def find_first_gap(cov):
        """Find the first x >= 1 where cov[x] < 2."""
        bad = np.where(cov[1:] < 2)[0]
        if len(bad) == 0:
            return MAX_L
        return int(bad[0])

    def update_coverage(cov, idx, old_r, new_r):
        """Update coverage array in-place when changing residue at idx."""
        p = primes[idx]
        old_start = old_r if old_r != 0 else p
        if old_start <= MAX_L:
            cov[old_start::p] -= 1
        new_start = new_r if new_r != 0 else p
        if new_start <= MAX_L:
            cov[new_start::p] += 1

    # Precompute for each prime the list of positions it covers for each residue
    # This is expensive to precompute fully, so we'll compute on the fly but cache hot ones

    best_L_global = 0
    best_residues_global = [0] * n

    # Phase 1: Greedy submodular maximization
    # Start with all residues = 0 (or some initial), then greedily pick the best residue
    # for each prime that maximizes the marginal gain in doubly-covered prefix length.

    def greedy_construct():
        """Greedy submodular maximization: assign residues one by one to maximize L."""
        residues = [0] * n
        coverage = np.zeros(MAX_L + 1, dtype=np.int32)

        # We'll process primes in order of increasing size (small primes first)
        # because they have more "choices" and affect more positions.
        # Actually, process in order of decreasing impact: small primes cover more positions.
        order = list(range(n))
        random.shuffle(order)

        for idx in order:
            p = primes[idx]
            old_r = residues[idx]

            # Try all possible residues for this prime
            best_r = old_r
            best_L = find_first_gap(coverage)

            # For large primes, we can't try all residues, so sample
            if p <= 200:
                candidates = list(range(p))
            else:
                # Sample residues: focus on residues that cover the current gap
                target = best_L + 1
                candidates = set()
                candidates.add(old_r)
                candidates.add(target % p)
                # Add some random
                for _ in range(50):
                    candidates.add(random.randint(0, p - 1))
                # Add residues that cover positions near the gap
                for offset in range(-10, 11):
                    pos = best_L + 1 + offset
                    if 1 <= pos <= MAX_L:
                        candidates.add(pos % p)
                candidates = list(candidates)

            for r in candidates:
                if r == old_r:
                    continue
                if time.time() > deadline - 15:
                    break

                update_coverage(coverage, idx, old_r, r)
                new_L = find_first_gap(coverage)
                update_coverage(coverage, idx, r, old_r)

                if new_L > best_L:
                    best_L = new_L
                    best_r = r

            # Commit best
            if best_r != old_r:
                update_coverage(coverage, idx, old_r, best_r)
                residues[idx] = best_r

        return residues, coverage, find_first_gap(coverage)

    # Phase 2: Local search refinement
    def local_search(residues, coverage, current_L, max_passes=30):
        """Iterative local search to improve L."""
        improved = True
        pass_count = 0
        while improved and pass_count < max_passes and time.time() < deadline - 10:
            improved = False
            pass_count += 1
            shuffled_indices = list(range(n))
            random.shuffle(shuffled_indices)

            for idx in shuffled_indices:
                if time.time() > deadline - 10:
                    break

                p = primes[idx]
                old_r = residues[idx]
                best_r_for_idx = old_r
                best_L_for_idx = current_L

                # Generate candidates
                candidates = set()
                candidates.add(old_r)

                # Target the next gap
                target = current_L + 1
                if target <= MAX_L:
                    candidates.add(target % p)

                # Target positions where coverage is exactly 1
                check_range = min(current_L + 100, MAX_L)
                if check_range >= 1:
                    one_cov_positions = np.where(coverage[1:check_range + 1] == 1)[0] + 1
                    if len(one_cov_positions) > 0:
                        sample_size = min(30, len(one_cov_positions))
                        sampled = random.sample(list(one_cov_positions), sample_size)
                        for pos in sampled:
                            candidates.add(pos % p)

                # Random candidates
                num_random = min(30, p)
                for _ in range(num_random):
                    candidates.add(random.randint(0, p - 1))

                # Neighbors
                candidates.add((old_r + 1) % p)
                candidates.add((old_r - 1) % p)

                # For small primes, try all
                if p <= 30:
                    candidates = set(range(p))

                for r in candidates:
                    if r == old_r:
                        continue
                    if time.time() > deadline - 10:
                        break

                    update_coverage(coverage, idx, old_r, r)
                    new_L = find_first_gap(coverage)
                    update_coverage(coverage, idx, r, old_r)

                    if new_L > best_L_for_idx:
                        best_L_for_idx = new_L
                        best_r_for_idx = r

                if best_r_for_idx != old_r:
                    update_coverage(coverage, idx, old_r, best_r_for_idx)
                    residues[idx] = best_r_for_idx
                    current_L = best_L_for_idx
                    improved = True

        return residues, coverage, current_L

    # Run multiple greedy constructions with different random seeds
    for attempt in range(20):
        if time.time() > deadline - 20:
            break

        random.seed(attempt * 12345 + 678)

        residues, coverage, current_L = greedy_construct()

        if current_L > best_L_global:
            best_L_global = current_L
            best_residues_global = residues[:]

        # Refine with local search
        residues, coverage, current_L = local_search(residues, coverage, current_L, max_passes=20)

        if current_L > best_L_global:
            best_L_global = current_L
            best_residues_global = residues[:]

    # Final polish on the best solution
    if best_L_global > 0 and time.time() < deadline - 5:
        residues = best_residues_global[:]
        coverage = build_coverage(residues)
        current_L = find_first_gap(coverage)

        # Additional targeted passes
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            target = current_L + 1
            if target > MAX_L:
                break
            for idx in range(n):
                p = primes[idx]
                target_r = target % p
                if residues[idx] == target_r:
                    continue
                old_r = residues[idx]
                update_coverage(coverage, idx, old_r, target_r)
                new_L = find_first_gap(coverage)
                update_coverage(coverage, idx, target_r, old_r)
                if new_L > current_L:
                    update_coverage(coverage, idx, old_r, target_r)
                    residues[idx] = target_r
                    current_L = new_L
                    if current_L > best_L_global:
                        best_L_global = current_L
                        best_residues_global = residues[:]
                    break

    return [int(r) for r in best_residues_global]

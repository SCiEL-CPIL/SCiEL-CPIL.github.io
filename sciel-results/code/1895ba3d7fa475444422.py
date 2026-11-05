import time
import random
from fractions import Fraction
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]

    primes = sieve(1000)
    prime_set = set(primes)

    # Fast coverage computation using numpy
    def compute_L(classes, max_check=100000):
        """Compute the largest L such that 1..L is all covered."""
        if not classes:
            return 0
        covered = np.zeros(max_check + 1, dtype=bool)
        for p, r in classes:
            start = r if r > 0 else p
            if start <= max_check:
                covered[start - 1::p] = True  # 0-indexed, so start-1

        # Find first False
        false_indices = np.where(~covered)[0]
        if len(false_indices) == 0:
            return max_check
        return false_indices[0]  # This is 0-indexed, so L = false_indices[0]

    # Precompute reciprocal sums for candidate prime sets
    def build_prime_set(base_primes, fill_strategy='smallest_first'):
        """Build a prime set starting from base_primes, filling remaining budget."""
        s = sum(Fraction(1, p) for p in base_primes)
        if s > 1:
            return []
        final_set = list(base_primes)
        base_set = set(base_primes)

        if fill_strategy == 'smallest_first':
            candidates = [p for p in primes if p not in base_set]
        else:
            candidates = [p for p in primes if p not in base_set]

        for p in candidates:
            if s + Fraction(1, p) <= 1:
                final_set.append(p)
                s += Fraction(1, p)

        return final_set

    # Generate several candidate prime sets with different strategies
    candidate_sets = []

    # Strategy 1: Include small primes for high density
    candidate_sets.append(build_prime_set([2]))
    candidate_sets.append(build_prime_set([3]))
    candidate_sets.append(build_prime_set([2, 3]))
    candidate_sets.append(build_prime_set([2, 5]))
    candidate_sets.append(build_prime_set([2, 7]))
    candidate_sets.append(build_prime_set([3, 5]))
    candidate_sets.append(build_prime_set([5, 7]))

    # Strategy 2: Skip some small primes to allow more medium primes
    candidate_sets.append(build_prime_set([2, 11]))
    candidate_sets.append(build_prime_set([2, 13]))
    candidate_sets.append(build_prime_set([3, 7]))
    candidate_sets.append(build_prime_set([5, 11]))

    # Strategy 3: No small primes, many medium/large
    candidate_sets.append(build_prime_set([]))

    # Strategy 4: Targeted sets that might cover longer prefixes
    candidate_sets.append(build_prime_set([2, 3, 5]))
    candidate_sets.append(build_prime_set([2, 3, 7]))
    candidate_sets.append(build_prime_set([2, 5, 7]))
    candidate_sets.append(build_prime_set([3, 5, 7]))

    # Remove empty sets and duplicates
    seen = set()
    unique_sets = []
    for s in candidate_sets:
        if s:
            key = tuple(sorted(s))
            if key not in seen:
                seen.add(key)
                unique_sets.append(s)
    candidate_sets = unique_sets

    best_L = 0
    best_classes = []

    # For each prime set, optimize residues using simulated annealing
    for prime_set_idx, prime_set in enumerate(candidate_sets):
        if time.time() > deadline - 20:
            break

        P = sorted(prime_set)
        n_primes = len(P)

        # Initial residues: start with a greedy approach
        residues = [0] * n_primes
        for i, p in enumerate(P):
            residues[i] = (i + 1) % p

        # Evaluate initial
        classes = [[P[i], residues[i]] for i in range(n_primes)]
        L = compute_L(classes, max_check=10000)

        if L > best_L:
            best_L = L
            best_classes = classes.copy()

        # Simulated annealing
        T = 100.0  # Initial temperature
        cooling_rate = 0.995
        min_T = 0.1
        max_iterations = 3000

        current_L = L
        current_residues = residues.copy()

        iteration = 0
        while iteration < max_iterations and time.time() < deadline - 15:
            iteration += 1

            # Choose a random prime to modify
            i = random.randint(0, n_primes - 1)
            p = P[i]
            old_r = current_residues[i]

            # Choose a new residue
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue

            current_residues[i] = new_r
            classes = [[P[j], current_residues[j]] for j in range(n_primes)]
            new_L = compute_L(classes, max_check=10000)

            # Accept or reject
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_classes = classes.copy()
            else:
                # Accept with probability exp((new_L - current_L) / T)
                delta = new_L - current_L
                if delta < 0:
                    prob = np.exp(delta / T)
                    if random.random() > prob:
                        # Reject
                        current_residues[i] = old_r

            # Cool down
            T *= cooling_rate
            if T < min_T:
                T = min_T

        # Local search refinement
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(n_primes):
                if time.time() > deadline - 10:
                    break

                p = P[i]
                old_r = current_residues[i]
                best_r = old_r
                best_L_i = current_L

                # Try residues 0 to min(p-1, 30)
                for r in range(min(p, 30)):
                    if r == old_r:
                        continue

                    current_residues[i] = r
                    classes = [[P[j], current_residues[j]] for j in range(n_primes)]
                    L_i = compute_L(classes, max_check=10000)

                    if L_i > best_L_i:
                        best_L_i = L_i
                        best_r = r

                if best_r != old_r:
                    current_residues[i] = best_r
                    current_L = best_L_i
                    improved = True

                    if current_L > best_L:
                        best_L = current_L
                        best_classes = [[P[j], current_residues[j]] for j in range(n_primes)]

    # Final verification with larger max_check
    if best_classes:
        final_L = compute_L(best_classes, max_check=100000)
        if final_L > best_L:
            best_L = final_L

    return best_classes

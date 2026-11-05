import time
import numpy as np
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    # 1. Precompute primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    max_L_target = 100000

    # 2. Generate diverse candidate prime sets
    def generate_greedy_set(exclude=set(), include=set()):
        sorted_primes = sorted(primes)
        current_sum = Fraction(0, 1)
        selected = []
        for p in sorted_primes:
            if p in exclude:
                continue
            if p in include:
                if current_sum + Fraction(1, p) <= 1:
                    selected.append(p)
                    current_sum += Fraction(1, p)
                continue
            if current_sum + Fraction(1, p) <= 1:
                selected.append(p)
                current_sum += Fraction(1, p)
        return selected

    candidates = []
    # Base greedy
    candidates.append(generate_greedy_set())
    # Exclude 2
    candidates.append(generate_greedy_set(exclude={2}))
    # Exclude 2,3
    candidates.append(generate_greedy_set(exclude={2, 3}))
    # Include specific small primes, exclude others
    candidates.append(generate_greedy_set(exclude={2, 3, 5}, include={7, 11, 13}))
    candidates.append(generate_greedy_set(exclude={2, 5}, include={3, 7, 11}))
    # A set that might have more flexibility
    candidates.append(generate_greedy_set(exclude={2, 3, 5, 7}, include={11, 13, 17, 19, 23}))

    best_L = 0
    best_classes = []

    # 3. For each candidate, run simulated annealing
    for P in candidates:
        if time.time() > deadline - 15:
            break

        P_sorted = sorted(P)
        n_primes = len(P_sorted)

        # Build index mapping
        prime_indices = {p: i for i, p in enumerate(P_sorted)}

        # Initialize with greedy assignment
        residues = np.zeros(n_primes, dtype=np.int32)
        counts = np.zeros(max_L_target + 1, dtype=np.int32)

        # Greedy init: for each prime in order, pick residue that covers most uncovered
        for i, p in enumerate(P_sorted):
            best_r = 0
            best_gain = -1
            for r in range(p):
                start = r if r > 0 else p
                if start > max_L_target:
                    continue
                idx = np.arange(start, max_L_target + 1, p)
                gain = np.sum(counts[idx] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            residues[i] = best_r
            start = best_r if best_r > 0 else p
            if start <= max_L_target:
                idx = np.arange(start, max_L_target + 1, p)
                counts[idx] += 1

        def get_L(counts):
            uncovered = np.where(counts[1:] == 0)[0]
            if len(uncovered) == 0:
                return max_L_target
            return int(uncovered[0])

        current_L = get_L(counts)
        if current_L > best_L:
            best_L = current_L
            best_classes = [[P_sorted[i], int(residues[i])] for i in range(n_primes)]

        # Simulated Annealing
        T_start = 100.0
        T_end = 0.01
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / 5000)  # 5000 iterations
        no_improve_count = 0
        best_T_L = current_L

        for it in range(5000):
            if time.time() > deadline - 5:
                break

            # Pick a random prime index
            i = random.randint(0, n_primes - 1)
            p = P_sorted[i]
            old_r = int(residues[i])

            # Pick a new random residue
            if p == 2:
                new_r = 1 - old_r
            else:
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue

            # Compute delta in L
            # Remove old, add new, compute new L, then decide
            start_old = old_r if old_r > 0 else p
            start_new = new_r if new_r > 0 else p

            # Temporarily update
            if start_old <= max_L_target:
                idx_old = np.arange(start_old, max_L_target + 1, p)
                counts[idx_old] -= 1
            if start_new <= max_L_target:
                idx_new = np.arange(start_new, max_L_target + 1, p)
                counts[idx_new] += 1

            new_L = get_L(counts)

            if new_L > current_L:
                # Accept
                residues[i] = new_r
                current_L = new_L
                no_improve_count = 0
                if current_L > best_L:
                    best_L = current_L
                    best_classes = [[P_sorted[j], int(residues[j])] for j in range(n_primes)]
            else:
                # Maybe accept worse
                delta = new_L - current_L
                if delta < 0 and random.random() < np.exp(delta / max(T, 1e-10)):
                    residues[i] = new_r
                    current_L = new_L
                    no_improve_count += 1
                else:
                    # Revert
                    if start_new <= max_L_target:
                        idx_new = np.arange(start_new, max_L_target + 1, p)
                        counts[idx_new] -= 1
                    if start_old <= max_L_target:
                        idx_old = np.arange(start_old, max_L_target + 1, p)
                        counts[idx_old] += 1
                    no_improve_count += 1

            # Cool
            T *= cooling

            # Early stop if stuck too long
            if no_improve_count > 1000 and T < 1.0:
                break

    # Final verification
    if not best_classes:
        P = candidates[0]
        best_classes = [[p, 0] for p in P]

    s = Fraction(0, 1)
    for p, r in best_classes:
        s += Fraction(1, p)
    if s > 1:
        return [[p, 0] for p in [13, 17, 19]]

    return best_classes

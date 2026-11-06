import numpy as np
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

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
    MAX_L = 100000

    def run_sa(seed, time_budget):
        random.seed(seed)
        coverage = np.zeros(MAX_L + 1, dtype=np.int32)
        residues = [random.randint(0, p - 1) for p in primes]

        for i, p in enumerate(primes):
            r = residues[i]
            coverage[r::p] += 1

        # Find initial L by scanning forward
        current_L = 0
        while current_L < MAX_L and coverage[current_L + 1] >= 2:
            current_L += 1

        best_L = current_L
        best_residues = residues.copy()

        T = 10.0
        cooling = 0.9995
        min_T = 0.1
        exp = math.exp
        rand = random.random
        randint = random.randint

        run_start = time.time()
        run_deadline = run_start + time_budget

        while time.time() < run_deadline:
            if current_L >= MAX_L:
                break

            x_bad = current_L + 1

            # Find primes that do NOT cover x_bad (residue != x_bad % p)
            candidates = []
            for i in range(n_primes):
                p = primes[i]
                if residues[i] != x_bad % p:
                    candidates.append(i)

            if not candidates:
                # All primes already cover x_bad but coverage < 2? Impossible if all 168 cover.
                # This shouldn't happen, but handle gracefully
                i = randint(0, n_primes - 1)
                p = primes[i]
                old_r = residues[i]
                new_r = randint(0, p - 1)
                if old_r == new_r:
                    continue
                coverage[old_r::p] -= 1
                coverage[new_r::p] += 1
                # Scan forward to update current_L
                while current_L < MAX_L and coverage[current_L + 1] >= 2:
                    current_L += 1
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                else:
                    # Revert if we didn't improve
                    coverage[new_r::p] -= 1
                    coverage[old_r::p] += 1
                T *= cooling
                if T < min_T:
                    T = min_T
                continue

            # Pick a random candidate
            i = candidates[randint(0, len(candidates) - 1)]
            p = primes[i]
            old_r = residues[i]
            new_r = x_bad % p

            # Apply move
            coverage[old_r::p] -= 1
            coverage[new_r::p] += 1

            # Scan forward from current_L to find new L (amortized)
            old_current_L = current_L
            while current_L < MAX_L and coverage[current_L + 1] >= 2:
                current_L += 1

            if current_L > old_current_L:
                # Improved
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
            else:
                # No improvement or regression
                delta = current_L - old_current_L
                if delta < 0:
                    prob = exp(delta / T)
                    if rand() < prob:
                        # Accept
                        pass
                    else:
                        # Revert
                        coverage[new_r::p] -= 1
                        coverage[old_r::p] += 1
                        current_L = old_current_L
                # delta == 0: accept (neutral move)

            T *= cooling
            if T < min_T:
                T = min_T

        return best_L, best_residues

    num_restarts = 5
    time_per_restart = 110.0 / num_restarts

    global_best_L = 0
    global_best_residues = None

    for k in range(num_restarts):
        remaining_time = deadline - time.time()
        if remaining_time < 1.0:
            break

        if k == num_restarts - 1:
            t_budget = remaining_time - 0.5
        else:
            t_budget = min(time_per_restart, remaining_time)

        if t_budget < 0.5:
            break

        L, residues = run_sa(42 + k, t_budget)

        if L > global_best_L:
            global_best_L = L
            global_best_residues = residues

        if global_best_L >= MAX_L:
            break

    if global_best_residues is None:
        return [random.randint(0, p - 1) for p in primes]

    return global_best_residues

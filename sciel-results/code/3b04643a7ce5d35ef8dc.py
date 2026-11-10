import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85

    def count_nonzero(poly):
        return int(np.count_nonzero(poly))

    def poly_square(coeffs):
        return np.convolve(coeffs, coeffs)

    def evaluate(coeffs):
        return count_nonzero(poly_square(coeffs))

    def validate(coeffs):
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        if not np.all((coeffs >= -64) & (coeffs <= 64)):
            return False
        if count_nonzero(coeffs) != 12:
            return False
        return True

    best_score = float('inf')
    best_poly = None

    random.seed(42)

    def try_poly(poly_list):
        nonlocal best_score, best_poly
        arr = np.array(poly_list, dtype=np.int64)
        if not validate(arr):
            return
        score = evaluate(arr)
        if score < best_score:
            best_score = score
            best_poly = arr.copy()

    # Phase 1: Algebraic structural seeds
    # Try P = sum of blocks where each block is a small sparse-square polynomial
    # Block: [1, a, 1] -> square = [1, 2a, (a^2+2), 2a, 1] -> 5 terms
    # Two blocks separated: P = [1,a,1,0,...,0,1,a,1] -> P^2 has overlap if close

    # Try P as sum of 3 blocks of 4 terms each, with geometric spacing
    for spacing in range(4, 10):
        for a in [-2, -1, 1, 2, 3]:
            c = [0] * 41
            c[0] = 1
            c[1] = a
            c[2] = 1
            c[spacing] = 1
            c[spacing+1] = a
            c[spacing+2] = 1
            c[2*spacing] = 1
            c[2*spacing+1] = a
            c[2*spacing+2] = 1
            try_poly(c)

    # Try P with alternating signs in blocks
    for spacing in range(4, 10):
        for a in [-2, -1, 1, 2]:
            c = [0] * 41
            c[0] = 1
            c[1] = a
            c[2] = 1
            c[spacing] = -1
            c[spacing+1] = -a
            c[spacing+2] = -1
            c[2*spacing] = 1
            c[2*spacing+1] = a
            c[2*spacing+2] = 1
            try_poly(c)

    # Try P = 1 + x^a + x^b + ... with 12 terms, symmetric
    for _ in range(300):
        exponents = sorted(random.sample(range(1, 41), 11))
        c = [0] * 41
        c[0] = random.choice([-1, 1])
        for e in exponents:
            c[e] = random.choice([-2, -1, 1, 2])
        if c[-1] == 0:
            c[exponents[-1]] = 1
        try_poly(c)

    # Phase 2: Differential evolution
    if best_poly is not None:
        pop_size = 20
        pop = [best_poly.copy() for _ in range(pop_size)]
        for i in range(1, pop_size):
            # Mutate best
            mutated = best_poly.copy()
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, len(mutated)-1)
                mutated[idx] = random.randint(-8, 8)
                if mutated[idx] == 0:
                    mutated[idx] = random.choice([-1, 1])
            # Ensure 12 nonzero
            while count_nonzero(mutated) < 12:
                idx = random.randint(0, len(mutated)-1)
                if mutated[idx] == 0:
                    mutated[idx] = random.choice([-1, 1])
            while count_nonzero(mutated) > 12:
                idx = random.randint(0, len(mutated)-1)
                if mutated[idx] != 0 and count_nonzero(mutated) > 12:
                    mutated[idx] = 0
            if mutated[-1] == 0:
                mutated[exponents[-1] if 'exponents' in dir() else len(mutated)-1] = 1
            pop[i] = mutated

        for gen in range(200):
            if time.time() > deadline - 5:
                break
            for i in range(pop_size):
                if time.time() > deadline - 5:
                    break
                # Create trial vector
                r1, r2 = random.sample(range(pop_size), 2)
                while r1 == i or r2 == i:
                    r1, r2 = random.sample(range(pop_size), 2)
                trial = pop[r1].copy()
                for j in range(len(trial)):
                    if random.random() < 0.3:
                        trial[j] = pop[r2][j] + random.uniform(-2, 2) * (pop[r1][j] - pop[r2][j])
                        trial[j] = int(round(trial[j]))
                        trial[j] = max(-64, min(64, trial[j]))
                # Ensure 12 nonzero
                nz = count_nonzero(trial)
                if nz < 12:
                    for j in range(len(trial)):
                        if trial[j] == 0:
                            trial[j] = random.choice([-1, 1])
                            if count_nonzero(trial) >= 12:
                                break
                elif nz > 12:
                    nz_indices = [j for j in range(len(trial)) if trial[j] != 0]
                    random.shuffle(nz_indices)
                    for j in nz_indices:
                        if count_nonzero(trial) <= 12:
                            break
                        trial[j] = 0
                if trial[-1] == 0:
                    trial[len(trial)-1] = 1

                if validate(trial):
                    score = evaluate(trial)
                    if score < best_score:
                        best_score = score
                        best_poly = trial.copy()
                        pop[0] = trial.copy()
                    if score < evaluate(pop[i]):
                        pop[i] = trial.copy()

    # Phase 3: Focused local search with multiple steps
    if best_poly is not None:
        curr = best_poly.copy()
        curr_score = best_score
        for step in range(5000):
            if time.time() > deadline - 3:
                break
            # Try changing 1-2 coefficients
            n_changes = random.choice([1, 1, 1, 2])
            idxs = random.sample(range(len(curr)), n_changes)
            old_vals = [curr[i] for i in idxs]
            for i in idxs:
                new_val = random.randint(-6, 6)
                if new_val == 0:
                    new_val = random.choice([-1, 1])
                curr[i] = new_val

            # Fix to exactly 12 nonzero
            nz = count_nonzero(curr)
            if nz < 12:
                for i in range(len(curr)):
                    if curr[i] == 0:
                        curr[i] = random.choice([-1, 1])
                        if count_nonzero(curr) >= 12:
                            break
            elif nz > 12:
                nz_idx = [i for i in range(len(curr)) if curr[i] != 0]
                random.shuffle(nz_idx)
                for i in nz_idx:
                    if count_nonzero(curr) <= 12:
                        break
                    curr[i] = 0
            if curr[-1] == 0:
                curr[-1] = 1

            if validate(curr):
                new_score = evaluate(curr)
                if new_score < curr_score:
                    curr_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_poly = curr.copy()
                else:
                    for i, v in zip(idxs, old_vals):
                        curr[i] = v

    if best_poly is None:
        best_poly = np.array([1]*12, dtype=np.int64)

    while len(best_poly) > 0 and best_poly[-1] == 0:
        best_poly = best_poly[:-1]

    return best_poly.tolist()

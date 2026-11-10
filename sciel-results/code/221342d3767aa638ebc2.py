import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    def count_nonzero(poly):
        return np.count_nonzero(poly)

    def poly_square(coeffs):
        return np.convolve(coeffs, coeffs)

    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)

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

    def try_and_update(poly_list):
        nonlocal best_score, best_poly
        c = np.array(poly_list, dtype=np.int64)
        if not validate(c):
            return
        score = evaluate(c)
        if score < best_score:
            best_score = score
            best_poly = c.copy()

    # Phase 1: Structured initial candidates
    candidates = []

    # Baseline
    candidates.append([1]*12)

    # Alternating signs
    candidates.append([1 if i % 2 == 0 else -1 for i in range(12)])

    # Patterns with potential for cancellation
    candidates.append([1]*11 + [-1])
    candidates.append([1]*10 + [-1, -1])
    candidates.append([1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1])
    candidates.append([1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1])
    candidates.append([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
    candidates.append([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1])
    candidates.append([1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2])
    candidates.append([1, -2, 1, -2, 1, -2, 1, -2, 1, -2, 1, -2])
    candidates.append([2, -1, 2, -1, 2, -1, 2, -1, 2, -1, 2, -1])

    # Random structured candidates
    random.seed(42)
    for _ in range(30):
        c = [random.randint(-6, 6) for _ in range(12)]
        for i in range(12):
            if c[i] == 0:
                c[i] = 1 if random.random() < 0.5 else -1
        candidates.append(c)

    for init_c in candidates:
        if time.time() > deadline:
            break
        try_and_update(init_c)

    # Phase 2: Intensive local search on best found
    if best_poly is not None and time.time() < deadline - 10:
        curr_c = best_poly.copy()
        curr_score = best_score

        # Simulated annealing with multi-coordinate moves
        no_improvement_count = 0
        max_no_improvement = 5000
        temperature = 3.0
        cooling_rate = 0.9998

        while time.time() < deadline - 5 and no_improvement_count < max_no_improvement:
            # Choose number of coordinates to modify
            if random.random() < 0.3:
                num_mods = 2
            elif random.random() < 0.1:
                num_mods = 3
            else:
                num_mods = 1

            indices = random.sample(range(12), num_mods)
            old_vals = [curr_c[i] for i in indices]

            new_vals = []
            for i in indices:
                val = random.randint(-10, 10)
                if val == 0:
                    val = 1 if random.random() < 0.5 else -1
                new_vals.append(val)

            for i, val in zip(indices, new_vals):
                curr_c[i] = val

            new_score = evaluate(curr_c)

            if new_score < curr_score:
                curr_score = new_score
                no_improvement_count = 0
                if curr_score < best_score:
                    best_score = curr_score
                    best_poly = curr_c.copy()
                    if best_score <= 12:
                        break
            else:
                # Simulated annealing acceptance
                delta = new_score - curr_score
                if random.random() < np.exp(-delta / max(temperature, 0.1)):
                    curr_score = new_score
                    no_improvement_count = 0
                    if curr_score < best_score:
                        best_score = curr_score
                        best_poly = curr_c.copy()
                else:
                    for i, val in zip(indices, old_vals):
                        curr_c[i] = val
                    no_improvement_count += 1

            temperature *= cooling_rate

    # Phase 3: Multiple restarts with different seeds
    if time.time() < deadline - 5:
        for seed in range(10):
            if time.time() > deadline:
                break
            random.seed(100 + seed)

            # Generate a new random starting point
            c = [random.randint(-8, 8) for _ in range(12)]
            for i in range(12):
                if c[i] == 0:
                    c[i] = 1 if random.random() < 0.5 else -1

            curr_c = np.array(c, dtype=np.int64)
            curr_score = evaluate(curr_c)

            if curr_score < best_score:
                best_score = curr_score
                best_poly = curr_c.copy()

            # Quick local search from this restart
            no_improvement_count = 0
            max_no_improvement = 2000
            temperature = 2.0
            cooling_rate = 0.9995

            while time.time() < deadline - 2 and no_improvement_count < max_no_improvement:
                num_mods = random.choice([1, 1, 1, 2, 2])
                indices = random.sample(range(12), num_mods)
                old_vals = [curr_c[i] for i in indices]

                new_vals = []
                for i in indices:
                    val = random.randint(-10, 10)
                    if val == 0:
                        val = 1 if random.random() < 0.5 else -1
                    new_vals.append(val)

                for i, val in zip(indices, new_vals):
                    curr_c[i] = val

                new_score = evaluate(curr_c)

                if new_score < curr_score:
                    curr_score = new_score
                    no_improvement_count = 0
                    if curr_score < best_score:
                        best_score = curr_score
                        best_poly = curr_c.copy()
                        if best_score <= 12:
                            break
                else:
                    delta = new_score - curr_score
                    if random.random() < np.exp(-delta / max(temperature, 0.1)):
                        curr_score = new_score
                        no_improvement_count = 0
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = curr_c.copy()
                    else:
                        for i, val in zip(indices, old_vals):
                            curr_c[i] = val
                        no_improvement_count += 1

                temperature *= cooling_rate

            if best_score <= 12:
                break

    # Final cleanup
    if best_poly is None:
        best_poly = np.array([1]*12, dtype=np.int64)

    while len(best_poly) > 0 and best_poly[-1] == 0:
        best_poly = best_poly[:-1]

    return best_poly.tolist()

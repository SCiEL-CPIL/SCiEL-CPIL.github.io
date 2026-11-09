import numpy as np
import time
import itertools
import random
import math

def construct(q, n, radius):
    """
    Constructs a covering code using:
    1. Fast Greedy Set Cover with vectorized coverage counting
    2. Simulated Annealing with Add, Remove, and Swap moves
    3. Multi-restart polish with aggressive pruning
    """
    start_time = time.time()
    timeout = 115

    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(word) for word in itertools.product(range(q), repeat=n)]

    total_words = 3 ** 7
    n_words = total_words

    # Precompute digits
    digits = np.zeros((n_words, n), dtype=int)
    for i in range(n_words):
        temp = i
        for k in range(n):
            digits[i, k] = temp % 3
            temp //= 3

    # Precompute balls as lists
    balls = []
    powers3 = [1, 3, 9, 27, 81, 243, 729]
    for i in range(n_words):
        ball = [i]
        for pos in range(n):
            orig_digit = digits[i, pos]
            p = powers3[pos]
            base_val = i - orig_digit * p
            for new_digit in range(3):
                if new_digit == orig_digit:
                    continue
                j = base_val + new_digit * p
                ball.append(j)
        balls.append(ball)

    # Precompute ball sets for faster membership
    ball_sets = [set(b) for b in balls]

    # Precompute ball membership matrix for fast coverage counting
    # ball_matrix[i, j] = 1 if word j is in ball i
    ball_matrix = np.zeros((n_words, n_words), dtype=np.uint8)
    for i in range(n_words):
        for j in balls[i]:
            ball_matrix[i, j] = 1

    # 1. Fast Greedy Set Cover using vectorized operations
    uncovered = np.ones(n_words, dtype=bool)
    code = []

    while np.any(uncovered):
        if time.time() - start_time > timeout - 30:
            break
        # Vectorized: count uncovered words covered by each ball
        cov = ball_matrix[:, uncovered].sum(axis=1)
        max_cov = cov.max()
        if max_cov == 0:
            break
        best_idx = int(cov.argmax())
        code.append(best_idx)
        uncovered[balls[best_idx]] = False

    # Initial state for SA
    count = np.zeros(n_words, dtype=int)
    for c in code:
        for j in balls[c]:
            count[j] += 1

    # State variables
    current_code = list(code)
    current_count = count.copy()
    current_uncovered = int(np.sum(current_count == 0))

    best_code = list(current_code)
    best_size = len(current_code)

    # SA Parameters - tuned for better exploration
    lambda_cost = 1.0
    mu_cost = 3.0
    T_start = 15.0
    T_end = 0.02
    cooling = 0.997

    rng = random.Random(42)
    T = T_start

    code_set = set(current_code)

    # Helper: prune redundant codewords
    def prune(code_list, count_arr, set_obj):
        new_code = []
        changed = False
        for c in code_list:
            redundant = True
            for j in balls[c]:
                if count_arr[j] < 2:
                    redundant = False
                    break
            if redundant:
                for j in balls[c]:
                    count_arr[j] -= 1
                set_obj.discard(c)
                changed = True
            else:
                new_code.append(c)
        return new_code, changed

    # Prune initial code
    current_code, _ = prune(current_code, current_count, code_set)
    current_uncovered = int(np.sum(current_count == 0))
    if current_uncovered == 0 and len(current_code) < best_size:
        best_size = len(current_code)
        best_code = list(current_code)

    # SA loop with Add, Remove, and Swap moves
    while time.time() - start_time < timeout - 20 and T > T_end:

        # Choose action: Remove (0.25), Add (0.25), Swap (0.5)
        r = rng.random()

        if r < 0.25:
            # Try Remove
            if not current_code:
                T *= cooling
                continue
            c_idx = rng.randint(0, len(current_code) - 1)
            c = current_code[c_idx]

            newly_uncovered = 0
            for j in balls[c]:
                if current_count[j] == 1:
                    newly_uncovered += 1

            delta_E = -lambda_cost + newly_uncovered * mu_cost

            if delta_E < 0 or (newly_uncovered > 0 and rng.random() < math.exp(-delta_E / T)):
                for j in balls[c]:
                    current_count[j] -= 1
                current_code.pop(c_idx)
                code_set.remove(c)
                current_uncovered += newly_uncovered

                if current_uncovered == 0:
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
        elif r < 0.5:
            # Try Add
            w = rng.randint(0, n_words - 1)
            if w in code_set:
                T *= cooling
                continue

            newly_covered = 0
            for j in balls[w]:
                if current_count[j] == 0:
                    newly_covered += 1

            delta_E = lambda_cost - newly_covered * mu_cost

            if delta_E < 0 or (newly_covered > 0 and rng.random() < math.exp(-delta_E / T)):
                for j in balls[w]:
                    current_count[j] += 1
                current_code.append(w)
                code_set.add(w)
                current_uncovered -= newly_covered

                if current_uncovered == 0:
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
        else:
            # Try Swap
            if not current_code:
                T *= cooling
                continue
            c_idx = rng.randint(0, len(current_code) - 1)
            c = current_code[c_idx]
            w = rng.randint(0, n_words - 1)

            if w == c or w in code_set:
                T *= cooling
                continue

            # Check validity: B(c) \ B(w) must be covered by others
            ball_w_set = ball_sets[w]
            valid = True
            for j in balls[c]:
                if j not in ball_w_set and current_count[j] < 2:
                    valid = False
                    break

            if valid:
                # Perform swap
                for j in balls[c]:
                    current_count[j] -= 1
                for j in balls[w]:
                    current_count[j] += 1
                current_code[c_idx] = w
                code_set.discard(c)
                code_set.add(w)

                if current_uncovered == 0:
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)

        T *= cooling

    # Final Pruning of best_code
    final_count = np.zeros(n_words, dtype=int)
    for c in best_code:
        for j in balls[c]:
            final_count[j] += 1

    pruned_code, _ = prune(best_code, final_count, set(best_code))
    best_code = pruned_code

    # Multi-restart intensive swap polish
    remaining_time = timeout - (time.time() - start_time)
    if remaining_time > 10:
        num_restarts = max(1, int(remaining_time / 15))
        for restart in range(num_restarts):
            if time.time() - start_time > timeout - 5:
                break

            polish_count = np.zeros(n_words, dtype=int)
            for c in best_code:
                for j in balls[c]:
                    polish_count[j] += 1

            polish_set = set(best_code)
            polish_rng = random.Random(123 + restart * 7919)
            polish_code = list(best_code)
            polish_size = len(polish_code)

            # Intensive swap attempts with periodic pruning
            stall_counter = 0
            max_stall = 500
            while time.time() - start_time < timeout - 5:
                if len(polish_code) == 0:
                    break

                c_idx = polish_rng.randint(0, len(polish_code) - 1)
                c = polish_code[c_idx]
                w = polish_rng.randint(0, n_words - 1)

                if w == c or w in polish_set:
                    stall_counter += 1
                    if stall_counter > max_stall:
                        break
                    continue

                # Check validity
                ball_w_set = ball_sets[w]
                valid = True
                for j in balls[c]:
                    if j not in ball_w_set and polish_count[j] < 2:
                        valid = False
                        break

                if valid:
                    # Perform swap
                    for j in balls[c]:
                        polish_count[j] -= 1
                    for j in balls[w]:
                        polish_count[j] += 1
                    polish_code[c_idx] = w
                    polish_set.discard(c)
                    polish_set.add(w)
                    stall_counter = 0

                    # Try to prune after swap
                    new_polish_code = []
                    changed = False
                    for cc in polish_code:
                        redundant = True
                        for j in balls[cc]:
                            if polish_count[j] < 2:
                                redundant = False
                                break
                        if redundant:
                            for j in balls[cc]:
                                polish_count[j] -= 1
                            polish_set.discard(cc)
                            changed = True
                        else:
                            new_polish_code.append(cc)
                    if changed:
                        polish_code = new_polish_code
                        if len(polish_code) < polish_size:
                            polish_size = len(polish_code)
                else:
                    stall_counter += 1
                    if stall_counter > max_stall:
                        break

            if polish_size < best_size:
                best_size = polish_size
                best_code = list(polish_code)

    result = [list(digits[c]) for c in best_code]
    return result

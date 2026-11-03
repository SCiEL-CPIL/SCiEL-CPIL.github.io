import time
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code using tetracode-seeded greedy, vectorized local search
    with inverse-index candidate search, and multi-phase optimization including
    simulated annealing on the complement representation.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill

    total_words = q ** n
    ball_size = 1 + n * (q - 1)

    def int_to_digits(num, length, base):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % base
            num //= base
        return digits

    def digits_to_int(digits, base):
        num = 0
        for d in reversed(digits):
            num = num * base + d
        return num

    # Precompute balls as a 2D array
    balls = np.zeros((total_words, ball_size), dtype=np.int32)
    for i in range(total_words):
        d = int_to_digits(i, n, q)
        balls[i, 0] = i
        idx = 1
        for pos in range(n):
            orig = d[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                d[pos] = new_val
                balls[i, idx] = digits_to_int(d, q)
                idx += 1
            d[pos] = orig

    # Precompute inverse index
    inverse = [[] for _ in range(total_words)]
    for c in range(total_words):
        for w in balls[c]:
            inverse[w].append(c)
    inverse_arrs = [np.array(inv, dtype=np.int32) for inv in inverse]

    def get_structured_seed():
        if q != 3 or n != 8:
            return list(range(total_words))
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        seed = []
        for u in tetracode:
            for v in tetracode:
                w = u + v
                seed.append(digits_to_int(w, q))
        return list(set(seed))

    def greedy_cover(seed_set, rng=None):
        if rng is None:
            rng = np.random.RandomState(0)
        uncovered = np.ones(total_words, dtype=bool)
        code = list(seed_set)
        cover_count = np.zeros(total_words, dtype=np.int32)

        for c in code:
            uncovered[balls[c]] = False
            cover_count[balls[c]] += 1

        if not np.any(uncovered):
            return code, cover_count

        while np.any(uncovered) and time.time() < deadline:
            gains = uncovered[balls].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break

            best_centers = np.where(gains == max_gain)[0]
            if len(best_centers) == 1:
                best_center = int(best_centers[0])
            else:
                best_center = int(best_centers[rng.randint(len(best_centers))])

            code.append(best_center)
            uncovered[balls[best_center]] = False
            cover_count[balls[best_center]] += 1

        return code, cover_count

    def trim_code(code, cover_count):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if np.any(cover_count[balls[c]] == 1):
                    continue
                to_remove.append(i)

            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                changed = True
        return code, cover_count

    def local_search(code, cover_count, max_rounds=50000):
        code_set = set(code)
        improved = True
        round_count = 0

        while improved and time.time() < deadline and round_count < max_rounds:
            improved = False
            round_count += 1
            rng = np.random.RandomState(round_count * 7919)
            code_order = list(range(len(code)))
            rng.shuffle(code_order)

            for ci in code_order:
                if time.time() > deadline:
                    break
                i = code[ci]
                i_ball = balls[i]
                only_mask = cover_count[i_ball] == 1
                only_by_i = i_ball[only_mask]

                if len(only_by_i) == 0:
                    code.pop(ci)
                    code_set.discard(i)
                    cover_count[i_ball] -= 1
                    improved = True
                    break

                if len(only_by_i) == 1:
                    candidates = inverse_arrs[only_by_i[0]]
                elif len(only_by_i) == 2:
                    c0 = set(inverse_arrs[only_by_i[0]])
                    candidates = np.array([c for c in inverse_arrs[only_by_i[1]] if c in c0], dtype=np.int32)
                elif len(only_by_i) <= 4:
                    c0 = set(inverse_arrs[only_by_i[0]])
                    for w in only_by_i[1:]:
                        c0 = c0.intersection(inverse_arrs[w])
                        if len(c0) == 0:
                            break
                    candidates = np.array(list(c0), dtype=np.int32) if c0 else np.array([], dtype=np.int32)
                else:
                    continue

                candidates = candidates[~np.isin(candidates, list(code_set))]

                if len(candidates) == 0:
                    continue

                if len(candidates) > 300:
                    sample = rng.choice(len(candidates), size=300, replace=False)
                    candidates = candidates[sample]

                best_j = None
                best_score = float('inf')
                i_ball_set = set(i_ball.tolist())

                for j in candidates:
                    j_ball = balls[j]
                    j_ball_set = set(j_ball.tolist())
                    newly_unc = len(i_ball_set - j_ball_set)
                    newly_cov = len(j_ball_set - i_ball_set)
                    score = newly_unc - newly_cov
                    if score < best_score:
                        best_score = score
                        best_j = j

                if best_j is not None and best_score <= 0:
                    code[ci] = best_j
                    code_set.discard(i)
                    code_set.add(best_j)
                    cover_count[i_ball] -= 1
                    cover_count[balls[best_j]] += 1
                    improved = True
                    break

        return code, cover_count

    def simulated_annealing_complement(code, cover_count, sa_deadline, max_iter=200000):
        """
        Simulated annealing on the complement representation.
        We track which words are 'excluded' (not in the code).
        Energy = number of uncovered words.
        Moves: add/remove codewords, swap codewords.
        """
        rng = np.random.RandomState(54321)
        current_code = list(code)
        current_cc = cover_count.copy()
        current_size = len(current_code)

        uncovered_mask = current_cc == 0
        uncovered_count = int(uncovered_mask.sum())

        best_code = current_code[:]
        best_size = current_size
        best_uncovered = uncovered_count
        best_cc = current_cc.copy()

        code_set = set(current_code)

        T_start = 50.0
        T_end = 0.1
        max_steps = max(100, max_iter)

        for iteration in range(max_steps):
            if time.time() > sa_deadline:
                break

            progress = iteration / max_steps
            T = T_start * (T_end / T_start) ** progress

            move_type = rng.random()

            if move_type < 0.4:
                # Single swap or removal
                if len(current_code) == 0:
                    break
                i = rng.randint(len(current_code))
                old_c = current_code[i]
                old_ball = balls[old_c]
                only_by_old = old_ball[current_cc[old_ball] == 1]

                if len(only_by_old) == 0:
                    current_code.pop(i)
                    code_set.discard(old_c)
                    current_cc[old_ball] -= 1
                    current_size -= 1
                    new_unc = int((current_cc == 0).sum())
                    uncovered_count = new_unc
                    if current_size + uncovered_count < best_size + best_uncovered:
                        best_code = current_code[:]
                        best_size = current_size
                        best_uncovered = uncovered_count
                        best_cc = current_cc.copy()
                    continue

                if len(only_by_old) <= 3:
                    if len(only_by_old) == 1:
                        candidates = inverse_arrs[only_by_old[0]]
                    else:
                        c0 = set(inverse_arrs[only_by_old[0]])
                        for w in only_by_old[1:]:
                            c0 = c0.intersection(inverse_arrs[w])
                            if not c0:
                                break
                        candidates = np.array(list(c0), dtype=np.int32) if c0 else np.array([], dtype=np.int32)

                    candidates = candidates[~np.isin(candidates, list(code_set))]
                    if len(candidates) > 0:
                        if len(candidates) > 150:
                            sample = rng.choice(len(candidates), size=150, replace=False)
                            candidates = candidates[sample]

                        old_ball_set = set(old_ball.tolist())
                        best_j = None
                        best_delta = float('inf')
                        for j in candidates:
                            j_ball_set = set(balls[j].tolist())
                            delta = len(old_ball_set - j_ball_set) - len(j_ball_set - old_ball_set)
                            if delta < best_delta:
                                best_delta = delta
                                best_j = j

                        if best_j is not None:
                            if best_delta <= 0 or rng.random() < np.exp(-best_delta / max(T, 1e-10)):
                                current_code[i] = best_j
                                code_set.discard(old_c)
                                code_set.add(best_j)
                                current_cc[old_ball] -= 1
                                current_cc[balls[best_j]] += 1
                                new_unc = int((current_cc == 0).sum())
                                uncovered_count = new_unc
                                if current_size + uncovered_count < best_size + best_uncovered:
                                    best_code = current_code[:]
                                    best_size = current_size
                                    best_uncovered = uncovered_count
                                    best_cc = current_cc.copy()

            elif move_type < 0.7:
                # Remove a random codeword (perturbation)
                if len(current_code) > 100:
                    i = rng.randint(len(current_code))
                    old_c = current_code[i]
                    old_ball = balls[old_c]
                    current_code.pop(i)
                    code_set.discard(old_c)
                    current_cc[old_ball] -= 1
                    current_size -= 1
                    new_unc = int((current_cc == 0).sum())
                    uncovered_count = new_unc

            else:
                # Add a codeword covering most uncovered words
                if uncovered_count > 0:
                    gains = (current_cc[balls] == 0).sum(axis=1)
                    max_gain = gains.max()
                    if max_gain > 0:
                        best_centers = np.where(gains == max_gain)[0]
                        new_c = int(best_centers[rng.randint(len(best_centers))])
                        if new_c not in code_set:
                            current_code.append(new_c)
                            code_set.add(new_c)
                            current_cc[balls[new_c]] += 1
                            current_size += 1
                            new_unc = int((current_cc == 0).sum())
                            uncovered_count = new_unc
                            if current_size + uncovered_count < best_size + best_uncovered:
                                best_code = current_code[:]
                                best_size = current_size
                                best_uncovered = uncovered_count
                                best_cc = current_cc.copy()

        return best_code, best_cc

    # Main algorithm
    seed_set = get_structured_seed()
    best_code = None
    best_count = float('inf')

    # Phase 1: Multi-start greedy + trim
    n_starts = 6
    for start_i in range(n_starts):
        if time.time() > deadline - 70:
            break

        rng = np.random.RandomState(start_i * 1000 + 42)
        seed_copy = seed_set[:]
        rng.shuffle(seed_copy)

        code, cover_count = greedy_cover(seed_copy, rng)
        code, cover_count = trim_code(code, cover_count)

        if len(code) < best_count:
            best_count = len(code)
            best_code = code[:]

    # Phase 2: Local search on best
    if best_code is not None:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in best_code:
            cover_count[balls[c]] += 1

        code, cover_count = local_search(best_code, cover_count, max_rounds=40000)
        if len(code) < best_count:
            best_count = len(code)
            best_code = code[:]

    # Phase 3: Simulated annealing on complement
    if best_code is not None and time.time() < deadline - 30:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in best_code:
            cover_count[balls[c]] += 1

        sa_deadline = min(deadline - 15, time.time() + 50)
        code, cover_count = simulated_annealing_complement(best_code, cover_count, sa_deadline, max_iter=300000)
        code, cover_count = trim_code(code, cover_count)

        if len(code) < best_count:
            best_count = len(code)
            best_code = code[:]

    # Phase 4: Final local search polish
    if best_code is not None and time.time() < deadline - 5:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in best_code:
            cover_count[balls[c]] += 1

        code, cover_count = local_search(best_code, cover_count, max_rounds=20000)
        if len(code) < best_count:
            best_count = len(code)
            best_code = code[:]

    if best_code is None:
        best_code = []

    result = []
    for c in best_code:
        result.append(int_to_digits(c, n, q))

    return result

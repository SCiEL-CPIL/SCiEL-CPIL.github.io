import time
import numpy as np
from itertools import product

def construct(q, n, radius):
    """
    Constructs a covering code using a structured initialization with
    aggressive local search and simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # for radius=1

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

    # Precompute balls
    balls_list = [None] * total_words
    for i in range(total_words):
        d = int_to_digits(i, n, q)
        neighbors = [i]
        for pos in range(n):
            orig = d[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                d[pos] = new_val
                neighbors.append(digits_to_int(d, q))
            d[pos] = orig
        balls_list[i] = neighbors

    balls = np.array(balls_list, dtype=np.int32)

    # Precompute ball sets for fast subset checks
    ball_sets = [frozenset(balls[i].tolist()) for i in range(total_words)]

    # Build adjacency: for each word, which centers cover it
    word_to_centers = [[] for _ in range(total_words)]
    for i in range(total_words):
        for w in balls[i]:
            word_to_centers[w].append(i)

    def greedy_cover(seed_offset=0):
        """Greedy set cover with randomization."""
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        cover_count = np.zeros(total_words, dtype=np.int32)

        # Randomize the order of centers for diversity
        center_order = np.arange(total_words)
        if seed_offset > 0:
            rng = np.random.RandomState(seed_offset)
            rng.shuffle(center_order)

        while np.any(uncovered):
            if time.time() > deadline:
                break

            gains = uncovered[balls].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break

            best_centers = np.where(gains == max_gain)[0]
            if len(best_centers) == 1:
                best_center = int(best_centers[0])
            else:
                best_center = int(best_centers[np.random.randint(len(best_centers))])

            code.append(best_center)
            uncovered[balls[best_center]] = False
            cover_count[balls[best_center]] += 1

        return code, cover_count

    def trim_code(code, cover_count):
        """Remove redundant codewords."""
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

    def randomized_local_search(code, cover_count, max_time=30.0):
        """
        Aggressive randomized local search with simulated annealing.
        Uses swap moves: replace one codeword with another.
        """
        code_set = set(code)
        candidates = list(set(range(total_words)) - code_set)
        cand_arr = np.array(candidates, dtype=np.int32)
        n_cand = len(cand_arr)

        T_start = 3.0
        T_end = 0.05
        max_iters = 300000
        no_improve_count = 0
        max_no_improve = 5000
        best_size = len(code)
        best_code = code[:]
        best_cc = cover_count.copy()

        start_ls = time.time()

        for it in range(max_iters):
            if time.time() - start_ls > max_time:
                break

            T = T_start * (T_end / T_start) ** (it / max(1, max_iters))

            # Try a swap: replace code[i] with cand[a]
            i = np.random.randint(len(code))
            a = np.random.randint(n_cand)
            j = int(cand_arr[a])
            if j in code_set:
                continue

            old_i = code[i]
            i_ball = balls[old_i]
            j_ball = balls[j]

            # Words uniquely covered by old_i
            only_mask = cover_count[i_ball] == 1
            only_by_i = i_ball[only_mask]

            if len(only_by_i) == 0:
                # Can remove old_i
                code.pop(i)
                cover_count[i_ball] -= 1
                code_set.discard(old_i)
                no_improve_count = 0
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                    best_cc = cover_count.copy()
                continue

            # Check if j covers all of only_by_i
            j_set = ball_sets[j]
            deficit = 0
            for w in only_by_i:
                if w not in j_set:
                    deficit += 1

            if deficit == 0:
                # Good swap
                code[i] = j
                code_set.discard(old_i)
                code_set.add(j)
                cover_count[i_ball] -= 1
                cover_count[j_ball] += 1
                no_improve_count = 0
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                    best_cc = cover_count.copy()
            elif deficit <= 2 and np.random.random() < np.exp(-deficit / T):
                # Accept with probability
                code[i] = j
                code_set.discard(old_i)
                code_set.add(j)
                cover_count[i_ball] -= 1
                cover_count[j_ball] += 1
                no_improve_count += 1
            else:
                no_improve_count += 1

            # Try removing a codeword if possible
            if np.random.random() < 0.1:
                i = np.random.randint(len(code))
                old_i = code[i]
                i_ball = balls[old_i]
                if np.any(cover_count[i_ball] == 1):
                    continue
                code.pop(i)
                cover_count[i_ball] -= 1
                code_set.discard(old_i)
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                    best_cc = cover_count.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            if no_improve_count > max_no_improve:
                # Restart from best
                code = best_code[:]
                cover_count = best_cc.copy()
                code_set = set(code)
                no_improve_count = 0

        # Final trim
        code, cover_count = trim_code(code, cover_count)
        return code, cover_count

    # Multi-start with different random seeds
    best_code = None
    best_count = float('inf')
    np.random.seed(42)

    n_starts = 4
    for start_i in range(n_starts):
        if time.time() > deadline - 40:
            break

        code, cover_count = greedy_cover(seed_offset=start_i * 7919)
        code, cover_count = trim_code(code, cover_count)
        code, cover_count = randomized_local_search(code, cover_count, max_time=20.0)

        if len(code) < best_count:
            best_count = len(code)
            best_code = code[:]

    if best_code is None:
        best_code = []

    result = []
    for c in best_code:
        result.append(int_to_digits(c, n, q))

    return result

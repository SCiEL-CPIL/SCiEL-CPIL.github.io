import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0

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
    ball_sets = [frozenset(balls[i].tolist()) for i in range(total_words)]

    # Reverse index: for each word, which centers cover it
    center_covers = [[] for _ in range(total_words)]
    for i in range(total_words):
        for w in balls[i]:
            center_covers[w].append(i)

    # Convert to numpy arrays for faster access
    cc_lengths = np.array([len(center_covers[w]) for w in range(total_words)], dtype=np.int32)
    cc_flat = np.array([c for w in range(total_words) for c in center_covers[w]], dtype=np.int32)
    cc_offsets = np.zeros(total_words + 1, dtype=np.int32)
    for w in range(total_words):
        cc_offsets[w + 1] = cc_offsets[w] + len(center_covers[w])

    def get_coverers(word):
        """Return numpy array of centers that cover the given word."""
        start = cc_offsets[word]
        end = cc_offsets[word + 1]
        return cc_flat[start:end]

    def greedy_cover(seed_offset=0):
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        cover_count = np.zeros(total_words, dtype=np.int32)

        rng = np.random.RandomState(seed_offset)

        while np.any(uncovered):
            if time.time() > deadline - 20:
                break

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
        while changed and time.time() < deadline - 10:
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

    def local_search(code, cover_count, max_iter=200000):
        """
        Targeted local search: for each codeword, find a replacement that covers
        its uniquely-covered words using the reverse index for fast candidate lookup.
        """
        code_set = set(code)
        code_arr = np.array(code, dtype=np.int32) if code else np.array([], dtype=np.int32)

        improved = True
        iter_count = 0
        while improved and time.time() < deadline - 8 and iter_count < max_iter:
            improved = False
            iter_count += 1

            # Process codewords in order of fewest unique words
            unique_counts = np.array([
                np.sum(cover_count[balls[c]] == 1) for c in code
            ], dtype=np.int32) if code else np.array([], dtype=np.int32)

            if len(unique_counts) == 0:
                break

            order = np.argsort(unique_counts)

            for ci in order:
                if time.time() > deadline - 8:
                    break
                if ci >= len(code):
                    continue

                i = code[ci]
                i_ball = balls[i]
                only_mask = cover_count[i_ball] == 1
                only_by_i = i_ball[only_mask]

                if len(only_by_i) == 0:
                    code.pop(ci)
                    cover_count[i_ball] -= 1
                    code_set.discard(i)
                    improved = True
                    break

                # Use reverse index to find candidates that cover all uniquely-covered words
                # Start with coverers of the first unique word, intersect with others
                candidate_set = set(get_coverers(int(only_by_i[0])))
                candidate_set.discard(i)

                for w in only_by_i[1:]:
                    candidate_set &= set(get_coverers(int(w)))

                # Filter out current codewords
                candidate_set -= code_set

                if not candidate_set:
                    continue

                # Score candidates: prefer those with high redundancy gain
                # (words that would go from covered-by-2 to covered-by-1)
                best_candidate = None
                best_score = -1
                cand_list = list(candidate_set)

                # If too many candidates, sample
                if len(cand_list) > 200:
                    rng = np.random.RandomState(iter_count + ci)
                    cand_list = [cand_list[rng.randint(len(cand_list))] for _ in range(200)]

                for j in cand_list:
                    j_ball = balls[j]
                    # Score: count words in j_ball that are currently covered exactly twice
                    # (removing i and adding j would keep them covered, but reduce redundancy)
                    # Better: count words in j_ball NOT in i_ball that are covered >= 2
                    # (these gain redundancy when j replaces i)
                    score = 0
                    for w in j_ball:
                        if w not in ball_sets[i] and cover_count[w] >= 2:
                            score += 1
                    # Penalize words in i_ball not in j_ball that are covered exactly 2
                    for w in i_ball:
                        if w not in ball_sets[j] and cover_count[w] == 2:
                            score -= 1

                    if score > best_score:
                        best_score = score
                        best_candidate = j

                if best_candidate is not None:
                    j = best_candidate
                    code[ci] = j
                    code_set.discard(i)
                    code_set.add(j)
                    cover_count[i_ball] -= 1
                    cover_count[balls[j]] += 1
                    improved = True

        code, cover_count = trim_code(code, cover_count)
        return code, cover_count

    def polish(code, cover_count, max_iters=150000):
        """
        Simulated annealing polish with targeted moves.
        """
        code_set = set(code)
        cc = cover_count.copy()
        rng = np.random.RandomState(12345)

        T_start = 2.0
        T_end = 0.1
        n_iters = 0

        while n_iters < max_iters and time.time() < deadline - 5:
            n_iters += 1
            T = T_start * (T_end / T_start) ** (n_iters / max(1, max_iters))

            if not code:
                break

            move_type = rng.random()

            if move_type < 0.6:
                # Single swap: pick a codeword, try to replace with one covering its unique words
                i = rng.randint(len(code))
                old_i = code[i]
                i_ball = balls[old_i]
                only_mask = cc[i_ball] == 1
                only_by_i = i_ball[only_mask]

                if len(only_by_i) == 0:
                    code.pop(i)
                    cc[i_ball] -= 1
                    code_set.discard(old_i)
                    continue

                # Find candidates via reverse index
                candidate_set = set(get_coverers(int(only_by_i[0])))
                candidate_set.discard(old_i)
                for w in only_by_i[1:]:
                    candidate_set &= set(get_coverers(int(w)))
                candidate_set -= code_set

                if not candidate_set:
                    continue

                cand_list = list(candidate_set)
                j = cand_list[rng.randint(len(cand_list))]
                j_ball = balls[j]

                # Compute deficit (should be 0 by construction) and gain
                gain = 0
                for w in j_ball:
                    if w not in ball_sets[old_i] and cc[w] >= 2:
                        gain += 1
                for w in i_ball:
                    if w not in ball_sets[j] and cc[w] == 2:
                        gain -= 1

                # Accept if gain >= 0 or with probability exp(-deficit/T)
                if gain >= 0 or rng.random() < np.exp(-max(0, -gain) / T):
                    code[i] = j
                    code_set.discard(old_i)
                    code_set.add(j)
                    cc[i_ball] -= 1
                    cc[j_ball] += 1

            elif move_type < 0.8:
                # Try removal
                i = rng.randint(len(code))
                old_i = code[i]
                i_ball = balls[old_i]
                if not np.any(cc[i_ball] == 1):
                    code.pop(i)
                    cc[i_ball] -= 1
                    code_set.discard(old_i)

            else:
                # Pair swap
                if len(code) < 2:
                    continue
                i = rng.randint(len(code))
                j_idx = rng.randint(len(code))
                if j_idx == i:
                    continue
                old_i = code[i]
                old_j = code[j_idx]
                i_ball = balls[old_i]
                j_ball = balls[old_j]

                only_i = i_ball[cc[i_ball] == 1]
                only_j = j_ball[cc[j_ball] == 1]
                required = set(only_i.tolist()) | set(only_j.tolist())

                if not required:
                    # Remove both
                    if j_idx > i:
                        code.pop(j_idx)
                        cc[balls[old_j]] -= 1
                        code_set.discard(old_j)
                        code.pop(i)
                        cc[balls[old_i]] -= 1
                        code_set.discard(old_i)
                    else:
                        code.pop(i)
                        cc[balls[old_i]] -= 1
                        code_set.discard(old_i)
                        code.pop(j_idx)
                        cc[balls[old_j]] -= 1
                        code_set.discard(old_j)
                    continue

                # Find candidates covering required words
                first_req = list(required)[0]
                cand_set = set(get_coverers(first_req))
                cand_set.discard(old_i)
                cand_set.discard(old_j)
                for w in required:
                    if w != first_req:
                        cand_set &= set(get_coverers(w))
                cand_set -= code_set
                cand_set.discard(old_i)
                cand_set.discard(old_j)

                if len(cand_set) < 2:
                    continue

                cand_list = list(cand_set)
                a = cand_list[rng.randint(len(cand_list))]
                b = cand_list[rng.randint(len(cand_list))]
                if a == b:
                    continue

                a_ball = balls[a]
                b_ball = balls[b]

                # Check coverage
                deficit = 0
                for w in required:
                    if w not in ball_sets[a] and w not in ball_sets[b]:
                        deficit += 1

                if deficit == 0:
                    code[i] = a
                    code[j_idx] = b
                    code_set.discard(old_i)
                    code_set.discard(old_j)
                    code_set.add(a)
                    code_set.add(b)
                    cc[i_ball] -= 1
                    cc[j_ball] -= 1
                    cc[a_ball] += 1
                    cc[b_ball] += 1

        code, cc = trim_code(code, cc)
        return code, cc

    # Multi-start
    best_code = None
    best_count = float('inf')

    n_starts = 6
    for start_i in range(n_starts):
        if time.time() > deadline - 30:
            break

        code, cover_count = greedy_cover(seed_offset=start_i * 7919 + 13)
        code, cover_count = trim_code(code, cover_count)
        code, cover_count = local_search(code, cover_count, max_iter=100000)
        code, cover_count = polish(code, cover_count, max_iters=80000)

        if len(code) < best_count:
            best_count = len(code)
            best_code = code[:]

    # Final polish on best
    if best_code is not None:
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in best_code:
            cover_count[balls[c]] += 1
        best_code, cover_count = local_search(best_code, cover_count, max_iter=200000)
        best_code, cover_count = polish(best_code, cover_count, max_iters=100000)

    if best_code is None:
        best_code = []

    result = []
    for c in best_code:
        result.append(int_to_digits(c, n, q))

    return result

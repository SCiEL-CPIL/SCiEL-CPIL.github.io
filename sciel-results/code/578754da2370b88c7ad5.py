"""
Covering code K_3(8,1): structured tetracode init + vectorized greedy +
redundancy trimming + inverse-index swap local search.
"""
import time
import numpy as np
import random
import itertools

_TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
_TETRA_SET = frozenset(tuple(t) for t in _TETRA)
_TAIL_IDS = list(range(3 ** 4))


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin under the 150 s wall-clock kill

    total = q ** n
    ball_size = 1 + n * (q - 1)  # radius = 1

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

    # ---------- Trivial exact cases ----------
    if total <= ball_size:
        return [[0] * n]
    if q == 2 and n == 1:
        return [[0], [1]]
    if q == 3 and n == 1:
        return [[0], [1], [2]]

    if q != 3 or n != 8 or radius != 1:
        # Generic fallback for other parameters
        if total <= 200000:
            words = [list(w) for w in itertools.product(range(q), repeat=n)]
            chosen = []
            covered = set()
            for w in words:
                if tuple(w) in covered:
                    continue
                chosen.append(w)
                for i in range(n):
                    for v in range(q):
                        if v == w[i]:
                            covered.add(tuple(w))
                        else:
                            nw = w[:i] + [v] + w[i + 1:]
                            covered.add(tuple(nw))
                if len(covered) >= total:
                    break
            return chosen
        # Direct-sum seed fallback
        base = n - 4
        if base >= 0:
            return [h + list(t) for h in _TETRA
                    for t in itertools.product(range(q), repeat=base)]
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # ---- Specialized K_3(8,1) ----

    # ---------- Precompute balls ----------
    balls = np.empty((total, ball_size), dtype=np.int32)
    for w in range(total):
        digits = int_to_digits(w, n, q)
        ball = [w]
        for pos in range(n):
            orig = digits[pos]
            for nv in range(q):
                if nv == orig:
                    continue
                digits[pos] = nv
                ball.append(digits_to_int(digits, q))
            digits[pos] = orig
        balls[w] = ball

    # ---------- Inverse index: which centers cover each word ----------
    centers_covering = [None] * total
    for i in range(total):
        for w in balls[i]:
            if centers_covering[w] is None:
                centers_covering[w] = []
            centers_covering[w].append(i)
    for w in range(total):
        if centers_covering[w] is not None:
            centers_covering[w] = frozenset(centers_covering[w])

    # ---------- Structured tetracode-product initialization ----------
    def structured_init():
        tetracode = [
            [0, 0, 0, 0],
            [0, 1, 2, 1],
            [0, 2, 1, 2],
            [1, 0, 1, 2],
            [1, 1, 2, 0],
            [1, 2, 0, 2],
        ]
        product_code = []
        for a in tetracode:
            for b in tetracode:
                product_code.append(a + b)
        code_ints = list({digits_to_int(c, q) for c in product_code})
        cover_count = np.zeros(total, dtype=np.int32)
        for c in code_ints:
            cover_count[balls[c]] += 1
        # Greedy extension
        uncovered = (cover_count == 0)
        while np.any(uncovered) and time.time() < deadline - 5:
            gains = uncovered[balls].sum(axis=1)
            for c in code_ints:
                gains[c] = -1
            max_gain = int(gains.max())
            if max_gain <= 0:
                break
            best_center = int(np.argmax(gains))
            code_ints.append(best_center)
            cover_count[balls[best_center]] += 1
            uncovered = (cover_count == 0)
        return code_ints, cover_count

    # ---------- Pure greedy set-cover fallback ----------
    def greedy_cover():
        covered = np.zeros(total, dtype=np.bool_)
        covered_count = 0
        code = []
        cover_count = np.zeros(total, dtype=np.int32)
        while covered_count < total and time.time() < deadline - 5:
            gains = ball_size - covered[balls].sum(axis=1)
            best_word = int(np.argmax(gains))
            if gains[best_word] == 0:
                break
            code.append(best_word)
            bp = balls[best_word]
            newly = ~covered[bp]
            if newly.any():
                covered[bp] = True
                covered_count += int(newly.sum())
            cover_count[bp] += 1
        return code, cover_count

    # ---------- Choose the better starting point ----------
    code, cover_count = structured_init()
    if not code:
        code, cover_count = greedy_cover()
    code_set = set(code)

    # ---------- Aggressive trimming of redundant codewords ----------
    def trim():
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if (cover_count[balls[c]] > 1).all():
                    to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                    code_set.discard(c)
                changed = True

    trim()

    # ---------- Local search: targeted swap via inverse index ----------
    def local_search(time_budget=100.0):
        nonlocal code, cover_count
        end_time = time.time() + time_budget
        improved = True
        rng = random.Random(12345)
        while improved and time.time() < end_time:
            improved = False
            indices = list(range(len(code)))
            rng.shuffle(indices)
            for idx in indices:
                if time.time() > end_time:
                    break
                i = code[idx]
                bp_i = balls[i]
                unique_pts = bp_i[cover_count[bp_i] == 1]
                if len(unique_pts) == 0:
                    cover_count[bp_i] -= 1
                    code.pop(idx)
                    code_set.discard(i)
                    improved = True
                    break
                # Intersection of centers covering every uniquely-covered word
                candidate_set = None
                for w in unique_pts:
                    cc = centers_covering[w]
                    if candidate_set is None:
                        candidate_set = set(cc)
                    else:
                        candidate_set &= cc
                    if not candidate_set:
                        break
                if not candidate_set:
                    continue
                candidates = [c for c in candidate_set if c not in code_set]
                if not candidates:
                    continue
                # Score: minimize wasted coverage after the swap
                best_swap = None
                best_waste = None
                for j in candidates:
                    if time.time() > end_time:
                        return
                    bp_j = balls[j]
                    in_i = (cover_count[bp_i] > 0)
                    waste = int(np.sum((cover_count[bp_j] >= 2) & in_i))
                    if best_waste is None or waste < best_waste:
                        best_waste = waste
                        best_swap = j
                if best_swap is not None:
                    j = best_swap
                    cover_count[balls[i]] -= 1
                    cover_count[balls[j]] += 1
                    code[idx] = j
                    code_set.discard(i)
                    code_set.add(j)
                    improved = True
                    break

    local_search(time_budget=80.0)
    trim()

    # ---------- Simulated Annealing on codeword set ----------
    def simulated_annealing(time_budget=30.0):
        nonlocal code, cover_count
        end_time = time.time() + time_budget
        if time.time() >= end_time:
            return

        rng = random.Random(54321)
        T = 5.0  # initial temperature
        T_min = 0.1
        cooling = 0.995

        current_size = len(code)
        best_size = current_size
        best_code_snapshot = list(code)

        # Precompute: for each word, how many centers cover it (already in cover_count)
        # Energy = number of codewords (to minimize)
        # We also track uncovered count as a hard constraint

        step = 0
        while time.time() < end_time and T > T_min:
            step += 1
            if step % 1000 == 0 and time.time() > end_time:
                break

            # Choose a random move type:
            # 1. Remove a random codeword (if it doesn't create uncovered words)
            # 2. Add a random codeword
            # 3. Swap: remove one, add another
            move_type = rng.random()

            if move_type < 0.3 and len(code) > 1:
                # Try to remove a random codeword
                idx = rng.randrange(len(code))
                c = code[idx]
                bp = balls[c]
                # Check if removing c would leave any word uncovered
                would_uncover = (cover_count[bp] == 1).any()
                if not would_uncover:
                    # Safe removal
                    cover_count[bp] -= 1
                    code.pop(idx)
                    code_set.discard(c)
                    new_size = len(code)
                    if new_size < best_size:
                        best_size = new_size
                        best_code_snapshot = list(code)
                else:
                    # Would create uncovered words - skip
                    pass

            elif move_type < 0.6:
                # Try to add a random codeword (only useful if it helps with future removals)
                # Actually, adding increases size, so only do this if we think it enables
                # a better configuration. For now, skip pure additions in SA.
                pass

            else:
                # Swap: remove one codeword, add another
                if len(code) == 0:
                    continue
                idx = rng.randrange(len(code))
                old_c = code[idx]
                bp_old = balls[old_c]
                unique_old = bp_old[cover_count[bp_old] == 1]

                # Find candidates that cover all uniquely-covered words of old_c
                if len(unique_old) > 0:
                    candidate_set = None
                    for w in unique_old:
                        cc = centers_covering[w]
                        if candidate_set is None:
                            candidate_set = set(cc)
                        else:
                            candidate_set &= cc
                        if not candidate_set:
                            break
                    if candidate_set:
                        candidates = [c for c in candidate_set if c not in code_set]
                        if candidates:
                            new_c = rng.choice(candidates)
                            # Perform the swap
                            bp_new = balls[new_c]
                            cover_count[bp_old] -= 1
                            cover_count[bp_new] += 1
                            code[idx] = new_c
                            code_set.discard(old_c)
                            code_set.add(new_c)
                            new_size = len(code)
                            if new_size < best_size:
                                best_size = new_size
                                best_code_snapshot = list(code)
                else:
                    # old_c is redundant - remove it
                    cover_count[bp_old] -= 1
                    code.pop(idx)
                    code_set.discard(old_c)
                    new_size = len(code)
                    if new_size < best_size:
                        best_size = new_size
                        best_code_snapshot = list(code)

            # Occasionally try a more aggressive move: remove a codeword that has
            # some unique points, then try to find a replacement
            if move_type < 0.15 and len(code) > 1:
                idx = rng.randrange(len(code))
                old_c = code[idx]
                bp_old = balls[old_c]
                unique_old = bp_old[cover_count[bp_old] == 1]

                if len(unique_old) > 0 and len(unique_old) <= 3:
                    # Try to find a single codeword that covers all unique points
                    candidate_set = None
                    for w in unique_old:
                        cc = centers_covering[w]
                        if candidate_set is None:
                            candidate_set = set(cc)
                        else:
                            candidate_set &= cc
                        if not candidate_set:
                            break
                    if candidate_set:
                        candidates = [c for c in candidate_set if c not in code_set]
                        if candidates:
                            new_c = rng.choice(candidates)
                            bp_new = balls[new_c]
                            # Check energy change
                            old_energy = len(code)
                            # After swap, size stays same, but coverage might improve
                            cover_count[bp_old] -= 1
                            cover_count[bp_new] += 1
                            code[idx] = new_c
                            code_set.discard(old_c)
                            code_set.add(new_c)
                            new_size = len(code)
                            if new_size < best_size:
                                best_size = new_size
                                best_code_snapshot = list(code)

            # Cooling
            T *= cooling

        # Restore best solution found
        if best_size < len(code):
            code = best_code_snapshot
            # Rebuild cover_count
            cover_count = np.zeros(total, dtype=np.int32)
            code_set.clear()
            for c in code:
                cover_count[balls[c]] += 1
                code_set.add(c)

    simulated_annealing(time_budget=25.0)
    trim()

    # ---------- Convert back to list of lists of integers ----------
    result = [int_to_digits(c, n, q) for c in code]
    return result

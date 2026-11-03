"""
Covering code K_3(8,1): structured tetracode init + vectorized greedy +
redundancy trimming + inverse-index swap local search + multi-start SA.
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

    local_search(time_budget=60.0)
    trim()

    # ---------- Multi-start Simulated Annealing on codeword set ----------
    def multi_start_sa(time_budget=45.0):
        nonlocal code, cover_count
        end_time = time.time() + time_budget
        if time.time() >= end_time:
            return

        best_size = len(code)
        best_code_snapshot = list(code)

        # Try multiple random initializations with symmetry breaking
        n_starts = 3
        start_time_sa = time.time()
        sa_deadline = start_time_sa + time_budget

        for start_idx in range(n_starts):
            if time.time() > sa_deadline - 5:
                break

            # For each start, perturb the current solution
            if start_idx == 0:
                # Use current solution
                cur_code = list(code)
                cur_cover = cover_count.copy()
                cur_set = set(code)
            else:
                # Create perturbed version
                cur_code = list(code)
                cur_cover = cover_count.copy()
                cur_set = set(code)

                # Remove some random codewords and re-add
                n_remove = max(1, len(cur_code) // 10)
                rng = random.Random(1000 + start_idx)
                to_remove = rng.sample(range(len(cur_code)), min(n_remove, len(cur_code)))
                for idx in sorted(to_remove, reverse=True):
                    c = cur_code[idx]
                    cur_cover[balls[c]] -= 1
                    cur_code.pop(idx)
                    cur_set.discard(c)

                # Re-add random codewords to cover gaps
                uncovered_mask = (cur_cover == 0)
                if uncovered_mask.any():
                    n_add = int(uncovered_mask.sum()) // ball_size + 1
                    for _ in range(min(n_add, 50)):
                        # Pick a random word that covers many uncovered
                        gains = uncovered_mask[balls].sum(axis=1)
                        for c in cur_code:
                            gains[c] = -1
                        if gains.max() > 0:
                            best_c = int(np.argmax(gains))
                            cur_code.append(best_c)
                            cur_cover[balls[best_c]] += 1
                            cur_set.add(best_c)
                            uncovered_mask = (cur_cover == 0)

            # Run SA from this state
            rng = random.Random(54321 + start_idx * 1000)
            T = 3.0
            T_min = 0.1
            cooling = 0.997

            cur_size = len(cur_code)
            best_size_sa = cur_size
            best_code_sa = list(cur_code)

            step = 0
            while time.time() < sa_deadline and T > T_min:
                step += 1
                if step % 500 == 0 and time.time() > sa_deadline:
                    break

                move_type = rng.random()

                if move_type < 0.35 and len(cur_code) > 1:
                    # Try to remove a random codeword
                    idx = rng.randrange(len(cur_code))
                    c = cur_code[idx]
                    bp = balls[c]
                    would_uncover = (cur_cover[bp] == 1).any()
                    if not would_uncover:
                        cur_cover[bp] -= 1
                        cur_code.pop(idx)
                        cur_set.discard(c)
                        new_size = len(cur_code)
                        if new_size < best_size_sa:
                            best_size_sa = new_size
                            best_code_sa = list(cur_code)
                elif move_type < 0.65:
                    # Swap: remove one codeword, add another
                    if len(cur_code) == 0:
                        continue
                    idx = rng.randrange(len(cur_code))
                    old_c = cur_code[idx]
                    bp_old = balls[old_c]
                    unique_old = bp_old[cur_cover[bp_old] == 1]

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
                            candidates = [c for c in candidate_set if c not in cur_set]
                            if candidates:
                                new_c = rng.choice(candidates)
                                bp_new = balls[new_c]
                                cur_cover[bp_old] -= 1
                                cur_cover[bp_new] += 1
                                cur_code[idx] = new_c
                                cur_set.discard(old_c)
                                cur_set.add(new_c)
                                new_size = len(cur_code)
                                if new_size < best_size_sa:
                                    best_size_sa = new_size
                                    best_code_sa = list(cur_code)
                    else:
                        # old_c is redundant - remove it
                        cur_cover[bp_old] -= 1
                        cur_code.pop(idx)
                        cur_set.discard(old_c)
                        new_size = len(cur_code)
                        if new_size < best_size_sa:
                            best_size_sa = new_size
                            best_code_sa = list(cur_code)
                else:
                    # Aggressive move: try to find a better swap
                    if len(cur_code) > 1:
                        idx = rng.randrange(len(cur_code))
                        old_c = cur_code[idx]
                        bp_old = balls[old_c]
                        unique_old = bp_old[cur_cover[bp_old] == 1]

                        if 0 < len(unique_old) <= 4:
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
                                candidates = [c for c in candidate_set if c not in cur_set]
                                if candidates:
                                    new_c = rng.choice(candidates)
                                    bp_new = balls[new_c]
                                    cur_cover[bp_old] -= 1
                                    cur_cover[bp_new] += 1
                                    cur_code[idx] = new_c
                                    cur_set.discard(old_c)
                                    cur_set.add(new_c)
                                    new_size = len(cur_code)
                                    if new_size < best_size_sa:
                                        best_size_sa = new_size
                                        best_code_sa = list(cur_code)

                T *= cooling

            # Update global best if this start found something better
            if best_size_sa < best_size:
                best_size = best_size_sa
                best_code_snapshot = best_code_sa

        # Restore best solution found
        if best_size < len(code):
            code = best_code_snapshot
            cover_count = np.zeros(total, dtype=np.int32)
            code_set.clear()
            for c in code:
                cover_count[balls[c]] += 1
                code_set.add(c)

    multi_start_sa(time_budget=40.0)
    trim()

    # ---------- Final polish: targeted local search with tighter criteria ----------
    def final_polish(time_budget=10.0):
        nonlocal code, cover_count
        end_time = time.time() + time_budget
        improved = True
        rng = random.Random(99999)
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

    final_polish(time_budget=10.0)
    trim()

    # ---------- Convert back to list of lists of integers ----------
    result = [int_to_digits(c, n, q) for c in code]
    return result

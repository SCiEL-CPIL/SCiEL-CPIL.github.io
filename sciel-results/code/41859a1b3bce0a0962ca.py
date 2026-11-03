import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before the 120s kill

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
    balls_list = [list(row) for row in balls]
    balls_arr = balls

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

    # ---------- Tetracode structured initializer ----------
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
            cover_count[balls_arr[c]] += 1
        uncovered = (cover_count == 0)
        code_set = set(code_ints)
        while np.any(uncovered) and time.time() < deadline - 5:
            gains = uncovered[balls_arr].sum(axis=1)
            for c in code_ints:
                gains[c] = -1
            max_gain = int(gains.max())
            if max_gain <= 0:
                break
            best_center = int(np.argmax(gains))
            code_ints.append(best_center)
            code_set.add(best_center)
            cover_count[balls_arr[best_center]] += 1
            uncovered = (cover_count == 0)
        return code_ints, cover_count

    # ---------- Pure greedy set-cover fallback ----------
    def greedy_cover():
        covered = np.zeros(total, dtype=np.bool_)
        covered_count = 0
        code = []
        cover_count = np.zeros(total, dtype=np.int32)
        code_set = set()
        while covered_count < total and time.time() < deadline - 5:
            gains = ball_size - covered[balls_arr].sum(axis=1)
            best_word = int(np.argmax(gains))
            if gains[best_word] == 0:
                break
            code.append(best_word)
            code_set.add(best_word)
            bp = balls_arr[best_word]
            newly = ~covered[bp]
            if newly.any():
                covered[bp] = True
                covered_count += int(newly.sum())
            cover_count[bp] += 1
        return code, cover_count

    # ---------- LP-dual fractional lower bound + randomized rounding initializer ----------
    def lp_dual_rounding_init():
        end_time = time.time() + 25.0
        y = np.full(total, 1.0 / ball_size, dtype=np.float64)
        steps = 400
        eta = 0.05
        for it in range(steps):
            if time.time() > end_time:
                break
            ball_sums = y[balls_arr].sum(axis=1)
            lam = np.zeros(total, dtype=np.float64)
            for w in range(total):
                cc = centers_covering[w]
                if cc is None:
                    continue
                vals = ball_sums[list(cc)]
                m = float(vals.max())
                if m > 1.0:
                    lam[w] = m - 1.0
            y += eta * (1.0 - lam)
            np.clip(y, 0.0, 2.0 / ball_size, out=y)
        cover_count = np.zeros(total, dtype=np.int32)
        code = []
        code_set = set()
        ball_y_sum = y[balls_arr].sum(axis=1)
        order = np.argsort(-ball_y_sum)
        for c in order:
            if time.time() > end_time:
                break
            if len(code) > 600:
                break
            code.append(int(c))
            code_set.add(int(c))
            cover_count[balls_arr[c]] += 1
        uncovered = (cover_count == 0)
        while np.any(uncovered) and time.time() < deadline - 5:
            gains = uncovered[balls_arr].sum(axis=1)
            for c in code:
                gains[c] = -1
            max_gain = int(gains.max())
            if max_gain <= 0:
                break
            best_center = int(np.argmax(gains))
            code.append(best_center)
            code_set.add(best_center)
            cover_count[balls_arr[best_center]] += 1
            uncovered = (cover_count == 0)
        return code, cover_count

    # ---------- Trimming of redundant codewords ----------
    def trim_inplace(code, cover_count, code_set):
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

    # ---------- Single-swap local search (waste-minimizing) ----------
    def local_search(code, cover_count, code_set, end_time):
        improved = True
        rng = random.Random(98765)
        while improved and time.time() < end_time:
            improved = False
            indices = list(range(len(code)))
            rng.shuffle(indices)
            for idx in indices:
                if time.time() > end_time:
                    break
                i = code[idx]
                bp_i = balls_arr[i]
                unique_pts = bp_i[cover_count[bp_i] == 1]
                if len(unique_pts) == 0:
                    cover_count[bp_i] -= 1
                    code.pop(idx)
                    code_set.discard(i)
                    improved = True
                    break
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
                best_swap = None
                best_waste = None
                best_new = -1
                for j in candidates:
                    if time.time() > end_time:
                        return
                    bp_j = balls_arr[j]
                    in_i = np.isin(bp_j, bp_i)
                    eff = cover_count[bp_j] - in_i.astype(np.int32)
                    waste = int(np.sum(eff >= 2))
                    new_cov = int(np.sum(eff == 0))
                    if best_waste is None or waste < best_waste or (waste == best_waste and new_cov > best_new):
                        best_waste = waste
                        best_new = new_cov
                        best_swap = j
                if best_swap is not None:
                    j = best_swap
                    cover_count[balls_arr[i]] -= 1
                    cover_count[balls_arr[j]] += 1
                    code[idx] = j
                    code_set.discard(i)
                    code_set.add(j)
                    improved = True
                    break

    # ---------- Fast 2-opt merge: remove two, add one covering both unique sets (net -1) ----------
    def two_opt_merge(code, cover_count, code_set, end_time):
        improved = True
        rng = random.Random(12345)
        while improved and time.time() < end_time:
            improved = False
            L = len(code)
            if L < 2:
                break
            # Sample a limited number of pairs to avoid O(L^2) explosion
            num_samples = min(200, L * L // 10)
            samples = rng.sample(range(L * L), num_samples)
            for s in samples:
                if time.time() > end_time:
                    return
                a_idx = s % L
                b_idx = s // L
                if a_idx == b_idx:
                    continue
                i1 = code[a_idx]
                i2 = code[b_idx]
                bp1 = balls_arr[i1]
                bp2 = balls_arr[i2]
                U1 = set(bp1[cover_count[bp1] == 1].tolist())
                U2 = set(bp2[cover_count[bp2] == 1].tolist())
                if not U1 or not U2:
                    continue
                all_u = U1 | U2
                if len(all_u) > ball_size // 2: # Heuristic limit
                    continue
                # candidate must cover every point in all_u
                cand = None
                for w in all_u:
                    cc = centers_covering[w]
                    if cand is None:
                        cand = set(cc)
                    else:
                        cand &= cc
                    if not cand:
                        break
                if not cand:
                    continue
                cand = [c for c in cand if c not in code_set and c != i1 and c != i2]
                if not cand:
                    continue
                # choose candidate minimizing waste
                best_j = None
                best_waste = None
                best_new = -1
                for j in cand[:10]: # Limit candidates
                    if time.time() > end_time:
                        return
                    bpj = balls_arr[j]
                    mask = np.zeros(total, dtype=np.bool_)
                    mask[bp1] = True
                    mask[bp2] = True
                    eff = cover_count[bpj] - mask[bpj].astype(np.int32)
                    waste = int(np.sum(eff >= 2))
                    new_cov = int(np.sum(eff == 0))
                    if best_waste is None or waste < best_waste or (waste == best_waste and new_cov > best_new):
                        best_waste = waste
                        best_new = new_cov
                        best_j = j
                if best_j is None:
                    continue
                j = best_j
                cover_count[bp1] -= 1
                cover_count[bp2] -= 1
                cover_count[balls_arr[j]] += 1
                code_set.discard(i1)
                code_set.discard(i2)
                code_set.add(j)
                # rebuild code list: drop i1,i2, append j
                newcode = [c for c in code if c != i1 and c != i2]
                newcode.append(j)
                code[:] = newcode
                improved = True
                break

    # ---------- Full polish: trim, single-swap, then 2-opt merge ----------
    def polish(code, cover_count, code_set, end_time):
        trim_inplace(code, cover_count, code_set)
        local_search(code, cover_count, code_set, end_time)
        two_opt_merge(code, cover_count, code_set, end_time)
        trim_inplace(code, cover_count, code_set)

    # ---------- Choose the better starting point ----------
    code_struct, cc_struct = structured_init()
    code_set_struct = set(code_struct)
    trim_inplace(code_struct, cc_struct, code_set_struct)

    if time.time() < deadline - 40:
        code_lp, cc_lp = lp_dual_rounding_init()
        code_set_lp = set(code_lp)
        trim_inplace(code_lp, cc_lp, code_set_lp)
        if len(code_lp) < len(code_struct):
            code, cover_count, code_set = code_lp, cc_lp, code_set_lp
        else:
            code, cover_count, code_set = code_struct, cc_struct, code_set_struct
    else:
        code, cover_count, code_set = code_struct, cc_struct, code_set_struct

    if not code:
        code, cover_count = greedy_cover()
        code_set = set(code)
        trim_inplace(code, cover_count, code_set)

    trim_inplace(code, cover_count, code_set)

    # ---------- Iterated local search: polish, perturb, re-polish ----------
    best_len = len(code)
    best_snapshot = (list(code), cover_count.copy(), set(code_set))
    best_end = deadline - 5.0
    il_rng = random.Random(777)
    while time.time() < best_end:
        polish(code, cover_count, code_set, best_end)
        if len(code) < best_len:
            best_len = len(code)
            best_snapshot = (list(code), cover_count.copy(), set(code_set))
        # perturb: drop k random redundant-safe codewords, then greedily restore coverage
        k = il_rng.randint(1, 3)
        removable = [i for i, c in enumerate(code)
                     if (cover_count[balls_arr[c]] > 1).all()]
        if len(removable) < k:
            removable = list(range(len(code)))
        if not removable:
            break
        drop = il_rng.sample(removable, min(k, len(removable)))
        for idx in sorted(drop, reverse=True):
            c = code[idx]
            cover_count[balls_arr[c]] -= 1
            code.pop(idx)
            code_set.discard(c)
        # restore coverage greedily
        uncovered = (cover_count == 0)
        while np.any(uncovered) and time.time() < best_end:
            gains = uncovered[balls_arr].sum(axis=1)
            for c in code:
                gains[c] = -1
            mg = int(gains.max())
            if mg <= 0:
                break
            bc = int(np.argmax(gains))
            code.append(bc)
            code_set.add(bc)
            cover_count[balls_arr[bc]] += 1
            uncovered = (cover_count == 0)
        # restore from best snapshot if we lost ground badly
        if len(code) > best_len + 15:
            code[:] = best_snapshot[0]
            cover_count[:] = best_snapshot[1]
            code_set = set(best_snapshot[2])

    # restore best found
    code[:] = best_snapshot[0]
    cover_count[:] = best_snapshot[1]
    code_set = set(best_snapshot[2])
    trim_inplace(code, cover_count, code_set)

    # ---------- Final high-precision polish pass ----------
    polish(code, cover_count, code_set, deadline - 1.0)
    trim_inplace(code, cover_count, code_set)

    # ---------- Convert back to list of lists of integers ----------
    result = [int_to_digits(c, n, q) for c in code]
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)
    return unique_result

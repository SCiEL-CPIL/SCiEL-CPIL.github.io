import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before the 150s kill

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
    # Convert balls to a list of lists for faster indexing in local search
    balls_list = [list(row) for row in balls]
    balls_arr = balls  # keep numpy array for bulk ops

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
        # Greedy extension of the structured seed
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
        # Fractional set cover: min sum x_c s.t. for all words w: sum_{c covers w} x_c >= 1, x_c >= 0
        # Dual: max sum y_w s.t. for all c: sum_{w in ball(c)} y_w <= 1, y_w >= 0
        # We solve a smoothed dual via projected gradient (soft) to get dual vars y,
        # then round: include c with probability proportional to its "tightness" in the primal
        # shadow (how many dual-tight constraints it covers).
        end_time = time.time() + 25.0
        # Initialize y small
        y = np.full(total, 1.0 / ball_size, dtype=np.float64)
        # Subgradient ascent on dual with per-center capacity constraint sum_{w in ball(c)} y_w <= 1
        steps = 400
        eta = 0.05
        for it in range(steps):
            if time.time() > end_time:
                break
            # Compute for each center the ball sum
            ball_sums = y[balls_arr].sum(axis=1)  # (total,)
            # For each word w, lambda_w = max(0, max_{c covers w} (ball_sums[c] - 1))
            lam = np.zeros(total, dtype=np.float64)
            for w in range(total):
                cc = centers_covering[w]
                if cc is None:
                    continue
                vals = ball_sums[list(cc)]
                m = float(vals.max())
                if m > 1.0:
                    lam[w] = m - 1.0
            # Update y_w += eta*(1 - lam_w), clip to [0, 1/ball_size * 2]
            y += eta * (1.0 - lam)
            np.clip(y, 0.0, 2.0 / ball_size, out=y)
        # Now build a rounded integer cover guided by y:
        cover_count = np.zeros(total, dtype=np.int32)
        code = []
        code_set = set()
        # Seed with centers that have the highest sum of y over their ball
        ball_y_sum = y[balls_arr].sum(axis=1)
        order = np.argsort(-ball_y_sum)
        for c in order:
            if time.time() > end_time:
                break
            if len(code) > 600:  # cap
                break
            code.append(int(c))
            code_set.add(int(c))
            cover_count[balls_arr[c]] += 1
        # Greedy extension to full coverage
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

    # ---------- Choose the better starting point ----------
    # Try structured init (fast, strong) and LP-rounded init; keep the smaller (after trim we compare).
    code_struct, cc_struct = structured_init()
    # Trim structured
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

    code_set_struct = set(code_struct)
    trim_inplace(code_struct, cc_struct, code_set_struct)

    # LP-rounded init (only if we have time)
    if time.time() < deadline - 40:
        code_lp, cc_lp = lp_dual_rounding_init()
        code_set_lp = set(code_lp)
        trim_inplace(code_lp, cc_lp, code_set_lp)
        # Pick the smaller
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

    # ---------- Aggressive trimming of redundant codewords ----------
    trim_inplace(code, cover_count, code_set)

    # ---------- Submodular-style local search: waste-minimizing swaps via inverse index ----------
    def local_search(time_budget=105.0):
        nonlocal code, cover_count, code_set
        end_time = time.time() + time_budget
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
                # Submodular "waste" score: after removing i, adding j, the number of points
                # in ball(j) that are already covered by >=2 remaining centers is "waste".
                # We minimize waste; tie-break by maximizing newly-covered points.
                best_swap = None
                best_waste = None
                best_new = -1
                for j in candidates:
                    if time.time() > end_time:
                        return
                    bp_j = balls_arr[j]
                    # Effective coverage after removing i: eff[p] = cover_count[p] - (1 if p in bp_i else 0)
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

    local_search(time_budget=105.0)
    trim_inplace(code, cover_count, code_set)

    # ---------- Convert back to list of lists of integers ----------
    result = [int_to_digits(c, n, q) for c in code]
    # Ensure distinct
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)
    return unique_result

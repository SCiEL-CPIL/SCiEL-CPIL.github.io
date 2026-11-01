import time
import random
import numpy as np


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin under the 150s kill

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # base[i] = q^(n-1-i), so index = sum digit[i]*base[i]
    base = np.array([1] * n, dtype=np.int64)
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # digits[i] = value of coordinate i in word i  (shape total)
    digits = np.zeros((total, n), dtype=np.int64)
    for i in range(n):
        tmp = np.arange(total, dtype=np.int64)
        digits[:, i] = tmp % q
        tmp //= q

    # Ball matrix: ball_matrix[c] = list of indices in the radius-`radius` ball around c
    # For radius 1 this is exactly {c} ∪ {c ± base[i] for each coord i, v≠digit}
    if radius == 1:
        cols = [np.arange(total, dtype=np.int32)]
        for i in range(n):
            b = np.int64(base[i])
            for v in range(q):
                mask = digits[:, i] != v
                cols.append((np.arange(total, dtype=np.int64) + (v - digits[:, i]) * b)[mask].astype(np.int32))
        ball_matrix = np.concatenate(cols, axis=0).reshape(total, ball_size)
    else:
        # Fallback: build balls explicitly (only used if radius != 1)
        rows = []
        for c in range(total):
            bc = digits[c]
            ball = {c}
            for d in range(total):
                if d == c:
                    continue
                diff = 0
                for i in range(n):
                    if digits[d, i] != bc[i]:
                        diff += 1
                        if diff > radius:
                            break
                if diff <= radius:
                    ball.add(d)
            rows.append(sorted(ball))
        bs = len(rows[0])
        ball_matrix = np.array(rows, dtype=np.int32)
        ball_size = bs

    def idx_to_word(idx):
        w = [0] * n
        t = idx
        for i in range(n - 1, -1, -1):
            w[i] = t % q
            t //= q
        return w

    def greedy_build(seed):
        rng = np.random.default_rng(seed)
        covered = np.zeros(total, dtype=np.uint8)
        code = []
        for _ in range(total):
            # new_cov[c] = number of uncovered points in ball c
            new_cov = (1 - covered[ball_matrix]).sum(axis=1)
            best_c = int(np.argmax(new_cov))
            if new_cov[best_c] <= 0:
                break
            code.append(best_c)
            covered[ball_matrix[best_c]] = 1
            if len(code) >= total:
                break
        return code

    def local_search(code_list, time_budget, seed):
        rng = random.Random(seed)
        code_set = set(code_list)
        # coverage count per point
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[ball_matrix[c]] += 1

        best_size = len(code_set)
        best_code = list(code_set)

        def try_delete():
            nonlocal best_size, best_code
            changed = True
            while changed:
                changed = False
                for c in list(code_set):
                    if np.all(cnt[ball_matrix[c]] >= 2):
                        code_set.remove(c)
                        cnt[ball_matrix[c]] -= 1
                        changed = True
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)

        try_delete()

        t_start = time.time()
        no_success = 0
        max_streak = 20000

        while time.time() - t_start < time_budget:
            if not code_set:
                break
            op = rng.random()
            if op < 0.70:
                # 1-for-1 swap
                c_remove = rng.choice(list(code_set))
                c_add = rng.randrange(total)
                if c_add in code_set or c_add == c_remove:
                    continue
                ball_rem = ball_matrix[c_remove]
                ball_add = ball_matrix[c_add]
                # points only covered by c_remove (and not by c_add) must have cnt>=2
                only_rem = ball_rem[np.isin(ball_rem, ball_add, invert=True)]
                if len(only_rem) == 0 or np.all(cnt[only_rem] >= 2):
                    code_set.remove(c_remove)
                    code_set.add(c_add)
                    cnt[ball_rem] -= 1
                    cnt[ball_add] += 1
                    no_success = 0
                    try_delete()
                else:
                    no_success += 1
            else:
                # 2-for-3 shrink: replace two codewords by one if it covers their union
                if len(code_set) < 2:
                    continue
                a, b = rng.sample(list(code_set), 2)
                ba = ball_matrix[a]
                bb = ball_matrix[b]
                union = np.unique(np.concatenate((ba, bb)))
                # candidate must cover every point in the union
                # find candidates: points in union that are uncovered by others must be in candidate's ball
                must = union[cnt[union] <= 2]  # points covered by <=2 codewords (a and/or b)
                if len(must) > ball_size:
                    no_success += 1
                    continue
                # candidate's ball must contain all `must`
                # pick candidate = a word whose ball contains all must
                # Heuristic: candidate = one of the `must` points (its ball contains itself)
                # Only valid if that ball contains ALL must points
                found = -1
                for cand in must:
                    if np.all(np.isin(must, ball_matrix[cand])):
                        found = int(cand)
                        break
                if found >= 0 and found not in code_set:
                    code_set.discard(a)
                    code_set.discard(b)
                    code_set.add(found)
                    cnt[ba] -= 1
                    cnt[bb] -= 1
                    cnt[ball_matrix[found]] += 1
                    no_success = 0
                    try_delete()
                else:
                    no_success += 1

            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = list(code_set)

            if no_success > max_streak:
                no_success = 0

        return best_code, best_size

    best_code_overall = None
    best_size_overall = total

    # First greedy + LS
    greedy_code = greedy_build(seed=42)
    bc, bs = local_search(greedy_code, time_budget=10.0, seed=0)
    best_code_overall = bc
    best_size_overall = bs

    # Multi-restart with fresh greedy seeds and longer LS
    seed = 1
    while time.time() < deadline - 8.0 and best_size_overall > 160:
        greedy_code = greedy_build(seed=seed)
        time_left = deadline - 8.0 - time.time()
        tb = min(12.0, max(2.0, time_left / 4.0))
        bc, bs = local_search(greedy_code, time_budget=tb, seed=seed)
        if bs < best_size_overall:
            best_size_overall = bs
            best_code_overall = bc
        seed += 1

    # Final polish on the best code found
    if time.time() < deadline - 2.0:
        bc, bs = local_search(best_code_overall, time_budget=deadline - 2.0 - time.time(), seed=9999)
        if bs < best_size_overall:
            best_size_overall = bs
            best_code_overall = bc

    result = [idx_to_word(c) for c in best_code_overall]
    return result

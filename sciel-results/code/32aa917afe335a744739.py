import time
import random
import numpy as np

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin under 150s limit

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # ---- index <-> word ----
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    def idx_to_word(c):
        w = []
        for i in range(n):
            w.append(c % q)
            c //= q
        w.reverse()
        return w

    # ---- precompute balls, coverers, and bitsets ----
    ball = [[] for _ in range(total)]
    coverers = [set() for _ in range(total)]
    full_mask = (1 << total) - 1

    for c in range(total):
        temp = c
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = temp % q
            temp //= q
        ball[c] = [c]
        coverers[c].add(c)
        for coord in range(n):
            orig = digits[coord]
            for v in range(q):
                if v == orig:
                    continue
                w = c + (v - orig) * base[coord]
                ball[c].append(w)
                coverers[w].add(c)

    ball_arr = [np.array(b, dtype=np.int32) for b in ball]
    ball_mask = [0] * total
    for c in range(total):
        m = 0
        for w in ball[c]:
            m |= (1 << w)
        ball_mask[c] = m

    def greedy_build(seed=None):
        if seed is not None:
            random.seed(seed)
        covered = 0
        code = []
        while covered != full_mask:
            # vectorized soft-max over all candidates
            gains = np.zeros(total, dtype=np.int32)
            for c in range(total):
                gains[c] = (ball_mask[c] & ~covered).bit_count()
            maxg = int(gains.max())
            if maxg == 0:
                break
            cands = np.where(gains == maxg)[0]
            best_c = int(cands[random.randrange(len(cands))])
            code.append(best_c)
            covered |= ball_mask[best_c]
        return code

    def try_delete(code_set, cnt):
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                if np.min(cnt[ball_arr[c]]) >= 2:
                    code_set.remove(c)
                    cnt[ball_arr[c]] -= 1
                    changed = True

    def local_search(code_list, time_budget=5.0):
        code_set = set(code_list)
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[ball_arr[c]] += 1

        t_start = time.time()
        best_size = len(code_set)
        best_code = list(code_set)
        try_delete(code_set, cnt)
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)

        no_success = 0
        while time.time() - t_start < time_budget:
            if len(code_set) == 0:
                break

            # 1. deletion
            deleted = False
            for c in list(code_set):
                if np.min(cnt[ball_arr[c]]) >= 2:
                    code_set.remove(c)
                    cnt[ball_arr[c]] -= 1
                    deleted = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
                    break
            if deleted:
                no_success = 0
                continue

            # 2. 1-for-1 swap
            c_remove = random.choice(list(code_set))
            crit = np.where((cnt == 1) & ball_mask[c_remove])[0]
            if len(crit) == 0:
                code_set.remove(c_remove)
                cnt[ball_arr[c_remove]] -= 1
                no_success = 0
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code = list(code_set)
                continue

            # candidates covering all critical words
            coverer_sets = [coverers[w] for w in crit]
            inter = set.intersection(*coverer_sets) if coverer_sets else set()
            inter.difference_update(code_set)
            inter.discard(c_remove)
            if inter:
                c_add = random.choice(list(inter))
                code_set.remove(c_remove)
                code_set.add(c_add)
                cnt[ball_arr[c_remove]] -= 1
                cnt[ball_arr[c_add]] += 1
                no_success = 0
                try_delete(code_set, cnt)
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code = list(code_set)
            else:
                no_success += 1

            # 3. perturbation to escape local minima
            if no_success > 400:
                rand_c = random.randrange(total)
                if rand_c not in code_set:
                    code_set.add(rand_c)
                    cnt[ball_arr[rand_c]] += 1
                    try_delete(code_set, cnt)
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
                no_success = 0

        return best_code, best_size

    best_code_overall = []
    best_size_overall = total

    # first run
    greedy_code = greedy_build(seed=42)
    best_code, best_size = local_search(greedy_code, time_budget=15.0)
    best_code_overall, best_size_overall = best_code, best_size

    # multi-restart
    seed = 0
    while time.time() < deadline - 10.0:
        seed += 1
        greedy_code = greedy_build(seed=seed)
        time_left = deadline - 10.0 - time.time()
        if time_left < 4.0:
            break
        tb = min(12.0, max(3.0, time_left / 6.0))
        best_code, best_size = local_search(greedy_code, time_budget=tb)
        if best_size < best_size_overall:
            best_size_overall = best_size
            best_code_overall = best_code
            if best_size_overall <= 180:
                break

    result = [idx_to_word(idx) for idx in best_code_overall]
    return result

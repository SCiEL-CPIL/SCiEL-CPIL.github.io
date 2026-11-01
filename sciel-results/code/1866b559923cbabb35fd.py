import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # Precompute base
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Generate all words
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q

    # Precompute ball indices
    ball_indices = [[] for _ in range(total)]
    for i in range(total):
        w = words[i]
        ball_indices[i].append(i)
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i].append(new_idx)

    ball_arrays = [np.array(bi, dtype=np.int32) for bi in ball_indices]

    # Build ball_mask as numpy boolean matrix
    ball_mask = np.zeros((total, total), dtype=bool)
    for i in range(total):
        ball_mask[i, ball_arrays[i]] = True

    def greedy_cover(order=None):
        if order is None:
            order = list(range(total))
        covered = np.zeros(total, dtype=bool)
        code = []
        remaining = np.ones(total, dtype=bool)

        while np.any(remaining):
            best_c = -1
            best_new = -1
            for c in order:
                new_cov = np.count_nonzero(remaining & ball_mask[c])
                if new_cov > best_new:
                    best_new = new_cov
                    best_c = c
                    if best_new == ball_size:
                        break
            code.append(best_c)
            remaining[ball_mask[best_c]] = False
        return code

    def delete_redundant(code_set, cnt):
        changed = False
        for c in list(code_set):
            if np.all(cnt[ball_arrays[c]] >= 2):
                code_set.remove(c)
                cnt[ball_arrays[c]] -= 1
                changed = True
        return changed

    def compute_unique_points_fast(code_set, cnt):
        unique = {}
        for c in code_set:
            ball = ball_arrays[c]
            if np.any(cnt[ball] == 1):
                mask = (cnt[ball] == 1)
                unique[c] = ball[mask]
            else:
                unique[c] = np.array([], dtype=np.int32)
        return unique

    def is_valid_swap(c_remove, c_add, unique_points):
        u = unique_points[c_remove]
        if len(u) == 0:
            return True
        return np.all(ball_mask[c_add, u])

    def apply_swap(c_remove, c_add, code_set, cnt):
        code_set.remove(c_remove)
        code_set.add(c_add)
        cnt[ball_arrays[c_remove]] -= 1
        cnt[ball_arrays[c_add]] += 1

    def local_search(code_set, cnt, time_limit=1.0):
        start_time = time.time()
        no_improve_streak = 0
        max_streak = 200

        while time.time() - start_time < time_limit and no_improve_streak < max_streak:
            improved = False
            unique_points = compute_unique_points_fast(code_set, cnt)

            # 1. Deletions
            for c in list(code_set):
                if len(unique_points[c]) == 0:
                    code_set.remove(c)
                    cnt[ball_arrays[c]] -= 1
                    improved = True
                    break
            if improved:
                no_improve_streak = 0
                continue

            # 2. 1-for-1 Swaps
            swap_attempts = 0
            max_swap_attempts = 30
            while swap_attempts < max_swap_attempts:
                swap_attempts += 1
                if len(code_set) == 0:
                    break
                c_remove = random.choice(list(code_set))
                u_rem = unique_points[c_remove]
                if len(u_rem) == 0:
                    code_set.remove(c_remove)
                    cnt[ball_arrays[c_remove]] -= 1
                    improved = True
                    break
                for _ in range(8):
                    c_add = random.randint(0, total - 1)
                    if c_add in code_set or c_add == c_remove:
                        continue
                    if is_valid_swap(c_remove, c_add, unique_points):
                        apply_swap(c_remove, c_add, code_set, cnt)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                no_improve_streak = 0
                continue

            if not improved:
                no_improve_streak += 1

        return code_set, cnt

    # === Initial solution: greedy with multiple orders ===
    best_code = None
    best_size = total + 1
    best_cnt = None

    for s in range(8):
        if time.time() > deadline - 40:
            break
        order = list(range(total))
        if s > 0:
            random.shuffle(order)
        code = greedy_cover(order)
        code_set = set(code)
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[ball_arrays[c]] += 1
        while delete_redundant(code_set, cnt):
            pass
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            best_cnt = cnt.copy()

    if best_code is None:
        best_code = list(range(0, total, max(1, total // 186)))
        best_size = len(best_code)
        best_cnt = np.zeros(total, dtype=np.int32)
        for c in best_code:
            best_cnt[ball_arrays[c]] += 1

    # === Local search polish on best ===
    current_code = set(best_code)
    cnt = best_cnt.copy()
    current_code, cnt = local_search(current_code, cnt, time_limit=8.0)
    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = list(current_code)
        best_cnt = cnt.copy()

    # === Iterative improvement with restarts ===
    sa_best_code = set(best_code)
    sa_best_cnt = best_cnt.copy()
    sa_best_size = best_size

    while time.time() < deadline - 8.0:
        # Try a new random restart occasionally
        if time.time() < deadline - 60.0:
            # Do a fresh greedy
            order = list(range(total))
            random.shuffle(order)
            code = greedy_cover(order)
            code_set = set(code)
            cnt2 = np.zeros(total, dtype=np.int32)
            for c in code_set:
                cnt2[ball_arrays[c]] += 1
            while delete_redundant(code_set, cnt2):
                pass
            code_set, cnt2 = local_search(code_set, cnt2, time_limit=5.0)
            if len(code_set) < sa_best_size:
                sa_best_size = len(code_set)
                sa_best_code = set(code_set)
                sa_best_cnt = cnt2.copy()
            continue

        # Otherwise do local search on current best with random perturbations
        current_code = set(sa_best_code)
        cnt = sa_best_cnt.copy()

        # Random perturbation: remove a few and re-add
        if len(current_code) > 5:
            k = random.randint(1, 3)
            to_remove = random.sample(list(current_code), k)
            for c in to_remove:
                current_code.remove(c)
                cnt[ball_arrays[c]] -= 1
            # Re-cover uncovered
            uncovered = np.where(cnt == 0)[0]
            if len(uncovered) > 0:
                for _ in range(min(20, len(uncovered))):
                    pt = random.choice(uncovered)
                    candidates = np.where(ball_mask[:, pt])[0]
                    # Prefer candidates that cover most uncovered
                    best_c = -1
                    best_cov = -1
                    for c in candidates:
                        if c in current_code:
                            continue
                        cov = np.count_nonzero((cnt[ball_arrays[c]] == 0))
                        if cov > best_cov:
                            best_cov = cov
                            best_c = c
                    if best_c >= 0:
                        current_code.add(best_c)
                        cnt[ball_arrays[best_c]] += 1
                    else:
                        # Just add the first candidate
                        c = candidates[0]
                        current_code.add(c)
                        cnt[ball_arrays[c]] += 1

        # Polish
        current_code, cnt = local_search(current_code, cnt, time_limit=5.0)
        while delete_redundant(current_code, cnt):
            pass

        if len(current_code) < sa_best_size:
            sa_best_size = len(current_code)
            sa_best_code = set(current_code)
            sa_best_cnt = cnt.copy()

    # Final polish
    final_code = set(sa_best_code)
    final_cnt = sa_best_cnt.copy()
    final_code, final_cnt = local_search(final_code, final_cnt, time_limit=5.0)
    while delete_redundant(final_code, final_cnt):
        pass

    result = []
    for idx in final_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)

    return result

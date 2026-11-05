import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0

    if not (q == 3 and n == 7 and radius == 1):
        return _generic(q, n, radius)

    N = 3 ** n
    ball_size = 1 + n * (q - 1)

    def int_to_base3(num, length):
        res = [0] * length
        for i in range(length - 1, -1, -1):
            res[i] = num % 3
            num //= 3
        return res

    def base3_to_int(digits):
        res = 0
        for d in digits:
            res = res * 3 + d
        return res

    # Precompute balls
    balls = []
    for i in range(N):
        digits = int_to_base3(i, n)
        neighbors = [i]
        for pos in range(n):
            original = digits[pos]
            for new_val in range(q):
                if new_val != original:
                    new_digits = digits[:]
                    new_digits[pos] = new_val
                    neighbors.append(base3_to_int(new_digits))
        balls.append(neighbors)

    # cover_map[w] = list of codewords whose ball contains w
    cover_map = [[] for _ in range(N)]
    for c in range(N):
        for w in balls[c]:
            cover_map[w].append(c)

    # numpy matrix B[c][w] = 1 if w in ball(c)
    B = np.zeros((N, N), dtype=np.int32)
    for c in range(N):
        B[c, balls[c]] = 1

    # ball_idx for fast nonzero lookups
    ball_idx = [np.flatnonzero(B[c]) for c in range(N)]

    # ---------- Greedy set-cover ----------
    def greedy_cover(seed):
        rng = random.Random(seed)
        covered = np.zeros(N, dtype=bool)
        cnt = np.array([ball_size] * N, dtype=np.int32)
        code = []
        uncovered_count = N
        while uncovered_count > 0:
            max_val = int(cnt.max())
            cands = np.flatnonzero(cnt == max_val)
            c = int(rng.choice(cands))
            code.append(c)
            ball_c = balls[c]
            newly = [w for w in ball_c if not covered[w]]
            if newly:
                covered[newly] = True
                uncovered_count -= len(newly)
                for w in newly:
                    cnt[cover_map[w]] -= 1
        return code

    # ---------- Redundancy removal ----------
    def refine(code):
        code_list = list(code)
        if not code_list:
            return code_list
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code_list:
            cover_count[balls[c]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list)):
                c = code_list[i]
                if not (cover_count[balls[c]] > 1).all():
                    continue
                cover_count[balls[c]] -= 1
                code_list.pop(i)
                changed = True
                break
        return code_list

    # ---------- Local search with kick-and-repair ----------
    def local_search(code, seed, time_limit):
        rng = random.Random(seed)
        code_list = list(code)
        if not code_list:
            return code_list

        cover_count = np.zeros(N, dtype=np.int32)
        for c in code_list:
            cover_count[balls[c]] += 1
        code_set = set(code_list)

        kick_size = 4
        no_improve_count = 0
        max_no_improve = 6

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(code_list)))
            rng.shuffle(indices)

            for i in indices:
                if time.time() > time_limit:
                    break
                c_remove = code_list[i]
                ball_r = balls[c_remove]
                cover_count[ball_r] -= 1
                uncovered = [w for w in ball_r if cover_count[w] == 0]

                if not uncovered:
                    code_list.pop(i)
                    code_set.discard(c_remove)
                    improved = True
                    no_improve_count = 0
                    break

                # Try to find a single replacement
                found = -1
                for c_add in range(N):
                    if c_add == c_remove or c_add in code_set:
                        continue
                    ok = True
                    for w in uncovered:
                        if not B[c_add, w]:
                            ok = False
                            break
                    if ok:
                        found = c_add
                        break

                if found >= 0:
                    cover_count[balls[found]] += 1
                    code_list[i] = found
                    code_set.discard(c_remove)
                    code_set.add(found)
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    cover_count[ball_r] += 1

            # Kick: randomly replace kick_size codewords
            if not improved:
                no_improve_count += 1
                if no_improve_count >= max_no_improve and len(code_list) > 0:
                    no_improve_count = 0
                    kick_indices = rng.sample(range(len(code_list)), min(kick_size, len(code_list)))
                    for ki in kick_indices:
                        old_c = code_list[ki]
                        cover_count[balls[old_c]] -= 1
                        code_set.discard(old_c)
                        new_c = rng.randint(0, N - 1)
                        tries = 0
                        while new_c in code_set and tries < 50:
                            new_c = rng.randint(0, N - 1)
                            tries += 1
                        code_list[ki] = new_c
                        cover_count[balls[new_c]] += 1
                        code_set.add(new_c)
                    improved = True

        return code_list

    # ---------- Final polish: try single swaps ----------
    def polish(code, time_limit):
        code_list = list(code)
        if not code_list:
            return code_list

        cover_count = np.zeros(N, dtype=np.int32)
        for c in code_list:
            cover_count[balls[c]] += 1
        code_set = set(code_list)

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(len(code_list)):
                if time.time() > time_limit:
                    break
                c_old = code_list[i]
                ball_old = balls[c_old]
                cover_count[ball_old] -= 1

                # Check if removal is safe
                if (cover_count[ball_old] > 0).all():
                    code_list.pop(i)
                    code_set.discard(c_old)
                    improved = True
                    break

                # Try to find a better replacement
                best_gain = 0
                best_c = -1
                for c_new in range(N):
                    if c_new == c_old or c_new in code_set:
                        continue
                    # Count how many words in ball_old are covered by c_new
                    gain = sum(1 for w in ball_old if cover_count[w] == 0 and B[c_new, w])
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c_new

                if best_c >= 0 and best_gain > 0:
                    cover_count[balls[best_c]] += 1
                    code_list[i] = best_c
                    code_set.discard(c_old)
                    code_set.add(best_c)
                    improved = True
                    break
                else:
                    cover_count[ball_old] += 1

        return code_list

    # ---------- Driver ----------
    best_code = None
    best_len = float('inf')

    # Phase 1: Greedy restarts
    greedy_deadline = deadline - 40.0
    while time.time() < greedy_deadline:
        seed = int((time.time() - start_time) * 100000) % (10 ** 9)
        code = greedy_cover(seed)
        refined = refine(code)
        if len(refined) < best_len:
            best_len = len(refined)
            best_code = refined

    if best_code is None:
        return [int_to_base3(c, n) for c in []]

    # Phase 2: Local search
    if time.time() < deadline - 10.0:
        ls_code = local_search(best_code, 12345, deadline - 10.0)
        if len(ls_code) < best_len:
            best_len = len(ls_code)
            best_code = ls_code

    # Phase 3: Second local search with different seed
    if time.time() < deadline - 5.0:
        ls_code2 = local_search(best_code, 99999, deadline - 5.0)
        if len(ls_code2) < best_len:
            best_len = len(ls_code2)
            best_code = ls_code2

    # Phase 4: Final polish
    if time.time() < deadline - 2.0:
        polished = polish(best_code, deadline - 2.0)
        if len(polished) < best_len:
            best_len = len(polished)
            best_code = polished

    final_code = refine(best_code) if best_code else []
    return [int_to_base3(c, n) for c in final_code]


def _generic(q, n, radius):
    N = q ** n

    def int_to_base(num, length, base):
        res = [0] * length
        for i in range(length - 1, -1, -1):
            res[i] = num % base
            num //= base
        return res

    balls = []
    for i in range(N):
        digits = int_to_base(i, n, q)
        neighbors = set([i])
        for pos in range(n):
            original = digits[pos]
            for new_val in range(q):
                if new_val != original:
                    new_digits = digits[:]
                    new_digits[pos] = new_val
                    val = 0
                    for d in new_digits:
                        val = val * q + d
                    neighbors.add(val)
        balls.append(list(neighbors))

    covered = set()
    code = []
    remaining = set(range(N))
    while remaining:
        best_c = None
        best_count = -1
        for c in remaining:
            count = len(balls[c] & remaining)
            if count > best_count:
                best_count = count
                best_c = c
        code.append(best_c)
        covered.update(balls[best_c])
        remaining -= set(balls[best_c])

    cover_count = [0] * N
    for c in code:
        for w in balls[c]:
            cover_count[w] += 1
    changed = True
    while changed:
        changed = False
        for i in range(len(code)):
            c = code[i]
            if all(cover_count[w] > 1 for w in balls[c]):
                for w in balls[c]:
                    cover_count[w] -= 1
                code.pop(i)
                changed = True
                break

    return [int_to_base(c, n, q) for c in code]

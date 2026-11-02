import time
import random
import sys
import numpy as np

# EVOLVE-BLOCK-START

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before the 150s kill (120s task limit)

    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)

    return _construct_general(q, n, radius, deadline)


def _construct_3_7_1(deadline):
    N = 2187  # 3^7
    p3 = [1] * 7
    for i in range(1, 7):
        p3[i] = p3[i - 1] * 3

    digits_table = [[0] * 7 for _ in range(N)]
    for i in range(N):
        tmp = i
        for pos in range(7):
            digits_table[i][pos] = tmp % 3
            tmp //= 3

    # ball_masks[i]: bitmask of words within Hamming distance 1 of word i
    ball_masks = [0] * N
    for i in range(N):
        mask = 1 << i
        for pos in range(7):
            orig = digits_table[i][pos]
            for val in range(3):
                if val != orig:
                    mask |= (1 << (i + (val - orig) * p3[pos]))
        ball_masks[i] = mask

    ball_lists = [[] for _ in range(N)]
    for i in range(N):
        m = ball_masks[i]
        lst = []
        while m:
            lsb = m & -m
            lst.append(lsb.bit_length() - 1)
            m ^= lsb
        ball_lists[i] = lst

    full_mask = (1 << N) - 1
    FULL = full_mask

    # ---- vectorized coverage helpers (numpy int16, max per-word count <= 186 < 32767) ----
    def compute_cnt(code):
        cnt = np.zeros(N, dtype=np.int16)
        for c in code:
            for w in ball_lists[c]:
                cnt[w] += 1
        return cnt

    def decrement(cnt, c):
        for w in ball_lists[c]:
            cnt[w] -= 1

    def increment(cnt, c):
        for w in ball_lists[c]:
            cnt[w] += 1

    # ---- greedy initialization ----
    def greedy_init(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        selected_set = set()
        order = list(range(N))
        rng.shuffle(order)
        while uncovered:
            best_idx = -1
            best_count = -1
            for i in order:
                if i in selected_set:
                    continue
                cnt = (uncovered & ball_masks[i]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
                    if best_count == uncovered.bit_count():
                        break
            if best_idx == -1 or best_count == 0:
                break
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~ball_masks[best_idx]
            if time.time() > deadline - 8.0:
                break
        return selected

    # ---- local search ----
    def local_search(init, seed, budget):
        rng = random.Random(seed)
        code = list(init)
        code_set = set(code)
        cnt = compute_cnt(code)

        best_code = code[:]
        best_size = len(code)
        no_improve = 0

        def record():
            nonlocal best_code, best_size, no_improve
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
            no_improve = 0

        def delete_pass():
            nonlocal no_improve
            removed = 0
            changed = True
            while changed:
                changed = False
                for c in list(code):
                    if cnt[ball_lists[c]].min() >= 2:
                        code.remove(c)
                        code_set.discard(c)
                        decrement(cnt, c)
                        removed += 1
                        changed = True
            if removed:
                record()
            return removed

        def try_swap(c, cp):
            if cp in code_set or cp == c:
                return False
            bl = ball_lists[c]
            add = set(ball_lists[cp])
            for w in bl:
                if w not in add and cnt[w] < 2:
                    return False
            code.remove(c)
            code_set.discard(c)
            decrement(cnt, c)
            code.append(cp)
            code_set.add(cp)
            increment(cnt, cp)
            return True

        def get_valid_swaps(c):
            bl = ball_lists[c]
            unique = [w for w in bl if cnt[w] == 1]
            if not unique:
                return None  # redundant -> any non-code word works
            vm = ball_masks[unique[0]]
            for u in unique[1:]:
                vm &= ball_masks[u]
                if vm == 0:
                    return []
            out = []
            m = vm
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if idx not in code_set:
                    out.append(idx)
                m ^= lsb
            return out

        def two_for_one(c1, c2):
            if c1 == c2:
                return False
            # words covered only by c1 or c2
            need = []
            for w in ball_lists[c1]:
                if cnt[w] == 1:
                    need.append(w)
            for w in ball_lists[c2]:
                if cnt[w] == 1:
                    need.append(w)
            if not need:
                # both redundant -> drop one
                code.remove(c2)
                code_set.discard(c2)
                decrement(cnt, c2)
                record()
                return True
            # candidate must cover every word in `need`
            vm = ball_masks[need[0]]
            for w in need[1:]:
                vm &= ball_masks[w]
                if vm == 0:
                    return False
            vm &= ~(1 << c1)
            vm &= ~(1 << c2)
            if vm == 0:
                return False
            # vectorized check: after swap every word must stay covered
            snap = cnt.copy()
            decrement(cnt, c1)
            decrement(cnt, c2)
            m = vm
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if idx in code_set:
                    m ^= lsb
                    continue
                increment(cnt, idx)
                if cnt.min() >= 1:
                    code.remove(c1)
                    code_set.discard(c1)
                    code.remove(c2)
                    code_set.discard(c2)
                    code.append(idx)
                    code_set.add(idx)
                    record()
                    return True
                decrement(cnt, idx)
                m ^= lsb
            cnt[:] = snap
            return False

        def three_for_two(c1, c2, c3):
            if len({c1, c2, c3}) < 3:
                return False
            need = []
            for c in (c1, c2, c3):
                for w in ball_lists[c]:
                    if cnt[w] == 1:
                        need.append(w)
            if not need:
                code.remove(c3)
                code_set.discard(c3)
                decrement(cnt, c3)
                record()
                return True
            vm = ball_masks[need[0]]
            for w in need[1:]:
                vm &= ball_masks[w]
                if vm == 0:
                    return False
            vm &= ~(1 << c1)
            vm &= ~(1 << c2)
            vm &= ~(1 << c3)
            if vm == 0:
                return False
            snap = cnt.copy()
            for c in (c1, c2, c3):
                decrement(cnt, c)
            m = vm
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if idx in code_set:
                    m ^= lsb
                    continue
                increment(cnt, idx)
                if cnt.min() >= 1:
                    for c in (c1, c2, c3):
                        code.remove(c)
                        code_set.discard(c)
                    code.append(idx)
                    code_set.add(idx)
                    record()
                    return True
                decrement(cnt, idx)
                m ^= lsb
            cnt[:] = snap
            return False

        # ---- main loop ----
        while time.time() < budget:
            delete_pass()

            # 1-for-1 swaps
            swaps = 0
            while swaps < 400 and time.time() < budget:
                swaps += 1
                if not code:
                    break
                c = rng.choice(code)
                valid = get_valid_swaps(c)
                if valid is None:
                    cp = rng.randint(0, N - 1)
                    t = 0
                    while cp in code_set and t < 100:
                        cp = rng.randint(0, N - 1)
                        t += 1
                    if cp in code_set:
                        continue
                elif not valid:
                    no_improve += 1
                    continue
                else:
                    cp = rng.choice(valid)
                if try_swap(c, cp):
                    delete_pass()
                else:
                    no_improve += 1

            # 2-for-1 exchanges (vectorized)
            if time.time() < budget - 2.0 and len(code) >= 2:
                for _ in range(60):
                    if time.time() > budget - 2.0:
                        break
                    a, b = rng.sample(code, 2)
                    two_for_one(a, b)

            # 3-for-2 exchanges (vectorized)
            if time.time() < budget - 2.0 and len(code) >= 3:
                for _ in range(30):
                    if time.time() > budget - 2.0:
                        break
                    a, b, c = rng.sample(code, 3)
                    three_for_two(a, b, c)

            # perturbation when stuck
            if no_improve >= 12 and code:
                no_improve = 0
                c_out = rng.choice(code)
                c_in = rng.randint(0, N - 1)
                t = 0
                while c_in in code_set and t < 200:
                    c_in = rng.randint(0, N - 1)
                    t += 1
                if c_in in code_set:
                    continue
                decrement(cnt, c_out)
                increment(cnt, c_in)
                if cnt.min() >= 1:
                    code.remove(c_out)
                    code_set.discard(c_out)
                    code.append(c_in)
                    code_set.add(c_in)
                else:
                    decrement(cnt, c_in)
                    increment(cnt, c_out)
                    no_improve = 0

            no_improve += 1

        return best_code

    best_overall = None
    best_size = float('inf')
    seed = 42
    while time.time() < deadline - 5.0:
        init = greedy_init(seed)
        seed += 1
        if not init:
            break
        ls_budget = min(deadline - 3.0, time.time() + 18.0)
        res = local_search(init, seed, ls_budget)
        seed += 1
        if len(res) < best_size:
            best_size = len(res)
            best_overall = res[:]
        if best_size <= 170:
            break

    if not best_overall:
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        idx_list = []
        for w in words:
            idx = 0
            for pos, d in enumerate(w):
                idx += d * p3[pos]
            idx_list.append(idx)
        best_overall = idx_list

    result = []
    for idx in best_overall:
        digits = [0] * 7
        tmp = idx
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        digits.reverse()
        result.append(digits)
    return result


def _construct_general(q, n, radius, deadline):
    total = q ** n
    words = list(range(total))

    def decode(val):
        return [val // (q ** (n - i - 1)) % q for i in range(n)]

    def encode(digits):
        val = 0
        for i, d in enumerate(digits):
            val += d * (q ** (n - i - 1))
        return val

    balls = {}
    for w in words:
        digs = decode(w)
        neighbors = {w}
        for pos in range(n):
            original = digs[pos]
            for new_val in range(q):
                if new_val != original:
                    nd = digs[:]
                    nd[pos] = new_val
                    neighbors.add(encode(nd))
        balls[w] = neighbors

    uncovered = set(words)
    selected = []
    while uncovered and time.time() < deadline - 5.0:
        best_w = max(words, key=lambda w: len(balls[w] & uncovered))
        selected.append(best_w)
        uncovered -= balls[best_w]

    if selected:
        counts = {w: 0 for w in words}
        for c in selected:
            for w in balls[c]:
                counts[w] += 1
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                c = selected[i]
                if all(counts[w] > 1 for w in balls[c]):
                    for w in balls[c]:
                        counts[w] -= 1
                    selected.pop(i)
                    changed = True
    return [decode(w) for w in selected]

# EVOLVE-BLOCK-END

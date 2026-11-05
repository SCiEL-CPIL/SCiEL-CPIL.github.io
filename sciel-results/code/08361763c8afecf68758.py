import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin under 150s

    total = q ** n
    rng = random.Random(12345)

    # ---------- helpers ----------
    def ball(c):
        s = {c}
        for i in range(n):
            orig = c[i]
            for v in range(q):
                if v == orig:
                    continue
                nw = c[:i] + [v] + c[i+1:]
                s.add(tuple(nw))
        return s

    def is_covered(code, w):
        for c in code:
            d = 0
            for i in range(n):
                if c[i] != w[i]:
                    d += 1
                    if d > radius:
                        break
            if d <= radius:
                return True
        return False

    def uncovered_list(code):
        return [w for w in range(total) if not is_covered(code, w)]

    # ---------- fast greedy (bitmask) ----------
    # precompute ball masks
    ball_masks = [0] * total
    for w in range(total):
        digits = [0] * n
        t = w
        for i in range(n):
            digits[i] = t % q
            t //= q
        m = 1 << w
        for i in range(n):
            orig = digits[i]
            for v in range(q):
                if v == orig:
                    continue
                t2 = w
                # recompute index with digit i set to v
                # w = sum digits[k]*q^k
                # new = w - orig*q^i + v*q^i
                new = w + (v - orig) * (q ** i)
                m |= 1 << new
        ball_masks[w] = m

    # all ones mask
    ALL = (1 << total) - 1

    code = []
    covered = 0
    while covered < total:
        if time.time() > deadline:
            break
        best_c = -1
        best_gain = -1
        # only consider candidates covering at least one uncovered word
        # iterate all; cheap enough for 729
        for c in range(total):
            g = bin(ball_masks[c] & ~covered).count('1')
            if g > best_gain:
                best_gain = g
                best_c = c
        if best_c < 0:
            break
        code.append(best_c)
        covered |= ball_masks[best_c]

    # convert to list-of-lists representation
    def int_to_word(x):
        d = []
        t = x
        for i in range(n):
            d.append(t % q)
            t //= q
        return d

    code = [int_to_word(c) for c in code]
    initial_size = len(code)

    # ---------- add-remove local search ----------
    # index: word tuple -> index in code
    def code_index(code_list):
        return {tuple(w): i for i, w in enumerate(code_list)}

    idx = code_index(code)

    def remove_candidate(code_list, ci):
        # returns (new_code_list, idx, uncovered_count) or None
        c = code_list[ci]
        # find words covered only by c
        only = []
        # words in ball(c) not covered by any other codeword
        for w in ball(c):
            ok = False
            for j, cj in enumerate(code_list):
                if j == ci:
                    continue
                # distance w vs cj
                d = 0
                for i in range(n):
                    if w[i] != cj[i]:
                        d += 1
                        if d > radius:
                            break
                if d <= radius:
                    ok = True
                    break
            if not ok:
                only.append(w)
        # find replacement candidates: words in only, check coverage gain
        best_repl = None
        best_net = -1
        for w in only:
            # gain from adding w: words in ball(w) not currently covered
            gain = 0
            for u in ball(w):
                if not is_covered(code_list, u):
                    gain += 1
            net = gain - len(only)
            if net > best_net:
                best_net = net
                best_repl = w
        if best_repl is None or best_net <= 0:
            return None
        new_code = [w for i, w in enumerate(code_list) if i != ci]
        new_code.append(list(best_repl))
        return new_code, code_index(new_code)

    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        ci = 0
        while ci < len(code) and time.time() < deadline - 5.0:
            res = remove_candidate(code, ci)
            if res is not None:
                code, idx = res
                improved = True
                ci = 0  # restart scan since code changed
                continue
            ci += 1

    # ---------- fallback: ensure validity ----------
    # If we removed too much (shouldn't happen since we only accept net>0),
    # but verify and fix by greedy adding uncovered.
    unc = uncovered_list(code)
    while unc:
        if time.time() > deadline:
            break
        # greedy pick max coverage among uncovered
        best_c = None
        best_cov = -1
        # candidates: uncovered words themselves
        for w in unc:
            cov = 0
            for u in unc:
                d = 0
                for i in range(n):
                    if w[i] != u[i]:
                        d += 1
                        if d > radius:
                            break
                if d <= radius:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                best_c = w
        if best_c is None:
            break
        code.append(list(best_c))
        unc = uncovered_list(code)

    # final validity check; if invalid, fall back to pure greedy result
    # (rebuild greedy result)
    final_unc = uncovered_list(code)
    if final_unc:
        # rebuild from scratch greedily
        code2 = []
        covered2 = 0
        while covered2 < total:
            best_c = -1
            best_gain = -1
            for c in range(total):
                g = bin(ball_masks[c] & ~covered2).count('1')
                if g > best_gain:
                    best_gain = g
                    best_c = c
            if best_c < 0:
                break
            code2.append(int_to_word(best_c))
            covered2 |= ball_masks[best_c]
        code = code2

    # dedupe (safety)
    seen = set()
    out = []
    for w in code:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            out.append(w)
    return out

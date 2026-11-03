import time
import random

def construct(q, n, radius):
    start = time.time()
    deadline = start + 112.0  # safety margin under 120s

    # ---- Precompute sphere (set of words within Hamming distance `radius`) ----
    # For radius 1: the word itself + all single-position changes.
    if radius == 1:
        spheres = []
        for i in range(q ** n):
            x = i
            digits = [0] * n
            for p in range(n):
                digits[p] = x % q
                x //= q
            s = 1 << i
            for p in range(n):
                base = digits[p]
                for nv in range(q):
                    if nv != base:
                        j = i + (nv - base) * (q ** p)
                        s |= 1 << j
            spheres.append(s)
    else:
        # General (slower) fallback
        spheres = []
        for i in range(q ** n):
            x = i
            d0 = [0] * n
            for p in range(n):
                d0[p] = x % q
                x //= q
            s = 0
            from itertools import combinations, product
            for r in range(radius + 1):
                for pos in combinations(range(n), r):
                    vals = list(d0)
                    for pi in pos:
                        for nv in range(q):
                            if nv != d0[pi]:
                                vals[pi] = nv
                                x2 = 0
                                for p in range(n):
                                    x2 += vals[p] * (q ** p)
                                s |= 1 << x2
                                vals[pi] = d0[pi]
            spheres.append(s)

    N = q ** n
    all_bits = (1 << N) - 1
    best_code = None
    best_size = 10 ** 9

    def greedy_cover(covered, rng):
        """Pick codewords to cover all remaining uncovered words."""
        code = []
        remaining = all_bits ^ covered
        while remaining:
            best = -1
            best_cnt = -1
            # iterate only over uncovered words as candidate centers
            rem = remaining
            while rem:
                lb = rem & (-rem)
                i = lb.bit_length() - 1
                cnt = (spheres[i] & remaining).bit_count()
                if cnt > best_cnt or (cnt == best_cnt and rng.random() < 0.5):
                    best_cnt = cnt
                    best = i
                rem ^= lb
            code.append(best)
            remaining &= ~spheres[best]
        return code

    def local_search(code, rng):
        """Reduce code via remove / 1-swap / 2-opt until no improvement."""
        code = list(code)
        improved = True
        while improved:
            improved = False
            # coverage count per word
            cov = [0] * N
            for c in code:
                s = spheres[c]
                while s:
                    lb = s & (-s)
                    cov[lb.bit_length() - 1] += 1
                    s ^= lb
            # 1) removals
            kept = []
            for c in code:
                s = spheres[c]
                ok = True
                t = s
                while t:
                    lb = t & (-t)
                    if cov[lb.bit_length() - 1] <= 1:
                        ok = False
                        break
                    t ^= lb
                if ok:
                    t = s
                    while t:
                        lb = t & (-t)
                        cov[lb.bit_length() - 1] -= 1
                        t ^= lb
                    improved = True
                else:
                    kept.append(c)
            code = kept
            # 2) 1-for-1 swap (remove c, add w)
            cov = [0] * N
            for c in code:
                s = spheres[c]
                while s:
                    lb = s & (-s)
                    cov[lb.bit_length() - 1] += 1
                    s ^= lb
            for idx in range(len(code)):
                c = code[idx]
                # words only covered by c
                s = spheres[c]
                only = s
                t = s
                while t:
                    lb = t & (-t)
                    if cov[lb.bit_length() - 1] > 1:
                        only ^= lb
                    t ^= lb
                if only == 0:
                    continue
                w = only
                while w:
                    lb = w & (-w)
                    j = lb.bit_length() - 1
                    if (spheres[j] & only) == only:
                        # do swap
                        for k in code:
                            if k == c:
                                code[code.index(c)] = j
                                break
                        # recompute cov
                        cov = [0] * N
                        for cc in code:
                            ss = spheres[cc]
                            while ss:
                                lb2 = ss & (-ss)
                                cov[lb2.bit_length() - 1] += 1
                                ss ^= lb2
                        improved = True
                        break
                    w ^= lb
                if improved:
                    break
            # 3) 2-opt: remove two (c1,c2), add one w
            cov = [0] * N
            for c in code:
                s = spheres[c]
                while s:
                    lb = s & (-s)
                    cov[lb.bit_length() - 1] += 1
                    s ^= lb
            for a in range(len(code)):
                for b in range(a + 1, len(code)):
                    c1, c2 = code[a], code[b]
                    s1, s2 = spheres[c1], spheres[c2]
                    only1 = s1
                    t = s1
                    while t:
                        lb = t & (-t)
                        if cov[lb.bit_length() - 1] > 1:
                            only1 ^= lb
                        t ^= lb
                    only2 = s2
                    t = s2
                    while t:
                        lb = t & (-t)
                        if cov[lb.bit_length() - 1] > 1:
                            only2 ^= lb
                        t ^= lb
                    only = only1 | only2
                    if only == 0:
                        continue
                    w = only
                    found = False
                    while w:
                        lb = w & (-w)
                        j = lb.bit_length() - 1
                        if (spheres[j] & only) == only:
                            code = [c for c in code if c != c1 and c != c2]
                            code.append(j)
                            improved = True
                            found = True
                            break
                        w ^= lb
                    if found:
                        break
                if improved:
                    break
        return code

    # ---- Phase 1: deterministic greedy seed ----
    seed = greedy_cover(0, random.Random(0))
    seed = local_search(seed, random.Random(1))
    best_code = seed
    best_size = len(seed)

    # ---- Phase 2: randomized multi-start + perturb + local search ----
    rng = random.Random(12345)
    while True:
        if time.time() > deadline - 8.0:
            break
        # random start with skip
        covered = 0
        code = []
        remaining = all_bits
        while remaining:
            # randomly skip some candidates to diversify
            best = -1
            best_cnt = -1
            rem = remaining
            skip = rng.random()
            while rem:
                lb = rem & (-rem)
                i = lb.bit_length() - 1
                cnt = (spheres[i] & remaining).bit_count()
                if cnt > best_cnt or (cnt == best_cnt and rng.random() < 0.5):
                    best_cnt = cnt
                    best = i
                rem ^= lb
            code.append(best)
            remaining &= ~spheres[best]
        # perturb: drop a few random codewords, then repair
        perturb = local_search(code, rng)
        # try a perturbation: remove up to k random, greedy-repair
        cand = list(perturb)
        k = rng.randint(1, 4)
        for _ in range(k):
            if not cand:
                break
            rm = cand.pop(rng.randrange(len(cand)))
            covered = 0
            for c in cand:
                covered |= spheres[c]
            remaining = all_bits ^ covered
            if remaining:
                cand.extend(greedy_cover(covered, rng))
        cand = local_search(cand, rng)
        if len(cand) < best_size:
            best_size = len(cand)
            best_code = cand

    # ---- Final cleanup pass ----
    best_code = local_search(best_code, random.Random(999))

    # ---- Convert to list of digit lists ----
    def to_digits(num):
        digits = []
        for _ in range(n):
            digits.append(num % q)
            num //= q
        return digits

    return [to_digits(c) for c in best_code]

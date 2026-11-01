"""
K_3(7,1) solver: bit-parallel coverage + counting-based local search with swaps.

Structural idea: represent each codeword's radius-1 ball as a 2187-bit integer.
Maintain a coverage count per word (how many codewords cover it). Use this to:
  - delete redundant codewords (all its words have count >= 2),
  - perform swaps (remove c, add c' that covers c's critical words) to escape
    local minima that pure deletion cannot,
  - restarts from greedy set-cover seeds to diversify.
"""
import time
import random
import itertools


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0   # safety margin before the 150s kill

    total = q ** n
    FULL = (1 << total) - 1

    # ---- Build balls as bitmasks (only feasible for this size) ----
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    balls = [0] * total
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                mask |= 1 << word_to_idx[tuple(new_w)]
        balls[i] = mask

    # ---- Greedy set-cover seed (bit-parallel, randomized tie-break) ----
    def greedy_seed(seed):
        rng = random.Random(seed)
        uncovered = FULL
        selected = []
        while uncovered and time.time() < deadline - 1.0:
            best_idx = -1
            best_cnt = -1
            for i in range(total):
                if i in selected:
                    continue
                cnt = (uncovered & balls[i]).bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
                elif cnt == best_cnt and cnt > 0 and rng.random() < 0.1:
                    best_idx = i
            if best_idx == -1 or best_cnt == 0:
                break
            selected.append(best_idx)
            uncovered &= ~balls[best_idx]
        return selected

    # ---- Local search with coverage counts + swap moves ----
    def local_search(code, budget):
        code_set = set(code)
        cnt = [0] * total
        for c in code_set:
            for w in _ball_words(c):
                cnt[w] += 1
        best_size = len(code_set)
        best_code = list(code_set)
        t_start = time.time()
        rng = random.Random(12345)

        def ball_words(c):
            m = balls[c]
            out = []
            while m:
                lsb = m & -m
                out.append(lsb.bit_length() - 1)
                m ^= lsb
            return out

        def try_delete():
            nonlocal best_size, best_code
            changed = True
            while changed:
                changed = False
                for c in list(code_set):
                    # redundant if every word in its ball has count >= 2
                    redundant = True
                    m = balls[c]
                    while m:
                        lsb = m & -m
                        w = lsb.bit_length() - 1
                        if cnt[w] < 2:
                            redundant = False
                            break
                        m ^= lsb
                    if redundant:
                        for w in ball_words(c):
                            cnt[w] -= 1
                        code_set.remove(c)
                        changed = True
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)

        try_delete()

        stuck = 0
        while time.time() - t_start < budget and code_set:
            # 1) free deletions
            deleted = False
            for c in list(code_set):
                redundant = True
                m = balls[c]
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    if cnt[w] < 2:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    for w in ball_words(c):
                        cnt[w] -= 1
                    code_set.remove(c)
                    deleted = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
                    break
            if deleted:
                stuck = 0
                continue

            # 2) try a swap: remove a random c, find a c' covering its critical words
            c = rng.choice(list(code_set))
            # critical words of c: words covered only by c
            critical = []
            m = balls[c]
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                if cnt[w] == 1:
                    critical.append(w)
                m ^= lsb
            if not critical:
                # no critical words -> safe delete
                for w in ball_words(c):
                    cnt[w] -= 1
                code_set.remove(c)
                stuck = 0
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code = list(code_set)
                continue

            # find candidate c' not in code that covers all critical words
            crit_mask = 0
            for w in critical:
                crit_mask |= 1 << w
            cands = []
            for c2 in range(total):
                if c2 in code_set:
                    continue
                if (balls[c2] & crit_mask) == crit_mask:
                    cands.append(c2)
            if cands:
                c2 = rng.choice(cands)
                for w in ball_words(c):
                    cnt[w] -= 1
                for w in ball_words(c2):
                    cnt[w] += 1
                code_set.remove(c)
                code_set.add(c2)
                stuck = 0
                try_delete()
            else:
                stuck += 1
                if stuck > 300:
                    # kick: add a random word, then delete
                    r = rng.randrange(total)
                    if r not in code_set:
                        for w in ball_words(r):
                            cnt[w] += 1
                        code_set.add(r)
                        try_delete()
                    stuck = 0

        return best_code, best_size

    # ---- Orchestrate: multiple greedy seeds + local search ----
    best_code = None
    best_size = total
    seed = 0
    while time.time() < deadline - 5.0:
        code = greedy_seed(seed)
        # quick prune then local search
        budget = min(20.0, max(3.0, (deadline - 5.0 - time.time()) / 4.0))
        code_ls, size_ls = local_search(code, budget)
        if size_ls < best_size:
            best_size = size_ls
            best_code = code_ls
        seed += 1
        if best_size <= 186:
            break

    # final polish on the best found
    if best_code is not None and time.time() < deadline - 2.0:
        tb = deadline - 2.0 - time.time()
        if tb > 1.0:
            code_ls, size_ls = local_search(best_code, tb)
            if size_ls < best_size:
                best_size = size_ls
                best_code = code_ls

    if best_code is None:
        best_code = list(range(total))

    return [list(all_words[i]) for i in best_code]

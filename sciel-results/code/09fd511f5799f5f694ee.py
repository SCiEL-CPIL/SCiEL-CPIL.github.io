import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140

    total_words = q ** n
    ball_size = 1 + n * (q - 1)

    balls = np.empty((total_words, ball_size), dtype=np.int32)
    for idx in range(total_words):
        temp = idx
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = temp % q
            temp //= q
        balls[idx, 0] = idx
        k = 1
        for pos in range(n):
            orig = w[pos]
            for nv in range(q):
                if nv == orig:
                    continue
                w[pos] = nv
                enc = 0
                for d in w:
                    enc = enc * q + d
                balls[idx, k] = enc
                k += 1
            w[pos] = orig

    def time_left():
        return deadline - time.time()

    def greedy_cover(seed, t_limit):
        random.seed(seed)
        covered = np.zeros(total_words, dtype=np.int8)
        code = []
        t0 = time.time()
        while True:
            if time.time() - t0 > t_limit:
                break
            if covered.all():
                break
            gains = ball_size - covered[balls].sum(axis=1)
            mg = int(gains.max())
            if mg <= 0:
                break
            bc = np.where(gains == mg)[0]
            chosen = int(random.choice(bc))
            code.append(chosen)
            covered[balls[chosen]] += 1
        return code

    def fast_reduce(code):
        if not code:
            return code
        cc = np.zeros(total_words, dtype=np.int32)
        for c in code:
            cc[balls[c]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                red = True
                for w in balls[c]:
                    if cc[w] <= 1:
                        red = False
                        break
                if red:
                    for w in balls[c]:
                        cc[w] -= 1
                    code.pop(i)
                    changed = True
        return code

    def sa_optimize(code, t_limit):
        t0 = time.time()
        in_code = np.zeros(total_words, dtype=bool)
        for c in code:
            in_code[c] = True
        cc = np.zeros(total_words, dtype=np.int32)
        for c in code:
            cc[balls[c]] += 1

        uncovered = int((cc == 0).sum())
        selected = len(code)

        lam_u = 100.0
        lam_s = 1.0

        best_code = list(code)
        best_unc = uncovered
        best_sel = selected

        T = 3.0
        T_min = 0.05
        alpha = 0.9998

        # Precompute ball sets as frozensets for swap overlap check
        ball_sets = [frozenset(balls[c].tolist()) for c in range(total_words)]

        while time.time() - t0 < t_limit:
            if T < T_min:
                T = 2.0

            r = random.random()
            if r < 0.5 and selected > 0:
                c = code[random.randrange(selected)]
                du = int((cc[balls[c]] == 1).sum())
                dE = lam_u * du - lam_s
                # Accept/reject
                if dE < 0 or random.random() < np.exp(-dE / T):
                    for w in balls[c]:
                        cc[w] -= 1
                    in_code[c] = False
                    code.remove(c)
                    uncovered += du
                    selected -= 1
                    if uncovered < best_unc or (uncovered == best_unc and selected < best_sel):
                        best_unc = uncovered
                        best_sel = selected
                        best_code = list(code)
                T *= alpha

            elif r < 0.9:
                ns = np.where(~in_code)[0]
                if len(ns) == 0:
                    T *= alpha
                    continue
                c = int(random.choice(ns))
                du = -int((cc[balls[c]] == 0).sum())
                dE = lam_u * du + lam_s
                if dE < 0 or random.random() < np.exp(-dE / T):
                    for w in balls[c]:
                        cc[w] += 1
                    in_code[c] = True
                    code.append(c)
                    uncovered += du
                    selected += 1
                    if uncovered < best_unc or (uncovered == best_unc and selected < best_sel):
                        best_unc = uncovered
                        best_sel = selected
                        best_code = list(code)
                T *= alpha

            else:
                if selected == 0:
                    T *= alpha
                    continue
                cr = code[random.randrange(selected)]
                ns = np.where(~in_code)[0]
                if len(ns) == 0:
                    T *= alpha
                    continue
                ca = int(random.choice(ns))

                # Net uncovered change for swap
                set_cr = ball_sets[cr]
                set_ca = ball_sets[ca]
                # Words uniquely covered by cr (cc==1) and not in ca -> become uncovered
                # Words uncovered (cc==0) in ca but not in cr -> become covered
                du = 0
                for w in balls[cr]:
                    if cc[w] == 1 and w not in set_ca:
                        du += 1
                for w in balls[ca]:
                    if cc[w] == 0 and w not in set_cr:
                        du -= 1
                dE = lam_u * du

                if dE < 0 or random.random() < np.exp(-dE / T):
                    for w in balls[cr]:
                        cc[w] -= 1
                    in_code[cr] = False
                    code.remove(cr)
                    for w in balls[ca]:
                        cc[w] += 1
                    in_code[ca] = True
                    code.append(ca)
                    uncovered += du
                    if uncovered < best_unc or (uncovered == best_unc and selected < best_sel):
                        best_unc = uncovered
                        best_sel = selected
                        best_code = list(code)
                T *= alpha

        return best_code

    best_solution = None
    best_size = float('inf')

    t_g = min(8.0, time_left() * 0.25)
    init = greedy_cover(0, t_g)
    if init:
        init = fast_reduce(init)
        best_solution = init
        best_size = len(init)

    seed = 1
    while time_left() > 12.0:
        t_g = min(4.0, time_left() * 0.15)
        code = greedy_cover(seed, t_g)
        if code:
            code = fast_reduce(code)
            t_sa = min(time_left() * 0.4, 35.0)
            sa = sa_optimize(code, t_sa)
            sa = fast_reduce(sa)
            if len(sa) < best_size:
                best_solution = sa
                best_size = len(sa)

        if time_left() > 18.0 and best_solution:
            t_sa2 = min(time_left() * 0.25, 25.0)
            sa2 = sa_optimize(list(best_solution), t_sa2)
            sa2 = fast_reduce(sa2)
            if len(sa2) < best_size:
                best_solution = sa2
                best_size = len(sa2)

        seed += 1

    if best_solution:
        best_solution = fast_reduce(best_solution)

    if best_solution is None:
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    result = []
    for idx in best_solution:
        word = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            word[i] = temp % q
            temp //= q
        result.append(word)
    return result

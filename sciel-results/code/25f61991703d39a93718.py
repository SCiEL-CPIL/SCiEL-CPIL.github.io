"""Stronger algorithm for integer sum-product extremal sets (n=20, max=10000)."""
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    N = 20
    MAXV = 10000

    # ---- Fast vectorized evaluator ---------------------------------------
    def score(a):
        """Return max(|A+A|, |A*A|) for a sorted length-20 integer array."""
        n = len(a)
        iu, ju = np.triu_indices(n)
        sums = a[iu] + a[ju]
        prods = a[iu] * a[ju]
        s = np.unique(sums).size
        p = np.unique(prods).size
        return s if s > p else p

    def score_detail(a):
        n = len(a)
        iu, ju = np.triu_indices(n)
        sums = a[iu] + a[ju]
        prods = a[iu] * a[ju]
        s = np.unique(sums).size
        p = np.unique(prods).size
        return s, p

    def valid(a):
        if len(a) != N:
            return False
        if a[0] < 1 or a[-1] > MAXV:
            return False
        if np.any(np.diff(a) <= 0):
            return False
        return True

    def make_candidates(a):
        """Return a list of candidate values (sorted) that could replace one element,
        favouring small values and values sharing factors with the set."""
        cand = set()
        a_list = a.tolist()
        # small values
        for v in range(1, 41):
            cand.add(v)
        # multiples / divisors of existing elements (creates product overlaps)
        for v in a_list:
            for d in range(1, 40):
                if v * d <= MAXV:
                    cand.add(v * d)
            for d in range(1, 40):
                if v % d == 0:
                    cand.add(v // d)
        # cap the candidate pool for speed
        if len(cand) > 300:
            cand = list(cand)
            random.shuffle(cand)
            cand = set(cand[:300])
        cand = [c for c in cand if 1 <= c <= MAXV]
        cand.sort()
        return cand

    # ---- Simulated annealing with nudge/swap moves -----------------------
    def anneal(start, t0, t1, iters):
        a = np.array(sorted(start), dtype=np.int64)
        cur = score(a)
        best = cur
        best_a = a.copy()
        cands = make_candidates(a)
        rng = random.Random(hash(tuple(a.tolist())) & 0xFFFFFFFF)
        for it in range(iters):
            if time.time() > deadline:
                break
            T = t0 * (t1 / t0) ** (it / max(1, iters))
            # choose a move
            r = rng.random()
            if r < 0.45:
                # nudge one element to a nearby candidate
                idx = rng.randrange(N)
                old = a[idx]
                # pick a candidate near old value
                cands_list = cands
                # find candidates within a window of old
                window = [c for c in cands_list if abs(c - old) <= max(5, old // 2)]
                if not window:
                    window = cands_list
                new = rng.choice(window)
                if new == old:
                    continue
                a[idx] = new
                if not valid(a):
                    a[idx] = old
                    continue
                a.sort()
                ns = score(a)
            elif r < 0.75:
                # swap two elements' values with candidates
                i1 = rng.randrange(N)
                i2 = rng.randrange(N)
                if i1 == i2:
                    continue
                o1, o2 = a[i1], a[i2]
                c1 = rng.choice(cands)
                c2 = rng.choice(cands)
                a[i1] = c1
                a[i2] = c2
                if not valid(a):
                    a[i1] = o1
                    a[i2] = o2
                    continue
                a.sort()
                ns = score(a)
            else:
                # replace one element with a fresh candidate
                idx = rng.randrange(N)
                old = a[idx]
                new = rng.choice(cands)
                if new == old:
                    continue
                a[idx] = new
                if not valid(a):
                    a[idx] = old
                    continue
                a.sort()
                ns = score(a)
            # accept?
            if ns < cur or rng.random() < np.exp(-(ns - cur) / max(1e-6, T)):
                cur = ns
                if cur < best:
                    best = cur
                    best_a = a.copy()
                # occasionally refresh candidates
                if it % 200 == 0:
                    cands = make_candidates(a)
            else:
                # revert
                if r < 0.45:
                    a[idx] = old
                elif r < 0.75:
                    a[i1] = o1
                    a[i2] = o2
                else:
                    a[idx] = old
                a.sort()
        return best, best_a

    # ---- Start from several structured seeds -----------------------------
    seeds = []

    # 1. sequential
    seeds.append(list(range(1, 21)))
    # 2. divisors of 720
    def divisors(n):
        d = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                d.add(i)
                d.add(n // i)
            i += 1
        return sorted(d)
    for base in (720, 840, 1260, 2520, 5040, 27720):
        dv = [d for d in divisors(base) if d <= MAXV]
        if len(dv) >= N:
            seeds.append(dv[:N])
    # 3. geometric
    seeds.append([2 ** i for i in range(N)])
    # 4. small + multiples
    seeds.append(sorted(set(list(range(1, 11)) + [11 * i for i in range(1, 11)])))
    # 5. lattice: {a*b for small a,b}
    lat = set()
    for a in range(1, 6):
        for b in range(1, 6):
            lat.add(a * b)
    lat = sorted(lat)
    if len(lat) >= N:
        seeds.append(lat[:N])

    best_score = float('inf')
    best_set = None

    # Evaluate all seeds, keep the best
    for s in seeds:
        if not valid(np.array(sorted(s), dtype=np.int64)):
            continue
        sc = score(np.array(sorted(s), dtype=np.int64))
        if sc < best_score:
            best_score = sc
            best_set = np.array(sorted(s), dtype=np.int64)

    # Run simulated annealing from each seed, time-budgeted
    t0 = 10.0
    t1 = 0.05
    for s in seeds:
        if time.time() > deadline:
            break
        if not valid(np.array(sorted(s), dtype=np.int64)):
            continue
        # budget per seed
        budget = 12.0
        start_t = time.time()
        # estimate iterations by timing a small batch
        a = np.array(sorted(s), dtype=np.int64)
        t0time = time.time()
        for _ in range(50):
            score(a)
        per_iter = (time.time() - t0time) / 50.0
        iters = max(100, int(budget / per_iter))
        bs, ba = anneal(s, t0, t1, iters)
        if bs < best_score:
            best_score = bs
            best_set = ba

    # Final local polish on the best set (greedy single-element improvement)
    if best_set is not None and time.time() < deadline:
        a = best_set.copy()
        cur = score(a)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cands = make_candidates(a)
            for idx in range(N):
                if time.time() > deadline:
                    break
                old = a[idx]
                for new in cands:
                    if new == old:
                        continue
                    a[idx] = new
                    if not valid(a):
                        a[idx] = old
                        continue
                    a.sort()
                    ns = score(a)
                    if ns < cur:
                        cur = ns
                        improved = True
                        if cur < best_score:
                            best_score = cur
                            best_set = a.copy()
                        break
                    else:
                        a[idx] = old
                        a.sort()
        if cur < best_score:
            best_score = cur
            best_set = a.copy()

    return [int(x) for x in sorted(best_set.tolist())]

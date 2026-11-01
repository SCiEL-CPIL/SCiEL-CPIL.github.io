"""Covering code K_3(6,1) via greedy + local search with deadline guard."""
import time
import random
import numpy as np

def _to_flat(w, n):
    v = 0
    for i in range(n):
        v = v * 3 + w[i]
    return v

def _from_flat(v, n):
    out = [0] * n
    for i in range(n - 1, -1, -1):
        out[i] = v % 3
        v //= 3
    return out

def _neighbors(v, n):
    yield v
    for i in range(n):
        base = v // (3 ** i)
        cur = base % 3
        for d in range(1, 3):
            newc = (cur + d) % 3
            yield v - cur * (3 ** i) + newc * (3 ** i)

def _build_coverage(codewords, n, q):
    N = q ** n
    cover = np.zeros(N, dtype=np.int32)
    for cw in codewords:
        for nb in _neighbors(_to_flat(cw, n), n):
            cover[nb] += 1
    return cover

def _greedy_init(n, q, rng, seed_words):
    N = q ** n
    cover = np.zeros(N, dtype=np.int32)
    covered = 0
    words = []
    if seed_words:
        for cw in seed_words:
            for nb in _neighbors(_to_flat(cw, n), n):
                if cover[nb] == 0:
                    covered += 1
                cover[nb] += 1
        words = [list(cw) for cw in seed_words]
    all_words = np.arange(N, dtype=np.int32)
    while covered < N:
        gains = np.zeros(N, dtype=np.int32)
        for cw in words:
            for nb in _neighbors(_to_flat(cw, n), n):
                gains[nb] += 1
        # gain of adding w = number of currently-uncovered neighbors
        newg = 0
        for w in all_words:
            c = 0
            for nb in _neighbors(int(w), n):
                if cover[nb] == 0:
                    c += 1
            newg = c if False else newg  # placeholder
        # vectorized gain computation
        gain = np.zeros(N, dtype=np.int32)
        for w in range(N):
            pass
        # Efficient: for each coordinate change, accumulate
        # gain[w] = count of uncovered in {w} U {w with one coord changed}
        # = (1 if cover[w]==0 else 0) + sum_i (1 if cover[w_i']==0 else 0)
        gain = (cover == 0).astype(np.int32).copy()
        for i in range(n):
            pw = 3 ** i
            # for each residue, shift
            for r in range(3):
                mask = (all_words % (3 * pw)) // pw == r
                idx = np.where(mask)[0]
                for d in range(1, 3):
                    shifted = idx + ((r + d) % 3 - r) * pw
                    gain[idx] += (cover[shifted] == 0).astype(np.int32)
        # tie-break by total coverage (spread)
        total = np.zeros(N, dtype=np.int32)
        for i in range(n):
            pw = 3 ** i
            for r in range(3):
                idx = np.where((all_words % (3 * pw)) // pw == r)[0]
                for d in range(1, 3):
                    shifted = idx + ((r + d) % 3 - r) * pw
                    total[idx] += cover[shifted]
        total += cover
        # pick max gain, tie-break max total, then random
        best = -1
        best_key = None
        for w in rng.sample(range(N), min(N, 20000)):
            g = gain[w]
            if g == 0:
                continue
            t = total[w]
            key = (g, t, rng.random())
            if best_key is None or key > best_key:
                best_key = key
                best = w
        if best < 0:
            # fallback: scan all
            best = int(np.argmax(gain))
        for nb in _neighbors(best, n):
            if cover[nb] == 0:
                covered += 1
            cover[nb] += 1
        words.append(_from_flat(best, n))
    return words

def _local_search(words, n, q, rng, deadline, safety=8.0):
    N = q ** n
    cover = _build_coverage(words, n, q)
    cur = len(words)
    while time.time() < deadline - safety:
        improved = False
        # 1-for-1 swaps: replace a codeword with an uncovered point
        uncov = np.where(cover == 0)[0]
        if len(uncov) > 0:
            # try random swaps
            for _ in range(200):
                if time.time() > deadline - safety:
                    break
                u = int(rng.choice(uncov))
                j = rng.randrange(len(words))
                # check: removing words[j] must not uncover any point
                # i.e., for all nb of words[j], cover[nb] >= 2 OR nb == u and we add u
                ok = True
                for nb in _neighbors(_to_flat(words[j], n), n):
                    if nb == u:
                        if cover[nb] < 1:
                            ok = False
                            break
                    else:
                        if cover[nb] < 2:
                            ok = False
                            break
                if ok:
                    # perform swap
                    for nb in _neighbors(_to_flat(words[j], n), n):
                        cover[nb] -= 1
                    for nb in _neighbors(u, n):
                        cover[nb] += 1
                    words[j] = _from_flat(u, n)
                    uncov = np.where(cover == 0)[0]
                    improved = True
                    break
        # 2-for-3 reductions
        if len(words) >= 3:
            for _ in range(300):
                if time.time() > deadline - safety:
                    break
                a, b, c = rng.sample(range(len(words)), 3)
                va = _to_flat(words[a], n)
                vb = _to_flat(words[b], n)
                vc = _to_flat(words[c], n)
                # points covered only by these three
                only = []
                for nb in set(list(_neighbors(va, n)) + list(_neighbors(vb, n)) + list(_neighbors(vc, n))):
                    if cover[nb] == 1:
                        only.append(nb)
                if len(only) == 0:
                    continue
                # need 2 codewords whose union covers `only`
                # candidate new codewords: uncovered points + points in only
                candidates = set(only) | set(np.where(cover == 0)[0].tolist())
                found = False
                cand_list = list(candidates)
                if len(cand_list) > 200:
                    cand_list = rng.sample(cand_list, 200)
                for x in cand_list:
                    covx = set(_neighbors(x, n))
                    rem = [p for p in only if p not in covx]
                    if not rem:
                        # x alone covers only; but we need 2 total, use x + something
                        # actually we remove 3 add 2, so we need exactly 2 new covering all `only`
                        # if x covers all only, second can be anything (but must be distinct & valid)
                        # pick second = an uncovered point or x's neighbor
                        y = int(rng.choice(np.where(cover == 0)[0])) if len(np.where(cover == 0)[0]) > 0 else (x + 1) % N
                        if y == x:
                            continue
                        # verify union covers only
                        covy = set(_neighbors(y, n))
                        if all(p in covx or p in covy for p in only):
                            # perform
                            for nb in _neighbors(va, n):
                                cover[nb] -= 1
                            for nb in _neighbors(vb, n):
                                cover[nb] -= 1
                            for nb in _neighbors(vc, n):
                                cover[nb] -= 1
                            for nb in _neighbors(x, n):
                                cover[nb] += 1
                            for nb in _neighbors(y, n):
                                cover[nb] += 1
                            words[a] = _from_flat(x, n)
                            words[b] = _from_flat(y, n)
                            del words[c]
                            cur = len(words)
                            improved = True
                            found = True
                            break
                    else:
                        # need y to cover `rem`
                        for y in cand_list:
                            if y == x:
                                continue
                            covy = set(_neighbors(y, n))
                            if all(p in covy for p in rem):
                                # verify union covers all only
                                if all(p in covx or p in covy for p in only):
                                    for nb in _neighbors(va, n):
                                        cover[nb] -= 1
                                    for nb in _neighbors(vb, n):
                                        cover[nb] -= 1
                                    for nb in _neighbors(vc, n):
                                        cover[nb] -= 1
                                    for nb in _neighbors(x, n):
                                        cover[nb] += 1
                                    for nb in _neighbors(y, n):
                                        cover[nb] += 1
                                    words[a] = _from_flat(x, n)
                                    words[b] = _from_flat(y, n)
                                    del words[c]
                                    cur = len(words)
                                    improved = True
                                    found = True
                                    break
                        if found:
                            break
                if found:
                    break
        if not improved:
            # try a few random perturbations to escape local optimum
            for _ in range(5):
                if time.time() > deadline - safety:
                    break
                j = rng.randrange(len(words))
                u = int(rng.choice(np.where(cover == 0)[0])) if len(np.where(cover == 0)[0]) > 0 else None
                if u is None:
                    continue
                ok = True
                for nb in _neighbors(_to_flat(words[j], n), n):
                    if nb != u and cover[nb] < 2:
                        ok = False
                        break
                if ok:
                    for nb in _neighbors(_to_flat(words[j], n), n):
                        cover[nb] -= 1
                    for nb in _neighbors(u, n):
                        cover[nb] += 1
                    words[j] = _from_flat(u, n)

def construct(q, n, radius):
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s
    rng = random.Random(12345)
    best = None
    N = q ** n

    # Initial greedy solution
    words = _greedy_init(n, q, rng, seed_words=None)
    cover = _build_coverage(words, n, q)
    assert np.all(cover >= 1), "greedy did not cover all"
    best = [list(w) for w in words]

    # Local search on the greedy solution
    _local_search(words, n, q, rng, deadline, safety=10.0)
    cover = _build_coverage(words, n, q)
    if np.all(cover >= 1) and len(words) < len(best):
        best = [list(w) for w in words]

    # Additional randomized restarts if time permits
    attempt = 0
    while time.time() < deadline - 15.0:
        attempt += 1
        # small random seed set to diversify
        seed_words = []
        for _ in range(min(3, n)):
            seed_words.append([rng.randrange(q) for _ in range(n)])
        words2 = _greedy_init(n, q, rng, seed_words=seed_words)
        cover2 = _build_coverage(words2, n, q)
        if np.all(cover2 >= 1):
            _local_search(words2, n, q, rng, deadline, safety=10.0)
            cover2 = _build_coverage(words2, n, q)
            if np.all(cover2 >= 1) and len(words2) < len(best):
                best = [list(w) for w in words2]
        if time.time() > deadline - 20.0:
            break

    # Final safety: ensure best is valid and distinct
    seen = set()
    valid = []
    for w in best:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            valid.append(list(w))
    # verify coverage
    cover = _build_coverage(valid, n, q)
    if not np.all(cover >= 1):
        # fallback to a known valid construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        valid = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    return valid

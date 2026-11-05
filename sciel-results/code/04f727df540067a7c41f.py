import numpy as np
import time

def construct():
    t0 = time.time()
    DEADLINE = 140.0

    def get_primes(n):
        s = [True]*(n+1)
        s[0]=s[1]=False
        for i in range(2, int(n**0.5)+1):
            if s[i]:
                s[i*i::i] = [False]*len(s[i*i::i])
        return [i for i in range(n+1) if s[i]]
    primes = get_primes(1000)
    npr = len(primes)
    MAX = 100000

    def score(c):
        below = np.nonzero(c[1:] < 2)[0]
        return MAX if below.size == 0 else int(below[0])

    def pos(r, p):
        start = r if r > 0 else p
        return np.arange(start, MAX+1, p)

    # ---- Greedy initialization ----
    def init(seed):
        rng = np.random.default_rng(seed)
        c = np.zeros(MAX+1, dtype=np.int32)
        res = np.zeros(npr, dtype=np.int64)
        for i, p in enumerate(primes):
            r = int(rng.integers(p))
            res[i] = r
            c[pos(r, p)] += 1
        S = score(c)
        for it in range(40):
            if time.time() - t0 > DEADLINE - 70: break
            improved = False
            for i, p in enumerate(primes):
                old_r = int(res[i])
                dec = pos(old_r, p)
                best_r = old_r
                best_S = S
                cands = [(S+1) % p, (S+2) % p, int(rng.integers(p))]
                for new_r in cands:
                    if new_r == old_r: continue
                    inc = pos(new_r, p)
                    c[dec] -= 1; c[inc] += 1
                    nS = score(c)
                    c[inc] -= 1; c[dec] += 1
                    if nS > best_S:
                        best_S = nS; best_r = new_r
                if best_r != old_r:
                    inc = pos(best_r, p)
                    c[dec] -= 1; c[inc] += 1
                    res[i] = best_r
                    S = best_S
                    improved = True
            if not improved:
                break
        return c, res, S

    # Run a few inits, keep best
    best_S = -1
    best_c = None
    best_res = None
    for seed in range(3):
        if time.time() - t0 > DEADLINE - 70: break
        c, res, S = init(seed)
        if S > best_S:
            best_S = S; best_c = c; best_res = res

    c = best_c
    res = best_res
    S = best_S
    rng = np.random.default_rng(123)

    # ---- Local search ----
    stagnant = 0
    while time.time() - t0 < DEADLINE - 30:
        improved = False
        for i in range(npr):
            if time.time() - t0 > DEADLINE - 30: break
            p = primes[i]
            old_r = int(res[i])
            dec = pos(old_r, p)
            best_r = old_r
            best_S_local = S
            cands = [(S+1) % p, (S+2) % p, int(rng.integers(p))]
            for new_r in cands:
                if new_r == old_r: continue
                inc = pos(new_r, p)
                c[dec] -= 1; c[inc] += 1
                nS = score(c)
                c[inc] -= 1; c[dec] += 1
                if nS > best_S_local:
                    best_S_local = nS; best_r = new_r
            if best_r != old_r:
                inc = pos(best_r, p)
                c[dec] -= 1; c[inc] += 1
                res[i] = best_r
                S = best_S_local
                improved = True
        if S > best_S:
            best_S = S; best_res = res.copy()
            stagnant = 0
        else:
            stagnant += 1
        if stagnant >= 150:
            # random kick
            for _ in range(10):
                i = int(rng.integers(npr))
                p = primes[i]
                old_r = int(res[i])
                new_r = int(rng.integers(p))
                if new_r == old_r: continue
                dec = pos(old_r, p); inc = pos(new_r, p)
                c[dec] -= 1; c[inc] += 1
                res[i] = new_r
            S = score(c)
            stagnant = 0
            if S > best_S:
                best_S = S; best_res = res.copy()

    # ---- Final polish: try all residues per prime (a couple passes) ----
    for _pass in range(3):
        if time.time() - t0 > DEADLINE - 5: break
        for i in range(npr):
            if time.time() - t0 > DEADLINE - 5: break
            p = primes[i]
            old_r = int(res[i])
            dec = pos(old_r, p)
            best_r = old_r
            best_S_local = S
            for new_r in range(p):
                if new_r == old_r: continue
                inc = pos(new_r, p)
                c[dec] -= 1; c[inc] += 1
                nS = score(c)
                c[inc] -= 1; c[dec] += 1
                if nS > best_S_local:
                    best_S_local = nS; best_r = new_r
            if best_r != old_r:
                inc = pos(best_r, p)
                c[dec] -= 1; c[inc] += 1
                res[i] = best_r
                S = best_S_local
        if S > best_S:
            best_S = S; best_res = res.copy()

    return [int(x) for x in best_res]

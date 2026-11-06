import time
import numpy as np

def _sieve(limit):
    is_p = bytearray([1]) * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i:limit + 1:i] = bytearray(len(is_p[i * i:limit + 1:i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

PRIMES = _sieve(1000)
N = len(PRIMES)

def _compute_L(count, L_max):
    seg = count[1:L_max + 1]
    bad = np.flatnonzero(seg < 2)
    if bad.size == 0:
        return L_max
    return int(bad[0])

def _count_array(residues, L_max):
    count = np.zeros(L_max + 1, dtype=np.int32)
    for p, r in zip(PRIMES, residues):
        start = p if r == 0 else r
        if start <= L_max:
            count[start::p] += 1
    return count

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    rng = np.random.default_rng(12345)

    best_res = [((i + 2) // 2) % p for i, p in enumerate(PRIMES)]
    best_L = 0
    L_max = 2000

    def eval_res(res):
        c = _count_array(res, L_max)
        return _compute_L(c, L_max)

    best_L = eval_res(best_res)

    def local_search(res, Lm, budget):
        nonlocal best_res, best_L
        res = list(res)
        count = _count_array(res, Lm)
        cur_L = _compute_L(count, Lm)
        if cur_L > best_L:
            best_L = cur_L
            best_res = res[:]
        stuck = 0
        primes = PRIMES
        n = N
        while time.time() < budget:
            if cur_L >= Lm:
                break
            x_star = cur_L + 1
            best_move_L = cur_L
            best_idx = -1
            best_nr = -1
            for idx in range(n):
                p = primes[idx]
                nr = x_star % p
                if nr == res[idx]:
                    continue
                old_r = res[idx]
                for r, d in ((old_r, -1), (nr, 1)):
                    start = p if r == 0 else r
                    if start <= Lm:
                        count[start::p] += d
                L = _compute_L(count, Lm)
                for r, d in ((old_r, 1), (nr, -1)):
                    start = p if r == 0 else r
                    if start <= Lm:
                        count[start::p] += d
                if L > best_move_L:
                    best_move_L = L
                    best_idx = idx
                    best_nr = nr
            if best_idx >= 0:
                p = primes[best_idx]
                old_r = res[best_idx]
                for r, d in ((old_r, -1), (best_nr, 1)):
                    start = p if r == 0 else r
                    if start <= Lm:
                        count[start::p] += d
                res[best_idx] = best_nr
                cur_L = best_move_L
                stuck = 0
                if cur_L > best_L:
                    best_L = cur_L
                    best_res = res[:]
            else:
                stuck += 1
                if stuck > 25:
                    for _ in range(4):
                        idx = int(rng.integers(0, n))
                        res[idx] = int(rng.integers(0, primes[idx]))
                    count = _count_array(res, Lm)
                    cur_L = _compute_L(count, Lm)
                    stuck = 0
                    if cur_L > best_L:
                        best_L = cur_L
                        best_res = res[:]
        return cur_L

    # Init 0: seed assignment
    local_search(best_res, L_max, deadline - 12)

    # If we saturated the cap, extend and continue from best
    if best_L >= L_max and time.time() < deadline - 25:
        L_max = 5000
        local_search(best_res, L_max, deadline - 12)

    # Additional diverse random inits if time remains
    for k in range(1, 6):
        if time.time() > deadline - 18:
            break
        res = [int(rng.integers(0, p)) for p in PRIMES]
        local_search(res, L_max, deadline - 18)

    return best_res

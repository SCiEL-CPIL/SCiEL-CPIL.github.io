import time
from math import log, gcd, exp
from itertools import combinations
import random
import numpy as np

def _score_raw(U):
    """Score an already-normalized sorted set (contains 0, len>=2)."""
    n = len(U)
    if n < 2:
        return -1.0
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    sc = len(np.unique(sums))
    dc = len(np.unique(diffs))
    if sc == 0:
        return -1.0
    ratio = dc / sc
    if ratio <= 1:
        return -1.0
    D = 2 * U[-1] + 1
    if D <= 1:
        return -1.0
    return 1.0 + log(ratio) / log(D)

def normalize(U):
    """Return sorted distinct nonnegative ints containing 0, divided by gcd."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U.sort()
    g = 0
    for x in U:
        g = gcd(g, x)
    if g > 1:
        U = sorted(set(x // g for x in U))
    return U

def score_set(U):
    U = normalize(list(U))
    if len(U) < 2:
        return -1.0
    return _score_raw(U)

def ap_set(ap_params):
    """
    Build a set from a list of AP parameters.
    ap_params: list of (start, diff, length)
    Returns sorted list of distinct nonnegative integers.
    """
    elems = set([0])
    for start, diff, length in ap_params:
        for i in range(length):
            v = start + i * diff
            if v >= 0:
                elems.add(v)
    return normalize(list(elems))

def ap_local_search(ap_params, deadline, max_val=200, T0=0.03):
    """
    Local search on AP parameters.
    ap_params: list of (start, diff, length)
    """
    cur = ap_set(ap_params)
    cur_score = _score_raw(cur)
    best_score = cur_score
    best_params = [tuple(p) for p in ap_params]
    T = T0
    no_improve = 0
    max_no_improve = 400
    
    while time.time() < deadline - 1 and no_improve < max_no_improve:
        r = random.random()
        new_params = [tuple(p) for p in ap_params]
        
        if r < 0.25:
            # Modify an existing AP
            idx = random.randint(0, len(new_params) - 1)
            start, diff, length = new_params[idx]
            mod_type = random.random()
            if mod_type < 0.4:
                # Change start
                new_start = max(0, start + random.randint(-5, 5))
                new_params[idx] = (new_start, diff, length)
            elif mod_type < 0.8:
                # Change diff
                new_diff = max(1, diff + random.randint(-3, 3))
                new_params[idx] = (start, new_diff, length)
            else:
                # Change length
                new_length = max(1, length + random.randint(-2, 2))
                new_params[idx] = (start, diff, new_length)
        elif r < 0.45:
            # Add a new AP
            if len(new_params) < 8:
                new_start = random.randint(0, 50)
                new_diff = random.randint(2, 20)
                new_length = random.randint(2, 6)
                new_params.append((new_start, new_diff, new_length))
        elif r < 0.60:
            # Remove an AP
            if len(new_params) > 2:
                idx = random.randint(0, len(new_params) - 1)
                new_params.pop(idx)
        elif r < 0.80:
            # Perturb all params slightly
            new_params = []
            for start, diff, length in new_params:
                ns = max(0, start + random.randint(-2, 2))
                nd = max(1, diff + random.randint(-1, 1))
                nl = max(1, length + random.randint(-1, 1))
                new_params.append((ns, nd, nl))
        else:
            # Replace a random AP entirely
            idx = random.randint(0, len(new_params) - 1)
            new_start = random.randint(0, 80)
            new_diff = random.randint(2, 30)
            new_length = random.randint(2, 8)
            new_params[idx] = (new_start, new_diff, new_length)
        
        # Validate and compute score
        try:
            new_set = ap_set(new_params)
            if len(new_set) < 2 or new_set[-1] > max_val:
                no_improve += 1
                continue
            new_score = _score_raw(new_set)
        except:
            no_improve += 1
            continue
        
        if new_score > cur_score:
            ap_params = new_params
            cur_score = new_score
            no_improve = 0
            if cur_score > best_score:
                best_score = cur_score
                best_params = [tuple(p) for p in ap_params]
        else:
            delta = cur_score - new_score
            if delta < 0 or random.random() < exp(-delta / max(T, 1e-9)):
                ap_params = new_params
                cur_score = new_score
            no_improve += 1
        T = max(T * 0.998, 1e-5)
    
    return best_score, best_params

def local_search(U, deadline, max_val=260, max_n=28, T0=0.02):
    """Simulated-annealing local search in normalized space."""
    U = normalize(list(U))
    cur_score = _score_raw(U)
    best_score = cur_score
    best_set = list(U)
    T = T0
    no_improve = 0
    max_no_improve = 450
    U_set = set(U)
    while time.time() < deadline - 1 and no_improve < max_no_improve:
        r = random.random()
        n = len(U)
        new_U = None
        if r < 0.35 and n < max_n:
            v = random.randint(1, max_val)
            if v not in U_set:
                new_U = normalize(U + [v])
        elif r < 0.45 and n > 2:
            idx = random.randint(1, n - 1)
            new_U = normalize([x for j, x in enumerate(U) if j != idx])
        elif r < 0.62:
            idx = random.randint(1, n - 1)
            v = random.randint(1, max_val)
            if v not in U_set:
                cand = list(U)
                cand[idx] = v
                new_U = normalize(cand)
        elif r < 0.78 and n > 2:
            i = random.randint(1, n - 1)
            j = random.randint(1, n - 1)
            if i != j:
                cand = list(U)
                cand[i], cand[j] = cand[j], cand[i]
                new_U = normalize(cand)
        else:
            idx = random.randint(1, n - 1)
            v = U[idx] + random.randint(-15, 15)
            if v >= 1 and v not in U_set:
                cand = list(U)
                cand[idx] = v
                new_U = normalize(cand)
        if new_U is None or len(new_U) < 2:
            no_improve += 1
            continue
        new_score = _score_raw(new_U)
        if new_score > cur_score:
            U = new_U
            U_set = set(U)
            cur_score = new_score
            no_improve = 0
            if cur_score > best_score:
                best_score = cur_score
                best_set = list(U)
        else:
            delta = cur_score - new_score
            if delta < 0 or random.random() < exp(-delta / max(T, 1e-9)):
                U = new_U
                U_set = set(U)
                cur_score = new_score
            no_improve += 1
        T = max(T * 0.9985, 1e-5)
    return best_score, best_set

def greedy_build(deadline, maxv=80, maxn=14):
    """Constructively add the element that most improves the score."""
    U = [0]
    while time.time() < deadline - 4:
        cur = score_set(U)
        best_v = None
        best_sc = cur
        for v in range(1, maxv + 1):
            if v in U:
                continue
            sc = _score_raw(normalize(U + [v]))
            if sc > best_sc:
                best_sc = sc
                best_v = v
        if best_v is None or len(U) >= maxn:
            break
        U = normalize(U + [best_v])
    return U

def construct():
    start = time.time()
    deadline = start + 140  # 10s safety margin before 150s kill

    best_set = [0, 1, 3, 4]
    best_score = score_set(best_set)

    # --- Structured seeds ---
    structured = [
        [0,1,3,4],
        [0,1,3,4,9,10,12,13],
        [0,1,4,9,11],
        [0,1,4,10,20,35],
        [0,1,2,4,8,16],
        [0,1,3,7,15,31],
        [0,1,5,13,25,41],
        [0,1,3,9,27,81],
        [0,1,4,10,18,28],
        [0,1,2,5,13,34],
        [0,1,2,4,7,9],
        [0,1,3,6,7,11],
    ]
    for s in structured:
        sc = score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = normalize(list(s))

    # --- AP-based seeds ---
    ap_seeds = [
        [(1, 2, 4), (10, 3, 3)],
        [(1, 3, 4), (15, 2, 3)],
        [(2, 4, 3), (20, 3, 4)],
        [(1, 2, 3), (8, 5, 3), (30, 2, 2)],
        [(3, 4, 3), (25, 3, 4)],
        [(1, 5, 3), (20, 4, 3)],
        [(2, 3, 4), (18, 5, 3)],
        [(1, 2, 5), (15, 3, 4)],
        [(4, 3, 3), (22, 2, 4)],
        [(1, 4, 3), (12, 6, 3)],
    ]
    for ap in ap_seeds:
        s = ap_set(ap)
        if len(s) >= 2:
            sc = score_set(s)
            if sc > best_score:
                best_score = sc
                best_set = s

    # --- Random + greedy seeds ---
    seeds = [list(best_set)]
    for _ in range(30):
        n = random.randint(3, 8)
        vals = random.sample(range(1, 70), n)
        s = normalize([0] + vals)
        seeds.append(s)
    for mv in (50, 80, 120):
        if time.time() < deadline - 100:
            g = greedy_build(deadline, maxv=mv, maxn=12)
            if len(g) >= 2:
                seeds.append(g)
    for s in seeds:
        sc = score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = list(s)

    # --- Exhaustive small-n (normalized) ---
    def exhaustive(n, maxv):
        bs = best_score
        bc = best_set
        for comb in combinations(range(1, maxv + 1), n - 1):
            U = normalize([0] + list(comb))
            if len(U) != n:
                continue
            sc = _score_raw(U)
            if sc > bs:
                bs = sc
                bc = list(U)
        return bs, bc

    if time.time() < deadline - 75:
        bs, bc = exhaustive(3, 90)
        if bs > best_score: best_score, best_set = bs, bc
    if time.time() < deadline - 58:
        bs, bc = exhaustive(4, 55)
        if bs > best_score: best_score, best_set = bs, bc
    if time.time() < deadline - 42:
        bs, bc = exhaustive(5, 38)
        if bs > best_score: best_score, best_set = bs, bc

    # --- AP-based local search restarts ---
    ap_configs = [
        [(1, 2, 4), (10, 3, 3)],
        [(1, 3, 4), (15, 2, 3)],
        [(2, 4, 3), (20, 3, 4)],
        [(1, 2, 3), (8, 5, 3), (30, 2, 2)],
        [(3, 4, 3), (25, 3, 4)],
        [(1, 5, 3), (20, 4, 3)],
        [(2, 3, 4), (18, 5, 3)],
        [(1, 2, 5), (15, 3, 4)],
        [(4, 3, 3), (22, 2, 4)],
        [(1, 4, 3), (12, 6, 3)],
        [(2, 2, 4), (14, 4, 3), (40, 3, 2)],
        [(1, 3, 3), (10, 2, 4), (35, 5, 2)],
    ]
    
    if time.time() < deadline - 30:
        for ap in ap_configs:
            if time.time() > deadline - 30:
                break
            sc, params = ap_local_search(ap, deadline, max_val=200)
            if sc > best_score:
                best_score = sc
                best_set = ap_set(params)

    # --- Local search restarts ---
    restart_seeds = [list(best_set)] + seeds[:10]
    attempts = 0
    while time.time() < deadline - 22:
        base = restart_seeds[attempts % len(restart_seeds)]
        sc, s = local_search(base, deadline, max_val=260, max_n=28,
                             T0=random.uniform(0.01, 0.03))
        if sc > best_score:
            best_score = sc
            best_set = list(s)
            if len(restart_seeds) < 16:
                restart_seeds.append(list(s))
            else:
                restart_seeds[-1] = list(s)
        attempts += 1
    
    # --- Additional targeted search with structured patterns ---
    if time.time() < deadline - 5:
        pattern_seeds = [
            [0, 1, 3, 4, 9, 10, 12, 13, 27, 28, 30, 31],
            [0, 1, 4, 9, 10, 13, 18, 23, 28, 33],
            [0, 1, 5, 6, 11, 12, 16, 17, 22, 23],
            [0, 1, 2, 4, 8, 16, 32, 64],
            [0, 1, 3, 7, 15, 31, 63, 127],
        ]
        for ps in pattern_seeds:
            if time.time() > deadline - 5:
                break
            s = normalize(ps)
            sc = score_set(s)
            if sc > best_score:
                best_score = sc
                best_set = s
            sc, s2 = local_search(s, deadline, max_val=200, max_n=24, T0=0.015)
            if sc > best_score:
                best_score = sc
                best_set = list(s2)

    # --- Final high-precision polish (bounded) ---
    if time.time() < deadline - 3:
        cur = list(best_set)
        cur_score = _score_raw(cur)
        improved = True
        it = 0
        cur_set = set(cur)
        while improved and time.time() < deadline - 2 and it < 150:
            improved = False
            it += 1
            n = len(cur)
            for i in range(1, n):
                if time.time() > deadline - 2:
                    break
                old = cur[i]
                for delta in range(-40, 41):
                    if delta == 0:
                        continue
                    v = old + delta
                    if v < 1 or v in cur_set:
                        continue
                    cand = list(cur)
                    cand[i] = v
                    newU = normalize(cand)
                    if len(newU) != n:
                        continue
                    sc = _score_raw(newU)
                    if sc > cur_score:
                        cur, cur_score, improved = newU, sc, True
                        cur_set = set(cur)
                        break
                if improved:
                    break
            if improved:
                continue
            if n < 26:
                hi = min(160, cur[-1] + 40)
                for v in range(1, hi + 1):
                    if v in cur_set:
                        continue
                    newU = normalize(cur + [v])
                    sc = _score_raw(newU)
                    if sc > cur_score:
                        cur, cur_score, improved = newU, sc, True
                        cur_set = set(cur)
                        break
            if improved:
                continue
            for i in range(1, n):
                newU = normalize([x for j, x in enumerate(cur) if j != i])
                if len(newU) < 2:
                    continue
                sc = _score_raw(newU)
                if sc > cur_score:
                    cur, cur_score, improved = newU, sc, True
                    cur_set = set(cur)
                    break
        if cur_score > best_score:
            best_score = cur_score
            best_set = cur

    return normalize(list(best_set))

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))

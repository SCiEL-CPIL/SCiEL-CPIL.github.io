import time
from math import log, gcd
from itertools import combinations, product
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

def exhaustive_small_sets(max_n=6, max_val=60, deadline=None):
    """Exhaustively search small sets for best score."""
    best_score = -1
    best_set = [0, 1, 3, 4]
    
    for n in range(3, max_n + 1):
        for comb in combinations(range(1, max_val + 1), n - 1):
            if deadline and time.time() > deadline:
                return best_score, best_set
            U = [0] + list(comb)
            s = _score_raw(normalize(U))
            if s > best_score:
                best_score = s
                best_set = normalize(U)
    return best_score, best_set

def digit_based_set(bases, digit_sets):
    """Construct a set from mixed-radix digit expansion."""
    if not bases or not digit_sets:
        return [0]
    elements = [0]
    for base, digits in zip(bases, digit_sets):
        if not digits:
            continue
        new_elements = []
        for elem in elements:
            for d in digits:
                new_elements.append(elem * base + d)
        elements = new_elements
    if 0 not in elements:
        elements.append(0)
    return normalize(elements)

def geometric_seed(ratio, n):
    """Generate geometric-like seed."""
    U = [0]
    x = 1
    for i in range(n - 1):
        U.append(x)
        x = int(x * ratio)
    return normalize(U)

def structured_seeds():
    """Return a list of structured seed sets."""
    seeds = []
    # Basic structured sets
    basic = [
        [0, 1, 3, 4],
        [0, 1, 3, 4, 9, 10, 12, 13],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 20, 35],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 5, 13, 25, 41],
        [0, 1, 3, 9, 27, 81],
        [0, 1, 4, 10, 18, 28],
        [0, 1, 2, 5, 13, 34],
        [0, 1, 2, 4, 7, 9],
        [0, 1, 3, 6, 7, 11],
        [0, 1, 3, 4, 9, 10, 12, 13, 27, 28, 30, 31],
        [0, 1, 4, 9, 10, 13, 18, 23, 28, 33],
        [0, 1, 3, 7],
        [0, 1, 3, 8],
        [0, 1, 3, 9],
        [0, 1, 4, 16],
        [0, 1, 2, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 8, 16],
        [0, 1, 4, 9, 16],
        [0, 1, 2, 4, 8],
        [0, 1, 2, 4, 9],
        [0, 1, 3, 6, 10],
    ]
    seeds.extend([normalize(s) for s in basic])
    
    # Geometric seeds
    for ratio in [1.5, 1.7, 2.0, 2.2, 2.5, 3.0]:
        for n in range(4, 12):
            s = geometric_seed(ratio, n)
            if s[-1] <= 500:
                seeds.append(s)
    
    # Digit-based seeds
    digit_configs = [
        ([3, 4], [{0,1,2}, {0,1,2,3}]),
        ([4, 3], [{0,1,2,3}, {0,1,2}]),
        ([5, 4], [{0,1,2,3,4}, {0,1,2,3}]),
        ([3, 5], [{0,1,2}, {0,1,2,3,4}]),
        ([4, 5], [{0,1,2,3}, {0,1,2,3,4}]),
        ([5, 3], [{0,1,2,3,4}, {0,1,2}]),
        ([3, 4, 5], [{0,1,2}, {0,1,2,3}, {0,1,2,3,4}]),
        ([4, 3, 5], [{0,1,2,3}, {0,1,2}, {0,1,2,3,4}]),
        ([5, 4, 3], [{0,1,2,3,4}, {0,1,2,3}, {0,1,2}]),
        ([3, 5, 4], [{0,1,2}, {0,1,2,3,4}, {0,1,2,3}]),
        ([6, 4], [{0,1,2,3}, {0,1,2,3}]),
        ([4, 6], [{0,1,2,3}, {0,1,2,3,4,5}]),
        ([5, 5], [{0,1,2,3}, {0,1,2,3,4}]),
        ([7, 4], [{0,1,2,3,4}, {0,1,2,3}]),
        ([4, 7], [{0,1,2,3}, {0,1,2,3,4,5,6}]),
        ([3, 3, 3], [{0,1,2}, {0,1,2}, {0,1,2}]),
        ([4, 4, 3], [{0,1,2,3}, {0,1,2,3}, {0,1,2}]),
        ([5, 3, 3], [{0,1,2,3,4}, {0,1,2}, {0,1,2}]),
    ]
    for bases, dsets in digit_configs:
        s = digit_based_set(bases, [set(ds) for ds in dsets])
        if len(s) >= 2 and s[-1] <= 500:
            seeds.append(s)
    
    return seeds

def local_search(U, deadline, max_val=300, max_n=30, T0=0.02):
    """Simulated-annealing local search in normalized space."""
    U = normalize(list(U))
    cur_score = _score_raw(U)
    best_score = cur_score
    best_set = list(U)
    T = T0
    no_improve = 0
    max_no_improve = 800
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
        elif r < 0.60:
            idx = random.randint(1, n - 1)
            v = random.randint(1, max_val)
            if v not in U_set:
                cand = list(U)
                cand[idx] = v
                new_U = normalize(cand)
        elif r < 0.75 and n > 2:
            i = random.randint(1, n - 1)
            j = random.randint(1, n - 1)
            if i != j:
                cand = list(U)
                cand[i], cand[j] = cand[j], cand[i]
                new_U = normalize(cand)
        elif r < 0.85:
            idx = random.randint(1, n - 1)
            delta = random.randint(-30, 30)
            v = U[idx] + delta
            if v >= 1 and v not in U_set:
                cand = list(U)
                cand[idx] = v
                new_U = normalize(cand)
        else:
            # Add two elements at once
            if n < max_n - 1:
                v1 = random.randint(1, max_val)
                v2 = random.randint(1, max_val)
                if v1 not in U_set and v2 not in U_set and v1 != v2:
                    new_U = normalize(U + [v1, v2])
        
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
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-9)):
                U = new_U
                U_set = set(U)
                cur_score = new_score
            no_improve += 1
        T = max(T * 0.995, 1e-5)
    
    return best_score, best_set

def polish_pass(U, deadline):
    """High-precision polish: try small local moves exhaustively."""
    cur = normalize(list(U))
    cur_score = _score_raw(cur)
    improved = True
    it = 0
    
    while improved and time.time() < deadline - 2 and it < 300:
        improved = False
        it += 1
        n = len(cur)
        cur_set = set(cur)
        
        # Try replacing each element with nearby values
        for i in range(1, n):
            if time.time() > deadline - 2:
                break
            old = cur[i]
            for delta in range(-60, 61):
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
        
        # Try adding
        if n < 30:
            hi = min(300, cur[-1] + 80)
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
        
        # Try removing
        for i in range(1, n):
            newU = normalize([x for j, x in enumerate(cur) if j != i])
            if len(newU) < 2:
                continue
            sc = _score_raw(newU)
            if sc > cur_score:
                cur, cur_score, improved = newU, sc, True
                cur_set = set(cur)
                break
    
    return cur_score, cur

def construct():
    start = time.time()
    deadline = start + 145  # 5s safety margin before 150s kill
    
    best_set = [0, 1, 3, 4]
    best_score = score_set(best_set)
    
    # --- Phase 1: Exhaustive search for small sets ---
    exh_deadline = start + 25
    exh_score, exh_set = exhaustive_small_sets(max_n=6, max_val=55, deadline=exh_deadline)
    if exh_score > best_score:
        best_score = exh_score
        best_set = list(exh_set)
    
    # --- Phase 2: Evaluate structured seeds ---
    seeds = structured_seeds()
    for s in seeds:
        if time.time() > start + 40:
            break
        sc = score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = list(s)
    
    # --- Phase 3: Local search from multiple restarts ---
    restart_seeds = [list(best_set)]
    # Add more diverse seeds
    for s in seeds[:10]:
        restart_seeds.append(list(s))
    
    # Random seeds
    for _ in range(10):
        n = random.randint(4, 12)
        vals = random.sample(range(1, 200), n)
        s = normalize([0] + vals)
        restart_seeds.append(s)
    
    attempts = 0
    max_attempts = 15
    while time.time() < deadline - 30 and attempts < max_attempts:
        base = restart_seeds[attempts % len(restart_seeds)]
        sc, s = local_search(base, deadline, max_val=300, max_n=30,
                             T0=random.uniform(0.015, 0.04))
        if sc > best_score:
            best_score = sc
            best_set = list(s)
            if len(restart_seeds) < 20:
                restart_seeds.append(list(s))
            else:
                restart_seeds[-1] = list(s)
        attempts += 1
    
    # --- Phase 4: Final polish ---
    if time.time() < deadline - 3:
        ps, pset = polish_pass(best_set, deadline)
        if ps > best_score:
            best_score = ps
            best_set = list(pset)
    
    return normalize(list(best_set))

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))

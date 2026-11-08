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

def _score_fast(U):
    """Fast approximate score using sampling for large sets."""
    n = len(U)
    if n < 2:
        return -1.0
    if n <= 20:
        return _score_raw(U)
    # Sample-based approximation for large sets
    U_arr = np.array(U, dtype=np.int64)
    sample_size = min(n, 30)
    idx = np.random.choice(n, sample_size, replace=False)
    Us = U_arr[idx]
    sums = Us[:, None] + Us[None, :]
    diffs = Us[:, None] - Us[None, :]
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

def digit_based_set(bases, digit_sets):
    """
    Construct a set from mixed-radix digit expansion.
    bases: list of base values [b0, b1, ..., bk]
    digit_sets: list of sets of digits for each position
    Returns sorted list of distinct nonnegative integers.
    """
    if not bases or not digit_sets:
        return [0]
    if len(bases) != len(digit_sets):
        raise ValueError("bases and digit_sets must have same length")
    
    elements = [0]
    for pos, (base, digits) in enumerate(zip(bases, digit_sets)):
        if not digits:
            continue
        # Multiply existing elements by base and add each digit
        new_elements = []
        for elem in elements:
            for d in digits:
                new_elements.append(elem * base + d)
        elements = new_elements
    
    # Add 0 if not present (should be there from start)
    if 0 not in elements:
        elements.append(0)
    
    return normalize(elements)

def digit_based_set_fast(bases, digit_sets, max_size=500):
    """Fast version that stops if set gets too large."""
    if not bases or not digit_sets:
        return [0]
    if len(bases) != len(digit_sets):
        raise ValueError("bases and digit_sets must have same length")
    
    elements = [0]
    for pos, (base, digits) in enumerate(zip(bases, digit_sets)):
        if not digits:
            continue
        new_elements = []
        for elem in elements:
            for d in digits:
                new_elements.append(elem * base + d)
        elements = new_elements
        if len(elements) > max_size:
            return normalize(elements[:max_size])
    
    if 0 not in elements:
        elements.append(0)
    
    return normalize(elements)

def digit_local_search(bases, digit_sets, deadline, max_val=300, T0=0.02):
    """
    Local search that perturbs digit choices in the mixed-radix representation.
    """
    cur_set = digit_based_set(bases, digit_sets)
    cur_score = _score_raw(cur_set)
    best_score = cur_score
    best_bases = list(bases)
    best_digit_sets = [set(ds) for ds in digit_sets]
    T = T0
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline - 1 and no_improve < max_no_improve:
        r = random.random()
        new_bases = list(bases)
        new_digit_sets = [set(ds) for ds in digit_sets]
        
        if r < 0.25:
            # Perturb a base
            pos = random.randint(0, len(bases) - 1)
            delta = random.randint(-3, 3)
            new_bases[pos] = max(2, bases[pos] + delta)
        elif r < 0.45:
            # Add a digit to a position
            pos = random.randint(0, len(digit_sets) - 1)
            base = bases[pos]
            new_digit = random.randint(0, base - 1)
            new_digit_sets[pos].add(new_digit)
        elif r < 0.60:
            # Remove a digit from a position (keep at least 2 digits)
            pos = random.randint(0, len(digit_sets) - 1)
            if len(digit_sets[pos]) > 2:
                digits_list = list(digit_sets[pos])
                rem = random.choice(digits_list)
                new_digit_sets[pos].discard(rem)
        elif r < 0.70:
            # Add a new position
            if len(bases) < 6:
                new_base = random.randint(3, 12)
                new_digits = set(random.sample(range(new_base), min(3, new_base)))
                new_bases.append(new_base)
                new_digit_sets.append(new_digits)
        elif r < 0.85:
            # Remove a position (keep at least 2 positions)
            if len(bases) > 2:
                pos = random.randint(1, len(bases) - 1)
                new_bases.pop(pos)
                new_digit_sets.pop(pos)
        else:
            # Swap digits between positions
            if len(digit_sets) >= 2:
                pos1 = random.randint(0, len(digit_sets) - 1)
                pos2 = random.randint(0, len(digit_sets) - 1)
                if pos1 != pos2 and len(digit_sets[pos1]) > 0 and len(digit_sets[pos2]) > 0:
                    d1 = random.choice(list(digit_sets[pos1]))
                    d2 = random.choice(list(digit_sets[pos2]))
                    new_digit_sets[pos1].discard(d1)
                    new_digit_sets[pos2].discard(d2)
                    new_digit_sets[pos1].add(d2)
                    new_digit_sets[pos2].add(d1)
        
        # Validate and compute score
        try:
            new_set = digit_based_set(new_bases, new_digit_sets)
            if len(new_set) < 2 or new_set[-1] > max_val:
                no_improve += 1
                continue
            new_score = _score_raw(new_set)
        except:
            no_improve += 1
            continue
        
        if new_score > cur_score:
            bases = new_bases
            digit_sets = [set(ds) for ds in new_digit_sets]
            cur_score = new_score
            no_improve = 0
            if cur_score > best_score:
                best_score = cur_score
                best_bases = list(bases)
                best_digit_sets = [set(ds) for ds in digit_sets]
        else:
            delta = cur_score - new_score
            if delta < 0 or random.random() < exp(-delta / max(T, 1e-9)):
                bases = new_bases
                digit_sets = [set(ds) for ds in new_digit_sets]
                cur_score = new_score
            no_improve += 1
        T = max(T * 0.997, 1e-5)
    
    return best_score, best_bases, best_digit_sets

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

def local_search(U, deadline, max_val=300, max_n=30, T0=0.02):
    """Simulated-annealing local search in normalized space."""
    U = normalize(list(U))
    cur_score = _score_raw(U)
    best_score = cur_score
    best_set = list(U)
    T = T0
    no_improve = 0
    max_no_improve = 600
    U_set = set(U)
    while time.time() < deadline - 1 and no_improve < max_no_improve:
        r = random.random()
        n = len(U)
        new_U = None
        if r < 0.30 and n < max_n:
            v = random.randint(1, max_val)
            if v not in U_set:
                new_U = normalize(U + [v])
        elif r < 0.40 and n > 2:
            idx = random.randint(1, n - 1)
            new_U = normalize([x for j, x in enumerate(U) if j != idx])
        elif r < 0.55:
            idx = random.randint(1, n - 1)
            v = random.randint(1, max_val)
            if v not in U_set:
                cand = list(U)
                cand[idx] = v
                new_U = normalize(cand)
        elif r < 0.70 and n > 2:
            i = random.randint(1, n - 1)
            j = random.randint(1, n - 1)
            if i != j:
                cand = list(U)
                cand[i], cand[j] = cand[j], cand[i]
                new_U = normalize(cand)
        elif r < 0.85:
            idx = random.randint(1, n - 1)
            v = U[idx] + random.randint(-20, 20)
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
            if delta < 0 or random.random() < exp(-delta / max(T, 1e-9)):
                U = new_U
                U_set = set(U)
                cur_score = new_score
            no_improve += 1
        T = max(T * 0.997, 1e-5)
    return best_score, best_set

def construct():
    start = time.time()
    deadline = start + 145  # 5s safety margin before 150s kill

    best_set = [0, 1, 3, 4]
    best_score = score_set(best_set)

    # --- Exhaustive search for small sets (from inspiration 1) ---
    # This finds high-asymmetry ratio sets that outperform geometric seeds
    exhaustive_deadline = start + 30  # 30 seconds for exhaustive search
    
    # Search space: n in [3, 7], max_val in [10, 80]
    # We include 0, so we pick n-1 elements from 1..max_val
    max_val_limit = 80
    max_n_limit = 7
    
    for n in range(3, max_n_limit + 1):
        if time.time() > exhaustive_deadline:
            break
        # Max val should be at least n-1
        range_vals = range(1, max_val_limit + 1)
        
        # Combinations of n-1 elements
        for comb in combinations(range_vals, n - 1):
            if time.time() > exhaustive_deadline:
                break
            U = [0] + list(comb)
            # Quick check: if max(U) is too large for n, skip?
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = normalize(list(U))
    
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
        [0,1,3,4,9,10,12,13,27,28,30,31],
        [0,1,4,9,10,13,18,23,28,33],
        [0,1,3,7],
        [0,1,3,8],
        [0,1,3,9],
        [0,1,4,16],
        [0,1,2,7],
        [0,1,3,7,15],
        [0,1,3,8,16],
        [0,1,4,9,16],
        [0,1,2,4,8],
        [0,1,2,4,9],
        [0,1,3,6,10],
    ]
    for s in structured:
        sc = score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = normalize(list(s))

    # --- Digit-based seeds with more configurations ---
    digit_seeds = []
    # Various base/digit combinations
    for base in [3, 4, 5, 6, 7, 8, 9, 10, 12]:
        for ndig in [2, 3, 4]:
            digits = set(random.sample(range(base), min(ndig, base)))
            if 0 not in digits:
                digits.add(0)
            s = digit_based_set([base]*2, [digits, set(range(base))])
            if len(s) >= 2 and s[-1] <= 300:
                digit_seeds.append(s)
    
    # Mixed radix examples - expanded
    mixed_configs = [
        ([3, 4], [{0,1,2}, {0,1,2,3}]),
        ([4, 3], [{0,1,2,3}, {0,1,2}]),
        ([5, 4], [{0,1,2,3,4}, {0,1,2,3}]),
        ([3, 5], [{0,1,2}, {0,1,2,3,4}]),
        ([4, 5, 3], [{0,1,2,3}, {0,1,2,3,4}, {0,1,2}]),
        ([5, 3, 4], [{0,1,2,3,4}, {0,1,2}, {0,1,2,3}]),
        ([3, 4, 5], [{0,1,2}, {0,1,2,3}, {0,1,2,3,4}]),
        ([4, 3, 5], [{0,1,2,3}, {0,1,2}, {0,1,2,3,4}]),
        ([5, 4, 3], [{0,1,2,3,4}, {0,1,2,3}, {0,1,2}]),
        ([3, 5, 4], [{0,1,2}, {0,1,2,3,4}, {0,1,2,3}]),
        ([6, 4], [{0,1,2,3}, {0,1,2,3}]),
        ([4, 6], [{0,1,2,3}, {0,1,2,3,4,5}]),
        ([5, 5], [{0,1,2,3}, {0,1,2,3,4}]),
        ([7, 4], [{0,1,2,3,4}, {0,1,2,3}]),
        ([4, 7], [{0,1,2,3}, {0,1,2,3,4,5,6}]),
    ]
    for bases, dsets in mixed_configs:
        s = digit_based_set(bases, [set(ds) for ds in dsets])
        if len(s) >= 2 and s[-1] <= 300:
            digit_seeds.append(s)
    
    for s in digit_seeds:
        sc = score_set(s)
        if sc > best_score:
            best_score = sc
            best_set = normalize(list(s))

    # --- Random seeds ---
    seeds = [list(best_set)]
    for _ in range(20):
        n = random.randint(3, 10)
        vals = random.sample(range(1, 100), n)
        s = normalize([0] + vals)
        seeds.append(s)

    # --- Digit-based local search - primary strategy ---
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
    ]
    
    # Run digit-based local search with time budget
    time_per_config = max(2, (deadline - 50 - time.time()) / len(digit_configs))
    for bases, dsets in digit_configs:
        config_deadline = min(deadline - 50, time.time() + time_per_config)
        if time.time() > config_deadline:
            continue
        sc, b_bases, b_dsets = digit_local_search(bases, [set(ds) for ds in dsets], config_deadline, max_val=300)
        if sc > best_score:
            best_score = sc
            best_set = digit_based_set(b_bases, b_dsets)

    # --- Local search restarts from best and seeds ---
    restart_seeds = [list(best_set)] + seeds[:8]
    attempts = 0
    max_attempts = 12
    while time.time() < deadline - 25 and attempts < max_attempts:
        base = restart_seeds[attempts % len(restart_seeds)]
        sc, s = local_search(base, deadline, max_val=300, max_n=30,
                             T0=random.uniform(0.015, 0.035))
        if sc > best_score:
            best_score = sc
            best_set = list(s)
            if len(restart_seeds) < 14:
                restart_seeds.append(list(s))
            else:
                restart_seeds[-1] = list(s)
        attempts += 1
    
    # --- Final high-precision polish ---
    if time.time() < deadline - 3:
        cur = list(best_set)
        cur_score = _score_raw(cur)
        improved = True
        it = 0
        cur_set = set(cur)
        while improved and time.time() < deadline - 2 and it < 200:
            improved = False
            it += 1
            n = len(cur)
            # replace with small delta window
            for i in range(1, n):
                if time.time() > deadline - 2:
                    break
                old = cur[i]
                for delta in range(-50, 51):
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
            # add
            if n < 28:
                hi = min(200, cur[-1] + 50)
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
            # remove
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

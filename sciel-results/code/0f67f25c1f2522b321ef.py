import time
import math
import random
import numpy as np

_score_cache = {}

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    key = tuple(U)
    if key in _score_cache:
        return _score_cache[key]
    U_arr = np.array(U)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    sum_set_size = len(np.unique(sums))
    diff_set_size = len(np.unique(diffs))
    ratio = diff_set_size / sum_set_size
    if ratio <= 0:
        result = 0
    else:
        log_ratio = math.log(ratio)
        log_denom = math.log(2 * maxU + 1)
        if log_denom <= 0:
            result = 0
        else:
            result = 1 + log_ratio / log_denom
    _score_cache[key] = result
    return result

def local_optimize(U, deadline, max_iters=300):
    """Local search: add / remove / adjust elements to improve score."""
    U = sorted(set(U))
    current_score = compute_score(U)
    best_U = U[:]
    best_score = current_score

    for it in range(max_iters):
        if time.time() > deadline:
            break

        improved = False
        n = len(U)
        maxU = U[-1] if U else 1

        # 1) Add a new element
        if n < 20:
            candidates = set()
            for u in U:
                for d in [-3, -2, -1, 1, 2, 3, 4, 5, 7, 11, 13]:
                    v = u + d
                    if v >= 0 and v not in U:
                        candidates.add(v)
                for f in [2, 3, 5]:
                    v = u * f
                    if v not in U and v <= maxU * 20:
                        candidates.add(v)
            for factor in [2, 3, 4, 5, 7, 10, 13, 17, 21]:
                for adj in [-3, -2, -1, 0, 1, 2, 3]:
                    v = maxU * factor + adj
                    if v >= 1 and v not in U:
                        candidates.add(v)
            # Random jumps scaled to current range
            for _ in range(6):
                candidates.add(random.randint(1, maxU * 12 + 50))

            for cand in sorted(candidates):
                if cand < 0 or cand in U:
                    continue
                new_U = sorted(U + [cand])
                s = compute_score(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    improved = True
                    break

        # 2) Remove an element (keep at least 2, keep 0)
        if n > 2:
            for i in range(1, n):
                new_U = U[:i] + U[i+1:]
                s = compute_score(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    improved = True
                    break

        # 3) Adjust a single element
        if n > 1:
            for i in range(1, n):
                if improved:
                    break
                deltas = [-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 9, 12, 15, 20]
                for delta in deltas:
                    new_val = U[i] + delta
                    if new_val < 0 or new_val in U:
                        continue
                    new_U = U[:i] + [new_val] + U[i+1:]
                    new_U = sorted(set(new_U))
                    if len(new_U) != n:
                        continue
                    s = compute_score(new_U)
                    if s > best_score:
                        best_score = s
                        best_U = new_U
                        U = new_U
                        improved = True
                        break

        # 4) Two-swap: change two non-zero elements simultaneously
        if n >= 3 and not improved:
            for i in range(1, n):
                if improved:
                    break
                for j in range(i+1, n):
                    if improved:
                        break
                    di = random.randint(-5, 5)
                    dj = random.randint(-5, 5)
                    if di == 0 and dj == 0:
                        continue
                    ni = U[i] + di
                    nj = U[j] + dj
                    if ni < 0 or nj < 0:
                        continue
                    if ni in U or nj in U:
                        continue
                    if ni == nj:
                        continue
                    new_U = U[:]
                    new_U[i] = ni
                    new_U[j] = nj
                    new_U = sorted(set(new_U))
                    if len(new_U) != n:
                        continue
                    s = compute_score(new_U)
                    if s > best_score:
                        best_score = s
                        best_U = new_U
                        U = new_U
                        improved = True

        # 5) Periodic random restart nudge to escape plateaus
        if not improved and it % 50 == 49:
            if n > 2:
                idx = random.randint(1, n-1)
                new_val = random.randint(1, maxU * 3 + 30)
                if new_val not in U:
                    new_U = U[:]
                    new_U[idx] = new_val
                    new_U = sorted(set(new_U))
                    if len(new_U) == n:
                        s = compute_score(new_U)
                        if s > best_score:
                            best_score = s
                            best_U = new_U
                            U = new_U
                            improved = True

        if not improved:
            break

    return best_U, best_score

def construct():
    start = time.time()
    deadline = start + 140  # safety margin under 150s

    best_score = 0
    best_U = [0, 1, 3]
    s = compute_score([0, 1, 3])
    best_score = s
    best_U = [0, 1, 3]

    # ---- Phase 1: structured families ----
    families = []

    # Powers of 2
    for n in range(3, 18):
        families.append([0] + [2**i for i in range(1, n)])

    # Mersenne numbers (2^i - 1)
    for n in range(3, 18):
        families.append([0] + [2**i - 1 for i in range(1, n)])

    # Twin blocks {0..k-1, M..M+k-1}
    for k in [2, 3, 4, 5, 6, 7]:
        for M in [k+1, 2*k, 3*k, 5*k, 10*k, 20*k, 50*k, 100*k, 200*k, 500*k]:
            families.append(list(range(k)) + list(range(M, M+k)))

    # Geometric progressions {0, 1, r, r^2, ...}
    for r in [2, 3, 4, 5, 7, 10]:
        for k in [3, 4, 5, 6, 7, 8]:
            U = [0, 1]
            val = r
            for i in range(k-2):
                U.append(val)
                val *= r
            families.append(U)

    # Sparse {0, 1, M, M+1} and {0, 1, 2, M, M+1}
    for M in [2, 3, 5, 10, 50, 100, 500, 1000, 5000, 10000]:
        families.append([0, 1, M, M+1])
        families.append([0, 1, 2, M, M+1])

    # {0, 1, M, M+2, 2M}
    for M in [3, 5, 10, 50, 100, 500]:
        families.append([0, 1, M, M+2, 2*M])

    # Arithmetic {0,1} + multiples
    for k in [2, 3, 4, 5, 6]:
        for M in [2, 3, 5, 7, 10, 20, 50, 100]:
            U = [0, 1] + [i*M for i in range(1, k+1)]
            families.append(sorted(set(U)))

    # Fibonacci-like
    for start_val in [1, 2, 3]:
        U = [0, start_val]
        a, b = start_val, start_val + 1
        for _ in range(14):
            a, b = b, a + b
            U.append(a)
        families.append(sorted(U))

    for U in families:
        if time.time() > deadline - 30:
            break
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]

    # ---- Phase 2: exhaustive small search for a strong anchor ----
    if time.time() < deadline - 25:
        for a in range(1, 12):
            for b in range(a+1, 25):
                for c in range(b+1, 45):
                    if time.time() > deadline - 25:
                        break
                    U = [0, a, b, c]
                    s = compute_score(U)
                    if s > best_score:
                        best_score = s
                        best_U = U[:]

    # ---- Phase 3: random restarts ----
    random.seed(42)
    diverse = []
    if time.time() < deadline - 15:
        for _ in range(5000):
            if time.time() > deadline - 15:
                break
            n = random.randint(3, 15)
            if random.random() < 0.5:
                max_val = random.randint(10, 200)
            else:
                max_val = random.randint(50, 3000)
            U = set([0])
            for _ in range(n - 1):
                U.add(random.randint(1, max_val))
            U = sorted(U)
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
            if len(diverse) < 15 and s > best_score - 0.015:
                diverse.append(U[:])

    # ---- Phase 4: local search from best + diverse anchors ----
    if time.time() < deadline - 10:
        opt_U, opt_score = local_optimize(best_U, deadline, max_iters=600)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U

    seen = set()
    for cand in diverse:
        if time.time() > deadline - 5:
            break
        key = tuple(cand)
        if key in seen:
            continue
        seen.add(key)
        if key == tuple(best_U):
            continue
        opt_U, opt_score = local_optimize(cand, deadline, max_iters=300)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U

    # ---- Phase 5: final polish pass ----
    if time.time() < deadline - 2:
        opt_U, opt_score = local_optimize(best_U, deadline, max_iters=500)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U

    return best_U

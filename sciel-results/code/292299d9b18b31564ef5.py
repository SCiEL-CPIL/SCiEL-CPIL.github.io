import time
import math
import random
import numpy as np


def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    n = len(U)
    # Use numpy for moderate sizes; for larger, use set-based to avoid big arrays
    if n <= 40:
        U_arr = np.array(U)
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        sum_set_size = len(np.unique(sums))
        diff_set_size = len(np.unique(diffs))
    else:
        U_set = set(U)
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        sum_set_size = len(sums)
        diff_set_size = len(diffs)
    ratio = diff_set_size / sum_set_size
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom


def _candidate_pool(U, maxU, rng, count=300):
    """Generate a diverse set of candidate values for local moves."""
    cands = set()
    U_set = set(U)
    # Local perturbations around existing elements (wider range)
    for u in U:
        for d in list(range(-10, 11)) + [16, 21, 27, 34, 42, 55]:
            v = u + d
            if 0 < v:
                cands.add(v)
    # Multiplicative structure (more factors)
    for f in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 16, 21, 27]:
        for u in U:
            v = u * f
            if v <= maxU * 4:
                cands.add(v)
        cands.add(maxU * f)
        cands.add(maxU * f + 1)
        cands.add(maxU * f - 1)
        cands.add(maxU * f + 2)
        cands.add(maxU * f - 2)
    # Additive combinations of existing elements
    for a in U:
        for b in U:
            if a != b:
                v = a + b
                if 0 < v <= maxU * 3:
                    cands.add(v)
                v2 = abs(a - b)
                if 0 < v2 <= maxU * 3:
                    cands.add(v2)
    # Small dense region (extended)
    for v in range(1, max(40, maxU // 2 + 10)):
        cands.add(v)
    # Geometric candidates around maxU
    for base in U:
        for r in [2, 3, 4, 5]:
            v = base * r
            if v <= maxU * 5:
                cands.add(v)
    # Random
    for _ in range(count):
        r = rng.random()
        if r < 0.3:
            cands.add(rng.randint(1, max(50, maxU)))
        elif r < 0.6:
            cands.add(rng.randint(1, max(100, maxU * 2)))
        else:
            cands.add(rng.randint(1, maxU * 6 + 100))
    # Filter out existing
    cands -= U_set
    return cands


def local_optimize(U, deadline, rng, max_iters=1200):
    """Local search: add / remove / adjust elements to improve score."""
    U = sorted(set(U))
    current_score = compute_score(U)
    best_U = U[:]
    best_score = current_score
    U_set = set(U)

    for _ in range(max_iters):
        if time.time() > deadline:
            break

        improved = False
        n = len(U)
        maxU = U[-1] if U else 1

        # Build candidate pool
        pool = _candidate_pool(U, maxU, rng, count=200)
        pool = sorted(pool)

        # 1) Try adding a new element
        if n < 25:
            for cand in pool:
                new_U = sorted(U + [cand])
                s = compute_score(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    U_set = set(U)
                    improved = True
                    break

        # 2) Try removing an element
        if n > 3:
            for i in range(1, n):
                new_U = U[:i] + U[i + 1:]
                s = compute_score(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    U_set = set(U)
                    improved = True
                    break

        # 3) Try adjusting (replace) an element with a pool candidate
        if n > 2:
            for i in range(1, n):
                if improved:
                    break
                for cand in pool:
                    new_U = U[:i] + [cand] + U[i + 1:]
                    new_U = sorted(set(new_U))
                    if len(new_U) != n:
                        continue
                    s = compute_score(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U
                        U = new_U
                        U_set = set(U)
                        improved = True
                        break

        if not improved:
            # Try a random perturbation to escape local optimum
            if n > 3 and rng.random() < 0.3:
                idx = rng.randint(1, n - 1)
                new_val = rng.randint(1, maxU * 3 + 50)
                new_U = U[:idx] + [new_val] + U[idx + 1:]
                new_U = sorted(set(new_U))
                if len(new_U) == n:
                    s = compute_score(new_U)
                    if s > best_score * 0.99:  # accept slightly worse to escape
                        U = new_U
                        U_set = set(U)
            else:
                break

    return best_U, best_score


def construct():
    start = time.time()
    deadline = start + 140  # safety margin under 150s

    rng = random.Random(12345)

    best_score = 0
    best_U = [0, 1, 3]
    best_score = compute_score([0, 1, 3])

    # ---- Phase 1: structured families ----
    families = []

    # Powers of 2
    for n in range(3, 16):
        families.append([0] + [2 ** i for i in range(1, n)])

    # Mersenne numbers (2^i - 1)
    for n in range(3, 16):
        families.append([0] + [2 ** i - 1 for i in range(1, n)])

    # Geometric progressions {0, 1, r, r^2, ...}
    for r in [2, 3, 4, 5, 6, 7, 8, 10]:
        for k in [3, 4, 5, 6, 7, 8, 9]:
            U = [0, 1]
            val = r
            for i in range(k - 2):
                U.append(val)
                val *= r
            families.append(sorted(U))

    # Twin blocks {0..k-1, M..M+k-1}
    for k in [2, 3, 4, 5, 6, 7, 8]:
        for M in [k + 1, 2 * k, 3 * k, 5 * k, 10 * k, 20 * k, 50 * k, 100 * k]:
            families.append(list(range(k)) + list(range(M, M + k)))

    # Sparse {0, 1, M, M+1} and variants
    for M in [2, 3, 5, 8, 10, 20, 50, 100, 500, 1000, 5000]:
        families.append([0, 1, M, M + 1])
        families.append([0, 1, 2, M, M + 1])
        families.append([0, 1, M, M + 2, 2 * M])
        families.append([0, 1, M, 2 * M, 3 * M])

    # Arithmetic {0,1} + multiples
    for k in [2, 3, 4, 5, 6]:
        for M in [2, 3, 4, 5, 7, 10, 20, 50]:
            U = [0, 1] + [i * M for i in range(1, k + 1)]
            families.append(sorted(set(U)))

    # Mixed: small dense + sparse large
    for k in [2, 3, 4]:
        for M in [10, 20, 50, 100, 300, 1000]:
            U = list(range(k)) + [M, M + k]
            families.append(sorted(set(U)))

    for U in families:
        if time.time() > deadline - 40:
            break
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = sorted(set(U))

    # ---- Phase 2: exhaustive small search ----
    if time.time() < deadline - 30:
        # size 4: {0, a, b, c} with wider range
        for a in range(1, 20):
            for b in range(a + 1, 40):
                for c in range(b + 1, 80):
                    if time.time() > deadline - 30:
                        break
                    U = [0, a, b, c]
                    s = compute_score(U)
                    if s > best_score:
                        best_score = s
                        best_U = U[:]
        # size 5: {0, a, b, c, d} - limited search
        if time.time() < deadline - 25:
            for a in range(1, 12):
                for b in range(a + 1, 25):
                    for c in range(b + 1, 45):
                        for d in range(c + 1, 70):
                            if time.time() > deadline - 25:
                                break
                            U = [0, a, b, c, d]
                            s = compute_score(U)
                            if s > best_score:
                                best_score = s
                                best_U = U[:]

    # ---- Phase 3: random restarts ----
    diverse = []
    if time.time() < deadline - 20:
        for _ in range(8000):
            if time.time() > deadline - 20:
                break
            n = rng.randint(3, 14)
            strategy = rng.random()
            if strategy < 0.3:
                # Dense small set
                max_val = rng.randint(10, 80)
            elif strategy < 0.6:
                # Medium range
                max_val = rng.randint(50, 300)
            elif strategy < 0.8:
                # Large range
                max_val = rng.randint(100, 2000)
            else:
                # Very large sparse
                max_val = rng.randint(500, 5000)
            U = set([0])
            for _ in range(n - 1):
                U.add(rng.randint(1, max_val))
            U = sorted(U)
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
            if len(diverse) < 20 and s > best_score - 0.03:
                diverse.append(U[:])

    # ---- Phase 4: local search from best ----
    if time.time() < deadline - 10:
        opt_U, opt_score = local_optimize(best_U, deadline - 10, rng, max_iters=1000)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U

    # ---- Phase 5: local search from diverse candidates ----
    seen = set()
    diverse_remaining = [c for c in diverse if tuple(c) != tuple(best_U)]
    for cand in diverse_remaining[:12]:
        if time.time() > deadline - 8:
            break
        key = tuple(cand)
        if key in seen:
            continue
        seen.add(key)
        opt_U, opt_score = local_optimize(cand, deadline - 8, rng, max_iters=400)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U

    # ---- Phase 6: final polish with more iterations ----
    if time.time() < deadline - 1:
        opt_U, opt_score = local_optimize(best_U, deadline - 1, rng, max_iters=800)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U

    return best_U

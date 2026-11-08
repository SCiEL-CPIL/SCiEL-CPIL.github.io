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
    if n <= 50:
        U_arr = np.array(U, dtype=np.int64)
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


def _fast_score(U):
    """Quick score using NumPy vectorization for all sizes."""
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    n = len(U)
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    sum_set_size = len(np.unique(sums))
    diff_set_size = len(np.unique(diffs))
    ratio = diff_set_size / sum_set_size
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom


def _candidate_pool(U, maxU, rng, count=400):
    """Generate a diverse set of candidate values for local moves."""
    cands = set()
    U_set = set(U)
    n = len(U)
    
    # Local perturbations around existing elements
    for u in U:
        for d in range(-15, 16):
            v = u + d
            if 0 < v:
                cands.add(v)
        # Wider jumps
        for d in [20, 25, 30, 40, 50, 60, 80, 100]:
            v = u + d
            if 0 < v <= maxU * 5:
                cands.add(v)
            v = u - d
            if 0 < v:
                cands.add(v)
    
    # Multiplicative structure
    for f in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 50]:
        for u in U:
            v = u * f
            if v <= maxU * 8:
                cands.add(v)
        cands.add(maxU * f)
        cands.add(maxU * f + 1)
        cands.add(maxU * f - 1)
        cands.add(maxU * f + 2)
        cands.add(maxU * f - 2)
        cands.add(maxU * f + 5)
        cands.add(maxU * f - 5)
    
    # Additive combinations of existing elements
    for a in U:
        for b in U:
            if a != b:
                v = a + b
                if 0 < v <= maxU * 4:
                    cands.add(v)
                v2 = abs(a - b)
                if 0 < v2 <= maxU * 4:
                    cands.add(v2)
                # Linear combinations
                for c in [2, 3]:
                    v3 = a + c * b
                    if 0 < v3 <= maxU * 4:
                        cands.add(v3)
                    v4 = b + c * a
                    if 0 < v4 <= maxU * 4:
                        cands.add(v4)
    
    # Small dense region
    for v in range(1, max(60, maxU // 2 + 20)):
        cands.add(v)
    
    # Geometric candidates
    for base in U:
        for r in [2, 3, 4, 5, 7]:
            v = base * r
            if v <= maxU * 6:
                cands.add(v)
            v2 = base * r + 1
            if v2 <= maxU * 6:
                cands.add(v2)
            v3 = base * r - 1
            if 0 < v3 <= maxU * 6:
                cands.add(v3)
    
    # Random candidates with various scales
    for _ in range(count):
        r = rng.random()
        if r < 0.2:
            cands.add(rng.randint(1, max(30, maxU // 3)))
        elif r < 0.4:
            cands.add(rng.randint(1, max(50, maxU)))
        elif r < 0.6:
            cands.add(rng.randint(1, max(100, maxU * 2)))
        elif r < 0.8:
            cands.add(rng.randint(1, max(200, maxU * 4)))
        else:
            cands.add(rng.randint(1, maxU * 8 + 200))
    
    # Filter out existing
    cands -= U_set
    return cands


def local_optimize(U, deadline, rng, max_iters=2000):
    """Local search: add / remove / adjust elements to improve score."""
    U = sorted(set(U))
    current_score = _fast_score(U)
    best_U = U[:]
    best_score = current_score
    U_set = set(U)
    cache = {}
    
    no_improve_count = 0
    max_no_improve = 50
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        n = len(U)
        maxU = U[-1] if U else 1
        
        # Build candidate pool
        pool = _candidate_pool(U, maxU, rng, count=200)
        pool = sorted(pool)
        
        # 1) Try adding a new element
        if n < 20:
            for cand in pool[:80]:
                new_U = sorted(U + [cand])
                s = _fast_score(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    U_set = set(U)
                    improved = True
                    no_improve_count = 0
                    break
        
        # 2) Try removing an element
        if n > 3:
            for i in range(1, n):
                new_U = U[:i] + U[i + 1:]
                s = _fast_score(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    U_set = set(U)
                    improved = True
                    no_improve_count = 0
                    break
        
        # 3) Try adjusting (replace) an element with a pool candidate
        if n > 2:
            for i in range(1, n):
                if improved:
                    break
                for cand in pool[:60]:
                    new_U = U[:i] + [cand] + U[i + 1:]
                    new_U = sorted(set(new_U))
                    if len(new_U) != n:
                        continue
                    s = _fast_score(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U
                        U = new_U
                        U_set = set(U)
                        improved = True
                        no_improve_count = 0
                        break
        
        # 4) Try pairwise swap (replace two elements at once)
        if n > 4 and not improved:
            # Generate swap-specific candidates
            swap_cands = set()
            for i in range(1, n):
                for j in range(i + 1, n):
                    a, b = U[i], U[j]
                    for v in [a + b, abs(a - b), a * 2 - b, b * 2 - a, a + b + 1, a + b - 1]:
                        if 0 < v <= maxU * 5 and v not in U_set:
                            swap_cands.add(v)
            swap_cands = sorted(swap_cands)[:40]
            if swap_cands:
                for i in range(1, n):
                    if improved:
                        break
                    for j in range(i + 1, n):
                        if improved:
                            break
                        for cand in swap_cands:
                            new_U = U[:i] + U[i+1:j] + U[j+1:]
                            new_U = sorted(set(new_U + [cand]))
                            if len(new_U) != n - 1:
                                continue
                            s = _fast_score(new_U)
                            if s > best_score + 1e-12:
                                best_score = s
                                best_U = new_U
                                U = new_U
                                U_set = set(U)
                                improved = True
                                no_improve_count = 0
                                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Try a random perturbation to escape local optimum
                if n > 3 and rng.random() < 0.4:
                    idx = rng.randint(1, n - 1)
                    new_val = rng.randint(1, maxU * 4 + 100)
                    new_U = U[:idx] + [new_val] + U[idx + 1:]
                    new_U = sorted(set(new_U))
                    if len(new_U) == n:
                        s = _fast_score(new_U)
                        if s > best_score * 0.98:  # accept slightly worse to escape
                            U = new_U
                            U_set = set(U)
                            no_improve_count = 0
                elif n > 5 and rng.random() < 0.2:
                    # Remove two elements
                    idx1 = rng.randint(1, n - 1)
                    idx2 = rng.randint(1, n - 2)
                    if idx2 >= idx1:
                        idx2 += 1
                    new_U = U[:idx1] + U[idx1+1:idx2] + U[idx2+1:]
                    if len(new_U) >= 3:
                        s = _fast_score(new_U)
                        if s > best_score * 0.98:
                            U = new_U
                            U_set = set(U)
                            no_improve_count = 0
                else:
                    break
            else:
                # Try a smaller random perturbation
                if n > 3 and rng.random() < 0.3:
                    idx = rng.randint(1, n - 1)
                    delta = rng.choice([-2, -1, 1, 2, 3, 5, -3, -5])
                    new_val = U[idx] + delta
                    if new_val > 0:
                        new_U = U[:idx] + [new_val] + U[idx + 1:]
                        new_U = sorted(set(new_U))
                        if len(new_U) == n:
                            s = _fast_score(new_U)
                            if s > best_score * 0.99:
                                U = new_U
                                U_set = set(U)
                else:
                    break
    
    return best_U, best_score


def construct():
    start = time.time()
    deadline = start + 130  # safety margin under 150s

    rng = random.Random(12345)

    best_score = 0
    best_U = [0, 1, 3]
    best_score = _fast_score([0, 1, 3])

    # ---- Phase 1: structured families ----
    families = []

    # Powers of 2
    for n in range(3, 18):
        families.append([0] + [2 ** i for i in range(1, n)])

    # Mersenne numbers (2^i - 1)
    for n in range(3, 18):
        families.append([0] + [2 ** i - 1 for i in range(1, n)])

    # Geometric progressions {0, 1, r, r^2, ...}
    for r in [2, 3, 4, 5, 6, 7, 8, 10, 12]:
        for k in [3, 4, 5, 6, 7, 8, 9, 10]:
            U = [0, 1]
            val = r
            for i in range(k - 2):
                U.append(val)
                val *= r
            families.append(sorted(U))

    # Twin blocks {0..k-1, M..M+k-1}
    for k in [2, 3, 4, 5, 6, 7, 8, 10]:
        for M in [k + 1, 2 * k, 3 * k, 5 * k, 8 * k, 10 * k, 20 * k, 50 * k, 100 * k, 200 * k]:
            families.append(list(range(k)) + list(range(M, M + k)))

    # Sparse {0, 1, M, M+1} and variants
    for M in [2, 3, 5, 8, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
        families.append([0, 1, M, M + 1])
        families.append([0, 1, 2, M, M + 1])
        families.append([0, 1, M, M + 2, 2 * M])
        families.append([0, 1, M, 2 * M, 3 * M])
        families.append([0, 1, M, 2 * M, 3 * M, 4 * M])

    # Arithmetic {0,1} + multiples
    for k in [2, 3, 4, 5, 6, 7, 8]:
        for M in [2, 3, 4, 5, 7, 10, 15, 20, 30, 50]:
            U = [0, 1] + [i * M for i in range(1, k + 1)]
            families.append(sorted(set(U)))

    # Mixed: small dense + sparse large
    for k in [2, 3, 4, 5]:
        for M in [10, 20, 50, 100, 200, 500, 1000, 2000]:
            U = list(range(k)) + [M, M + k]
            families.append(sorted(set(U)))
            U2 = list(range(k)) + [M, M + 1, M + k]
            families.append(sorted(set(U2)))

    # Fibonacci-like sets
    for n in range(4, 14):
        fib = [0, 1]
        for i in range(n - 2):
            fib.append(fib[-1] + fib[-2])
        families.append(fib[:])

    # Mixed geometric + arithmetic
    for r in [2, 3, 4, 5]:
        for k in [2, 3, 4, 5]:
            for M in [5, 10, 20, 50, 100, 200]:
                U = [0, 1] + [r ** i for i in range(1, k)] + [M, M + 1]
                families.append(sorted(set(U)))

    # Sets with structured gaps
    for base in [3, 4, 5, 6, 7, 8]:
        for k in [3, 4, 5, 6, 7, 8]:
            U = [0] + [base ** i for i in range(1, k)]
            families.append(U)

    # NEW: Sidon-like constructions using modular arithmetic
    # {0, 1, m, m+1, 2m, 2m+1, ...} pattern
    for m in [3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        for k in [2, 3, 4, 5]:
            U = [0]
            for i in range(k):
                U.append(i * m)
                U.append(i * m + 1)
            families.append(sorted(set(U)))

    # NEW: Beatty sequence-like
    for alpha in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        for n in range(3, 10):
            U = [0]
            for i in range(1, n + 1):
                U.append(int(alpha * i))
            families.append(sorted(set(U)))

    # NEW: Mixed scale construction
    for a in [2, 3, 4, 5]:
        for b in [3, 4, 5, 7, 10]:
            for c in [10, 20, 50, 100]:
                U = [0, 1, a, b, c, c + 1, 2 * c, 2 * c + 1]
                families.append(sorted(set(U)))

    for U in families:
        if time.time() > deadline - 40:
            break
        s = _fast_score(U)
        if s > best_score:
            best_score = s
            best_U = sorted(set(U))

    # ---- Phase 2: exhaustive small search ----
    if time.time() < deadline - 35:
        # size 4: {0, a, b, c} with wider range
        for a in range(1, 25):
            for b in range(a + 1, 50):
                for c in range(b + 1, 100):
                    if time.time() > deadline - 35:
                        break
                    U = [0, a, b, c]
                    s = _fast_score(U)
                    if s > best_score:
                        best_score = s
                        best_U = U[:]
        # size 5: {0, a, b, c, d} - limited search
        if time.time() < deadline - 30:
            for a in range(1, 15):
                for b in range(a + 1, 30):
                    for c in range(b + 1, 55):
                        for d in range(c + 1, 85):
                            if time.time() > deadline - 30:
                                break
                            U = [0, a, b, c, d]
                            s = _fast_score(U)
                            if s > best_score:
                                best_score = s
                                best_U = U[:]
        # size 6: {0, a, b, c, d, e} - very limited
        if time.time() < deadline - 25:
            for a in range(1, 10):
                for b in range(a + 1, 20):
                    for c in range(b + 1, 38):
                        for d in range(c + 1, 58):
                            for e in range(d + 1, 80):
                                if time.time() > deadline - 25:
                                    break
                                U = [0, a, b, c, d, e]
                                s = _fast_score(U)
                                if s > best_score:
                                    best_score = s
                                    best_U = U[:]

    # ---- Phase 3: random restarts ----
    diverse = []
    if time.time() < deadline - 25:
        for _ in range(10000):
            if time.time() > deadline - 25:
                break
            n = rng.randint(3, 15)
            strategy = rng.random()
            if strategy < 0.25:
                # Dense small set
                max_val = rng.randint(10, 100)
            elif strategy < 0.5:
                # Medium range
                max_val = rng.randint(50, 400)
            elif strategy < 0.75:
                # Large range
                max_val = rng.randint(100, 3000)
            else:
                # Very large sparse
                max_val = rng.randint(500, 8000)
            U = set([0])
            for _ in range(n - 1):
                U.add(rng.randint(1, max_val))
            U = sorted(U)
            s = _fast_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
            if len(diverse) < 25 and s > best_score - 0.03:
                diverse.append(U[:])
        
        # Structured random restarts
        if time.time() < deadline - 20:
            for _ in range(4000):
                if time.time() > deadline - 20:
                    break
                # Generate sets with some structure
                n = rng.randint(4, 12)
                U = set([0])
                # Start with a geometric or arithmetic core
                if rng.random() < 0.5:
                    r = rng.choice([2, 3, 4, 5])
                    val = r
                    for _ in range(rng.randint(2, min(n-1, 7))):
                        U.add(val)
                        val *= r
                else:
                    step = rng.randint(2, 10)
                    for i in range(1, rng.randint(2, min(n-1, 7))):
                        U.add(i * step)
                # Fill remaining with random
                max_val = max(U) if U else 10
                for _ in range(n - len(U)):
                    U.add(rng.randint(1, max_val * 3))
                U = sorted(U)
                s = _fast_score(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]
                if len(diverse) < 35 and s > best_score - 0.02:
                    diverse.append(U[:])

    # ---- Phase 4: local search from best ----
    if time.time() < deadline - 15:
        opt_U, opt_score = local_optimize(best_U, deadline - 15, rng, max_iters=2000)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U

    # ---- Phase 5: local search from diverse candidates ----
    seen = set()
    diverse_remaining = [c for c in diverse if tuple(c) != tuple(best_U)]
    # Sort by score descending to prioritize best candidates
    diverse_remaining.sort(key=lambda x: -_fast_score(x))
    for cand in diverse_remaining[:20]:
        if time.time() > deadline - 10:
            break
        key = tuple(cand)
        if key in seen:
            continue
        seen.add(key)
        opt_U, opt_score = local_optimize(cand, deadline - 10, rng, max_iters=600)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U

    # ---- Phase 6: final polish with more iterations ----
    if time.time() < deadline - 2:
        opt_U, opt_score = local_optimize(best_U, deadline - 2, rng, max_iters=3000)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U

    # ---- Phase 7: high-precision final polish ----
    if time.time() < deadline - 0.5:
        final_deadline = deadline - 0.5
        U = sorted(set(best_U))
        cur_score = _fast_score(U)
        U_set = set(U)
        maxU = U[-1]
        for it in range(5000):
            if time.time() > final_deadline:
                break
            n = len(U)
            improved = False
            # Try single-element replacements with wider candidate range
            for i in range(1, n):
                if improved:
                    break
                u = U[i]
                for delta in range(-30, 31):
                    if delta == 0:
                        continue
                    v = u + delta
                    if v <= 0 or v in U_set:
                        continue
                    new_U = U[:i] + [v] + U[i+1:]
                    new_U = sorted(new_U)
                    s = _fast_score(new_U)
                    if s > cur_score + 1e-12:
                        cur_score = s
                        U = new_U
                        U_set = set(U)
                        maxU = U[-1]
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_U = U[:]
                        break
                # Try multiplicative adjustments
                if not improved:
                    for f in [2, 3, 4, 5]:
                        v = u * f
                        if v <= 0 or v in U_set:
                            continue
                        new_U = U[:i] + [v] + U[i+1:]
                        new_U = sorted(new_U)
                        s = _fast_score(new_U)
                        if s > cur_score + 1e-12:
                            cur_score = s
                            U = new_U
                            U_set = set(U)
                            maxU = U[-1]
                            improved = True
                            if s > best_score:
                                best_score = s
                                best_U = U[:]
                            break
            # Try removing an element
            if not improved and n > 3:
                for i in range(1, n):
                    new_U = U[:i] + U[i+1:]
                    s = _fast_score(new_U)
                    if s > cur_score + 1e-12:
                        cur_score = s
                        U = new_U
                        U_set = set(U)
                        maxU = U[-1]
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_U = U[:]
                        break
            # Try adding an element
            if not improved and n < 20:
                for v in range(1, maxU + 50):
                    if v in U_set:
                        continue
                    new_U = U + [v]
                    new_U = sorted(new_U)
                    s = _fast_score(new_U)
                    if s > cur_score + 1e-12:
                        cur_score = s
                        U = new_U
                        U_set = set(U)
                        maxU = U[-1]
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_U = U[:]
                        break
            if not improved:
                break

    return best_U

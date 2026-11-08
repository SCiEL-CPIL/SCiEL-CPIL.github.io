import time
import math
import random
import numpy as np

def compute_score(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    # Use numpy for speed on moderate sizes
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    
    sum_set_size = len(np.unique(sums))
    diff_set_size = len(np.unique(diffs))
    
    ratio = diff_set_size / sum_set_size
    if ratio <= 0:
        return 0.0
    score = 1.0 + math.log(ratio) / math.log(2 * maxU + 1)
    return score

def _score_with_cache(U, cache):
    """Score with a simple cache keyed by tuple."""
    key = tuple(U)
    if key in cache:
        return cache[key]
    s = compute_score(U)
    cache[key] = s
    return s

def exhaustive_search_small(n, max_val, deadline):
    """Exhaustive search for small sets of size n"""
    best_score = 0.0
    best_set = None
    candidates = list(range(1, max_val + 1))
    from itertools import combinations
    for combo in combinations(candidates, n - 1):
        if time.time() > deadline:
            break
        U = sorted([0] + list(combo))
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    return best_score, best_set

def parametric_families(deadline):
    """Try various parametric families"""
    best_score = 0.0
    best_set = None
    
    # Family 1: Arithmetic progression
    for n in range(3, 60):
        if time.time() > deadline:
            break
        U = list(range(n))
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Family 2: Mersenne-like (2^k - 1)
    for n in range(3, 20):
        if time.time() > deadline:
            break
        U = [0] + [2**k - 1 for k in range(1, n)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Family 3: Powers of 2
    for n in range(3, 20):
        if time.time() > deadline:
            break
        U = [0] + [2**k for k in range(1, n)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Family 4: Triangular numbers
    for n in range(3, 30):
        if time.time() > deadline:
            break
        U = [0] + [k*(k+1)//2 for k in range(1, n)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Family 5: Quadratic k^2//2
    for n in range(3, 30):
        if time.time() > deadline:
            break
        U = [0] + [k*k//2 for k in range(1, n)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Family 6: Mixed dense + sparse
    for base_n in range(3, 12):
        if time.time() > deadline:
            break
        for gap in range(2, 25):
            if time.time() > deadline:
                break
            for extra in range(0, 6):
                if time.time() > deadline:
                    break
                U = list(range(base_n))
                for i in range(extra):
                    U.append(base_n * gap * (i + 1))
                U = sorted(set(U))
                s = compute_score(U)
                if s > best_score:
                    best_score = s
                    best_set = U
    
    return best_score, best_set

def _candidate_pool(U, maxU, rng, count=300):
    """Generate a diverse set of candidate values for local moves."""
    cands = set()
    U_set = set(U)
    
    # Local perturbations around existing elements
    for u in U:
        for d in list(range(-15, 16)) + [21, 27, 34, 42, 55, 64, 81]:
            v = u + d
            if 0 < v:
                cands.add(v)
    
    # Multiplicative structure
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
    
    # Small dense region
    for v in range(1, max(50, maxU // 2 + 10)):
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
    
    cands -= U_set
    return cands

def local_optimize(U, deadline, rng, max_iters=1500):
    """Local search: add / remove / adjust elements to improve score."""
    U = sorted(set(U))
    current_score = compute_score(U)
    best_U = U[:]
    best_score = current_score
    cache = {}

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
        if n < 30:
            for cand in pool:
                new_U = sorted(U + [cand])
                s = _score_with_cache(new_U, cache)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    improved = True
                    break

        # 2) Try removing an element
        if n > 3:
            for i in range(1, n):
                new_U = U[:i] + U[i + 1:]
                s = _score_with_cache(new_U, cache)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    improved = True
                    break

        # 3) Try adjusting (replace) an element with a pool candidate
        if n > 2 and not improved:
            for i in range(1, n):
                if improved:
                    break
                for cand in pool:
                    new_U = U[:i] + [cand] + U[i + 1:]
                    new_U = sorted(set(new_U))
                    if len(new_U) != n:
                        continue
                    s = _score_with_cache(new_U, cache)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U
                        U = new_U
                        improved = True
                        break

        # 4) Try pairwise swap (replace two elements at once)
        if n > 4 and not improved:
            swap_pool = _pairwise_swap_pool(U, maxU, rng, count=100)
            swap_pool = sorted(swap_pool)
            if swap_pool:
                for i in range(1, n):
                    if improved:
                        break
                    for j in range(i + 1, n):
                        if improved:
                            break
                        for cand in swap_pool[:50]:
                            new_U = U[:i] + U[i+1:j] + U[j+1:]
                            new_U = sorted(set(new_U + [cand]))
                            if len(new_U) != n - 1:
                                continue
                            s = _score_with_cache(new_U, cache)
                            if s > best_score + 1e-12:
                                best_score = s
                                best_U = new_U
                                U = new_U
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
                    s = _score_with_cache(new_U, cache)
                    if s > best_score * 0.99:
                        U = new_U
            else:
                break

    return best_U, best_score

def _pairwise_swap_pool(U, maxU, rng, count=200):
    """Generate candidates specifically for pairwise swaps."""
    cands = set()
    U_set = set(U)
    n = len(U)
    for i in range(1, n):
        for j in range(i + 1, n):
            a, b = U[i], U[j]
            for v in [a + b, abs(a - b), a * b, a + b + 1, a + b - 1, a * b + 1, a * b - 1]:
                if 0 < v <= maxU * 5 and v not in U_set:
                    cands.add(v)
            for coeff in [1, 2, 3]:
                v1 = a + coeff * b
                v2 = b + coeff * a
                if 0 < v1 <= maxU * 5 and v1 not in U_set:
                    cands.add(v1)
                if 0 < v2 <= maxU * 5 and v2 not in U_set:
                    cands.add(v2)
    for _ in range(count):
        cands.add(rng.randint(1, maxU * 4 + 100))
    cands -= U_set
    return cands

def greedy_frequency_gap(deadline):
    """Greedy construction: each new element maximizes new differences relative to sumset growth."""
    best_score = 0.0
    best_set = [0]
    
    # Start with small dense base
    for base_size in range(3, 15):
        if time.time() > deadline:
            break
        U = list(range(base_size))
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Now greedily add elements
    U = best_set[:]
    current_score = best_score
    maxU = max(U) if U else 0
    cache = {}
    
    for step in range(50):
        if time.time() > deadline:
            break
        
        best_cand = None
        best_cand_score = current_score
        
        # Generate candidates
        cands = set()
        for u in U:
            for f in range(2, 20):
                cands.add(u * f)
                cands.add(u * f + 1)
                cands.add(u * f - 1)
        for v in range(1, maxU * 3 + 100, max(1, maxU // 20)):
            cands.add(v)
        for _ in range(200):
            cands.add(random.randint(1, maxU * 5 + 100))
        cands -= set(U)
        
        for cand in sorted(cands):
            if time.time() > deadline:
                break
            new_U = sorted(U + [cand])
            s = _score_with_cache(new_U, cache)
            if s > best_cand_score + 1e-12:
                best_cand_score = s
                best_cand = cand
        
        if best_cand is None:
            break
        
        U = sorted(U + [best_cand])
        current_score = best_cand_score
        maxU = max(U)
        
        if current_score > best_score:
            best_score = current_score
            best_set = U[:]
    
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # 1. Try small exhaustive search
    for n in range(3, 7):
        if time.time() > deadline:
            break
        max_val = 30 if n <= 5 else 20
        s, U = exhaustive_search_small(n, max_val, deadline)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 2. Try parametric families
    s, U = parametric_families(deadline)
    if s > best_score:
        best_score = s
        best_set = U
    
    # 3. Greedy frequency-gap construction
    if time.time() < deadline - 30:
        s, U = greedy_frequency_gap(deadline - 30)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 4. Local optimize from best found so far
    if time.time() < deadline - 10:
        rng = random.Random(42)
        s, U = local_optimize(best_set, deadline - 10, rng, max_iters=2000)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 5. Try more local optimization from different starting points
    if time.time() < deadline - 5:
        for seed in range(5):
            if time.time() > deadline:
                break
            rng = random.Random(seed)
            n = random.Random(seed).randint(4, 15)
            U = sorted(random.Random(seed).sample(range(1, 100), n - 1) + [0])
            s, U2 = local_optimize(U, deadline, rng, max_iters=1000)
            if s > best_score:
                best_score = s
                best_set = U2
    
    # 6. Final polish pass
    if time.time() < deadline - 2:
        rng = random.Random(999)
        s, U = local_optimize(best_set, deadline, rng, max_iters=500)
        if s > best_score:
            best_score = s
            best_set = U
    
    return best_set

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END

import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    deadline = 115.0
    n = 256

    # Precompute all 4-term APs (0-indexed)
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))

    aps_arr = np.array(aps, dtype=np.int32)
    M = len(aps_arr)

    # Per-position AP indices
    pos_aps = [[] for _ in range(n)]
    for idx, (i0, i1, i2, i3) in enumerate(aps):
        pos_aps[i0].append(idx)
        pos_aps[i1].append(idx)
        pos_aps[i2].append(idx)
        pos_aps[i3].append(idx)
    pos_aps_arr = [np.array(p, dtype=np.int32) for p in pos_aps]

    # Precompute "others": for each position p, the 3 other indices in each AP containing p
    pos_others = [None] * n
    for p in range(n):
        idxs = pos_aps_arr[p]
        if len(idxs) == 0:
            pos_others[p] = np.empty((0, 3), dtype=np.int32)
            continue
        all_idx = aps_arr[idxs]  # (K, 4)
        mask = (all_idx != p)
        rows, cols = np.where(mask)
        others_vals = all_idx[rows, cols]
        pos_others[p] = others_vals.reshape(-1, 3)

    def count_mono(col):
        cols = col[aps_arr]
        mono = (cols[:, 0] == cols[:, 1]) & (cols[:, 1] == cols[:, 2]) & (cols[:, 2] == cols[:, 3])
        return int(np.sum(mono))

    def flip_delta(col, p):
        """Fast delta: change in mono count if we flip position p."""
        others = pos_others[p]
        if others.shape[0] == 0:
            return 0
        c_p = col[p]
        c_new = 1 - c_p
        others_colors = col[others]  # (K, 3)
        is_mono_now = np.all(others_colors == c_p, axis=1)
        is_mono_new = np.all(others_colors == c_new, axis=1)
        return int(np.sum(is_mono_new) - np.sum(is_mono_now))

    def greedy_polish(col, max_iters=300):
        """First-improvement hill climbing with randomized order."""
        col = col.copy()
        cur = count_mono(col)
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            improved = False
            order = np.random.permutation(n)
            for p in order:
                d = flip_delta(col, p)
                if d < 0:
                    col[p] = 1 - col[p]
                    cur += d
                    improved = True
            if not improved:
                break
        return col, cur

    def sa_polish(col, max_iters=400):
        """Simulated annealing polish."""
        col = col.copy()
        cur = count_mono(col)
        best_col = col.copy()
        best_count = cur
        T = 6.0
        T_min = 0.01
        cooling = 0.997
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            T = max(T * cooling, T_min)
            order = np.random.permutation(n)
            for p in order:
                d = flip_delta(col, p)
                if d < 0:
                    col[p] = 1 - col[p]
                    cur += d
                    if cur < best_count:
                        best_count = cur
                        best_col = col.copy()
                elif d == 0 and random.random() < 0.2:
                    col[p] = 1 - col[p]
                elif d > 0 and random.random() < math.exp(-d / T):
                    col[p] = 1 - col[p]
                    cur += d
        return best_col, best_count

    def smooth_surrogate_polish(col, n_steps=50, lr=0.05):
        """
        Smooth surrogate: minimize sum over APs of soft_mono(AP).
        soft_mono uses log-sum-exp to approximate the indicator of monochromaticity.
        For a 4-AP with colors z_i in {-1,1}, mono if all equal.
        We use: penalty = log(1 + exp(-beta * (z0+z1+z2+z3)^2 / 4)) for all-same,
        but simpler: use the sum of pairwise disagreements as a smooth proxy.
        
        Actually, a cleaner smooth objective: for each AP, the number of color changes
        along the AP. Mono APs have 0 changes. Minimize total changes.
        changes = |z0-z1|/2 + |z1-z2|/2 + |z2-z3|/2
        This is differentiable if we use soft-absolute: |x| ≈ sqrt(x^2 + eps)
        """
        z = 2.0 * col.astype(np.float64) - 1.0  # map to {-1, 1}
        eps = 1e-4
        for step in range(n_steps):
            if time.time() > deadline - 5.0:
                break
            # Compute gradient of sum of soft-absolute differences
            grad = np.zeros(n, dtype=np.float64)
            # For each AP, compute contribution to gradient
            z0 = z[aps_arr[:, 0]]
            z1 = z[aps_arr[:, 1]]
            z2 = z[aps_arr[:, 2]]
            z3 = z[aps_arr[:, 3]]
            
            d01 = z0 - z1
            d12 = z1 - z2
            d23 = z2 - z3
            
            # soft-absolute derivative: d/dx sqrt(x^2+eps) = x/sqrt(x^2+eps)
            sd01 = d01 / np.sqrt(d01 * d01 + eps)
            sd12 = d12 / np.sqrt(d12 * d12 + eps)
            sd23 = d23 / np.sqrt(d23 * d23 + eps)
            
            # Accumulate gradient
            for i in range(M):
                g0 = 0.5 * sd01[i]
                g1 = -0.5 * sd01[i] + 0.5 * sd12[i]
                g2 = -0.5 * sd12[i] + 0.5 * sd23[i]
                g3 = -0.5 * sd23[i]
                grad[aps_arr[i, 0]] += g0
                grad[aps_arr[i, 1]] += g1
                grad[aps_arr[i, 2]] += g2
                grad[aps_arr[i, 3]] += g3
            
            z = z - lr * grad
            z = np.clip(z, -1.0, 1.0)
        
        # Threshold back to binary
        new_col = (z > 0).astype(np.int8)
        return new_col

    # Phase 1: Diverse initial solutions
    best_c = None
    best_count = float('inf')

    def try_polish(c, method='greedy', **kwargs):
        nonlocal best_c, best_count
        if method == 'greedy':
            c_p, cnt = greedy_polish(c, **kwargs)
        elif method == 'sa':
            c_p, cnt = sa_polish(c, **kwargs)
        elif method == 'smooth':
            c_s = smooth_surrogate_polish(c, **kwargs)
            c_p, cnt = greedy_polish(c_s, max_iters=200)
        else:
            c_p, cnt = greedy_polish(c, **kwargs)
        if cnt < best_count:
            best_count = cnt
            best_c = c_p.copy()

    # Thue-Morse
    c_tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    try_polish(c_tm, 'greedy', max_iters=300)

    # Perturbed Thue-Morse
    for _ in range(5):
        c_ptm = c_tm.copy()
        flips = np.random.choice(n, size=8, replace=False)
        c_ptm[flips] = 1 - c_ptm[flips]
        try_polish(c_ptm, 'greedy', max_iters=200)

    # Random
    for _ in range(6):
        c_r = np.random.randint(0, 2, n).astype(np.int8)
        try_polish(c_r, 'greedy', max_iters=200)

    # Block patterns
    for bs in [2, 3, 4, 5, 8, 16, 32]:
        c_b = np.array([(i // bs) % 2 for i in range(n)], dtype=np.int8)
        try_polish(c_b, 'greedy', max_iters=150)

    # Multi-scale Thue-Morse
    for scale in [2, 4, 8]:
        c_ms = np.array([bin(i // scale).count('1') % 2 for i in range(n)], dtype=np.int8)
        try_polish(c_ms, 'greedy', max_iters=150)

    # Phase 2: Iterated Local Search with smooth surrogate
    stuck = 0
    last_best = best_count
    while time.time() < deadline - 20.0:
        c_cur = best_c.copy()
        
        # Adaptive perturbation
        if best_count == last_best:
            stuck += 1
        else:
            stuck = 0
            last_best = best_count
        
        if stuck > 20:
            n_flips = random.randint(12, 20)
        elif stuck > 10:
            n_flips = random.randint(8, 14)
        elif stuck > 5:
            n_flips = random.randint(5, 10)
        else:
            n_flips = random.randint(3, 8)
        
        flips = np.random.choice(n, size=n_flips, replace=False)
        c_cur[flips] = 1 - c_cur[flips]
        
        # Try different polish strategies
        if random.random() < 0.3:
            # Smooth surrogate then greedy
            try_polish(c_cur, 'smooth', n_steps=40, lr=0.08)
        else:
            # SA polish
            try_polish(c_cur, 'sa', max_iters=250)
        
        # Also try pure greedy sometimes
        if random.random() < 0.25:
            try_polish(c_cur, 'greedy', max_iters=200)

    # Phase 3: Final deep polish
    if best_c is not None:
        # Multiple greedy passes
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
            c_p, cnt = greedy_polish(best_c, max_iters=400)
            if cnt < best_count:
                best_count = cnt
                best_c = c_p.copy()
        
        # SA with low temperature
        if time.time() < deadline - 8.0:
            c_p, cnt = sa_polish(best_c, max_iters=300)
            if cnt < best_count:
                best_count = cnt
                best_c = c_p.copy()
        
        # Smooth surrogate final polish
        if time.time() < deadline - 5.0:
            for lr in [0.03, 0.08, 0.15]:
                if time.time() > deadline - 3.0:
                    break
                c_s = smooth_surrogate_polish(best_c, n_steps=30, lr=lr)
                c_p, cnt = greedy_polish(c_s, max_iters=200)
                if cnt < best_count:
                    best_count = cnt
                    best_c = c_p.copy()
        
        # Final greedy
        if time.time() < deadline - 2.0:
            c_p, cnt = greedy_polish(best_c, max_iters=300)
            if cnt < best_count:
                best_count = cnt
                best_c = c_p.copy()

    # Verify
    if best_c is None:
        best_c = c_tm
    actual = count_mono(best_c)
    if actual < best_count:
        best_count = actual

    return best_c.tolist()

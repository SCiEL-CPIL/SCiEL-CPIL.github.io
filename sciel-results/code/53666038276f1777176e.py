"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    def ratio(h):
        h = np.asarray(h, dtype=float)
        l2_sq = float(np.dot(h, h))
        l1 = float(np.sum(h))
        if l1 == 0.0:
            return 0.0
        # autoconvolution (linear, full)
        ac = np.convolve(h, h, mode='full')
        sup = float(np.max(ac))
        if sup == 0.0:
            return 0.0
        return l2_sq / (l1 * sup)

    # Start with the Sidon-based seed
    S = [0]
    used_sums = {0}
    for x in range(1, N):
        new_sums = [x + s for s in S]
        new_sums.append(2 * x)
        collision = False
        for s in new_sums:
            if s in used_sums:
                collision = True
                break
        if not collision:
            S.append(x)
            for s in new_sums:
                used_sums.add(s)
        if time.time() > deadline - 60:
            break

    k = len(S)
    h = np.zeros(N)
    if k > 0:
        val = 1.0 / k
        for idx in S:
            h[idx] = val

    best_h = h.copy()
    best_r = ratio(best_h)

    # --- Structural change: local coordinate ascent with weight perturbation ---
    # Represent the vector as having a fixed support set, and adjust weights.
    # Use cyclic coordinate descent to maximize the ratio.
    
    # First, try to improve support by adding/removing elements
    support = set(S)
    
    def rebuild_from_support_weights(support_set, weights):
        """Build h vector from support set and corresponding weights."""
        hv = np.zeros(N)
        for i, idx in enumerate(support_set):
            hv[idx] = weights[i]
        return hv

    # Sort support for consistent ordering
    support_sorted = sorted(support)
    weights = np.array([1.0/len(support_sorted)] * len(support_sorted))
    
    def eval_support_weights(support_list, w):
        hv = np.zeros(N)
        for idx, wt in zip(support_list, w):
            hv[idx] = wt
        return ratio(hv)

    current_w = weights.copy()
    current_support = support_sorted
    current_r = best_r

    # Iterative local search: perturb weights, try support changes
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        # 1. Coordinate ascent on weights (fix support)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n_s = len(current_support)
            if n_s == 0:
                break
            for i in range(n_s):
                # Try a few perturbations of weight i
                base = current_w[i]
                if base == 0:
                    continue
                for factor in [0.7, 0.85, 0.95, 1.05, 1.15, 1.3]:
                    new_w = current_w.copy()
                    new_w[i] = base * factor
                    # Normalize to keep L1 fixed (ratio is scale-invariant)
                    if np.sum(new_w) > 0:
                        new_w = new_w / np.sum(new_w) * np.sum(current_w)
                    r = eval_support_weights(current_support, new_w)
                    if r > current_r + 1e-12:
                        current_w = new_w
                        current_r = r
                        improved = True
                        break
        
        # 2. Try adding a new element to support
        if time.time() > deadline:
            break
        candidates = [i for i in range(N) if i not in current_support]
        random.shuffle(candidates)
        for cand in candidates[:20]:  # limit for speed
            new_support = current_support + [cand]
            new_support.sort()
            new_w = np.ones(len(new_support)) / len(new_support)
            r = eval_support_weights(new_support, new_w)
            if r > current_r + 1e-12:
                current_support = new_support
                current_w = new_w
                current_r = r
                break
        
        # 3. Try removing an element from support
        if len(current_support) > 1 and time.time() < deadline:
            for i in range(len(current_support)):
                new_support = [x for j, x in enumerate(current_support) if j != i]
                new_w = np.ones(len(new_support)) / len(new_support)
                r = eval_support_weights(new_support, new_w)
                if r > current_r + 1e-12:
                    current_support = new_support
                    current_w = new_w
                    current_r = r
                    break

        # Update best
        if current_r > best_r:
            best_r = current_r
            best_h = np.zeros(N)
            for idx, wt in zip(current_support, current_w):
                best_h[idx] = wt

        # Occasional random restart with new support
        if iteration % 50 == 0 and time.time() < deadline - 30:
            # Random support of size 10-30
            sz = np.random.randint(10, min(31, N))
            rs = sorted(np.random.choice(N, sz, replace=False).tolist())
            rw = np.ones(sz) / sz
            rr = eval_support_weights(rs, rw)
            if rr > current_r:
                current_support = rs
                current_w = rw
                current_r = rr

    # Final check
    if best_r <= 0:
        # Fallback
        best_h = np.ones(N) / N
        best_r = ratio(best_h)

    return list(best_h)
# EVOLVE-BLOCK-END

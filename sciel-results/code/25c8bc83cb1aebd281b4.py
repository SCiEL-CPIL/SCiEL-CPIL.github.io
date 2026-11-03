import numpy as np
import time

N = 200

def _score(P: np.ndarray) -> float:
    # Full O(N^2) score, used sparingly for final checks / best tracking
    diffs = P[:, None, :] - P[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diffs, diffs)
    np.fill_diagonal(d2, np.inf)
    min_pair = float(np.sqrt(np.min(d2)))
    # boundary distances
    b = np.stack([0.5 - P[:, 0], 0.5 + P[:, 0], 0.5 - P[:, 1], 0.5 + P[:, 1]], axis=1)
    min_bound = float(np.min(b))
    return min(min_pair, min_bound)

def _build_hex(m: int) -> np.ndarray:
    base = N // m
    rem = N % m
    row_counts = [base + 1] * rem + [base] * (m - rem)
    n_max = max(row_counts)
    dx_width = 1.0 / n_max
    dx_height = 1.0 / (1.0 + (m - 1) * np.sqrt(3) / 2.0)
    dx = min(dx_width, dx_height)
    dy = dx * np.sqrt(3) / 2.0
    centers = []
    y_start = -((m - 1) * dy) / 2.0
    for j, n in enumerate(row_counts):
        y = y_start + j * dy
        x_start = -((n - 1) * dx) / 2.0
        for i in range(n):
            x = x_start + i * dx
            centers.append([x, y])
    return np.array(centers, dtype=float)

def _nearest_neighbors(P: np.ndarray, cell: float):
    # Build a simple grid and return for each point a list of nearby indices (excluding self)
    # P in [-0.5, 0.5]^2
    # cell should be > min_pair / 2 to capture close pairs; we'll pass something adaptive.
    xs = P[:, 0]
    ys = P[:, 1]
    # grid coords
    gx = ((xs + 0.5) / cell).astype(np.int64)
    gy = ((ys + 0.5) / cell).astype(np.int64)
    # map (gx, gy) -> list of indices
    from collections import defaultdict
    grid = defaultdict(list)
    for idx in range(P.shape[0]):
        grid[(gx[idx], gy[idx])].append(idx)
    nn = [[] for _ in range(P.shape[0])]
    for idx in range(P.shape[0]):
        cx, cy = gx[idx], gy[idx]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cx + dx, cy + dy), []):
                    if j != idx:
                        nn[idx].append(j)
    return nn

def _local_optimize(P: np.ndarray, budget_s: float) -> np.ndarray:
    # Local search maximizing min(min_pair_dist, min_bound_dist)
    # We maintain current best score and try to increase it.
    # Strategy:
    # - Maintain a spatial grid to find candidate close pairs.
    # - Each iteration:
    #     * Find current min pair and min bound (using full O(N^2) but N=200 is fine).
    #     * If min_pair < min_bound: pick the min pair (i,j). Try to move i and/or j apart.
    #     * Else: pick the point with min bound and try to move it inward.
    # - Use a small step size that adapts to the current bottleneck.
    # - To avoid getting stuck, occasionally do a random jitter + accept if improves.
    start = time.time()
    P = P.copy()
    best_score = _score(P)
    # step scale
    step = max(1e-4, best_score * 0.02)
    stall = 0
    rng = np.random.default_rng(12345)
    
    # Precompute a helper to evaluate only affected distances? For N=200, full is okay.
    # But we can speed up by checking only pairs involving moved point(s).
    # We'll implement a fast score that recomputes min_pair and min_bound fully each time.
    
    def fast_score(P):
        # same as _score but inline
        diffs = P[:, None, :] - P[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diffs, diffs)
        np.fill_diagonal(d2, np.inf)
        min_pair = float(np.sqrt(np.min(d2)))
        b = np.stack([0.5 - P[:, 0], 0.5 + P[:, 0], 0.5 - P[:, 1], 0.5 + P[:, 1]], axis=1)
        min_bound = float(np.min(b))
        return min(min_pair, min_bound)
    
    current_score = best_score
    
    max_iters = 200000
    for it in range(max_iters):
        if time.time() - start > budget_s:
            break
        # Find bottleneck
        diffs = P[:, None, :] - P[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diffs, diffs)
        np.fill_diagonal(d2, np.inf)
        min_pair_val = float(np.sqrt(np.min(d2)))
        b = np.stack([0.5 - P[:, 0], 0.5 + P[:, 0], 0.5 - P[:, 1], 0.5 + P[:, 1]], axis=1)
        min_bound_val = float(np.min(b))
        
        if min_pair_val <= min_bound_val:
            # Move a close pair apart
            ii = int(np.argmin(d2))
            i, j = np.unravel_index(ii, d2.shape)
            if i == j:
                continue
            vec = P[i] - P[j]
            norm = float(np.linalg.norm(vec))
            if norm < 1e-12:
                dir_vec = rng.standard_normal(2)
                dir_vec /= np.linalg.norm(dir_vec)
            else:
                dir_vec = vec / norm
            # Try moving i in +dir, j in -dir
            moved = False
            for trial in range(3):
                s = step * rng.uniform(0.5, 2.0)
                # Move i
                old_i = P[i].copy()
                P[i] = old_i + dir_vec * s
                ns = fast_score(P)
                if ns > current_score:
                    current_score = ns
                    if ns > best_score:
                        best_score = ns
                        stall = 0
                    else:
                        stall += 1
                    moved = True
                    break
                P[i] = old_i
                # Move j
                old_j = P[j].copy()
                P[j] = old_j - dir_vec * s
                ns = fast_score(P)
                if ns > current_score:
                    current_score = ns
                    if ns > best_score:
                        best_score = ns
                        stall = 0
                    else:
                        stall += 1
                    moved = True
                    break
                P[j] = old_j
            if not moved:
                # Try a random small move on i or j
                for k in (i, j):
                    old_k = P[k].copy()
                    P[k] = old_k + rng.standard_normal(2) * step * 0.3
                    ns = fast_score(P)
                    if ns > current_score:
                        current_score = ns
                        if ns > best_score:
                            best_score = ns
                            stall = 0
                        else:
                            stall += 1
                        break
                    P[k] = old_k
                else:
                    stall += 1
        else:
            # Move the point closest to a boundary inward
            bvals = np.stack([0.5 - P[:, 0], 0.5 + P[:, 0], 0.5 - P[:, 1], 0.5 + P[:, 1]], axis=1)
            i = int(np.argmin(bvals.min(axis=1)))
            # Determine which boundary is min for this point
            bl = 0.5 - P[i, 0]
            br = 0.5 + P[i, 0]
            bb = 0.5 - P[i, 1]
            bt = 0.5 + P[i, 1]
            m = min(bl, br, bb, bt)
            if m == bl:
                dir_vec = np.array([1.0, 0.0])
            elif m == br:
                dir_vec = np.array([-1.0, 0.0])
            elif m == bb:
                dir_vec = np.array([0.0, 1.0])
            else:
                dir_vec = np.array([0.0, -1.0])
            moved = False
            for trial in range(3):
                s = step * rng.uniform(0.5, 2.0)
                old_i = P[i].copy()
                P[i] = old_i + dir_vec * s
                ns = fast_score(P)
                if ns > current_score:
                    current_score = ns
                    if ns > best_score:
                        best_score = ns
                        stall = 0
                    else:
                        stall += 1
                    moved = True
                    break
                P[i] = old_i
            if not moved:
                # random jitter
                old_i = P[i].copy()
                P[i] = old_i + rng.standard_normal(2) * step * 0.3
                ns = fast_score(P)
                if ns > current_score:
                    current_score = ns
                    if ns > best_score:
                        best_score = ns
                        stall = 0
                    else:
                        stall += 1
                else:
                    P[i] = old_i
                    stall += 1
        
        # Adapt step
        if stall > 20:
            step *= 0.8
            stall = 0
        elif best_score > current_score * 1.001:
            # we improved best recently, maybe increase step a bit
            step = min(step * 1.05, best_score * 0.05)
    
    return P

def construct():
    # Try several hex grids and local optimize each, keep best
    start = time.time()
    max_time = 110.0
    best_P = None
    best_score = -1.0
    
    # Choose a few m values that are promising
    m_candidates = [13, 14, 15, 16, 17, 18]
    # Also try a slightly perturbed grid for each m
    for m in m_candidates:
        if time.time() - start > max_time:
            break
        P0 = _build_hex(m)
        # Add tiny deterministic jitter to break symmetry
        rng = np.random.default_rng(1000 + m)
        P0 = P0 + rng.standard_normal(P0.shape) * 1e-4
        # Clip to stay inside
        P0[:, 0] = np.clip(P0[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
        P0[:, 1] = np.clip(P0[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
        s0 = _score(P0)
        if s0 > best_score:
            best_score = s0
            best_P = P0.copy()
        # Local optimize with remaining budget
        rem = max_time - (time.time() - start)
        if rem > 5:
            P_opt = _local_optimize(P0, rem * 0.8)
            s_opt = _score(P_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_P = P_opt.copy()
    
    # If we still have time, do a final polish on the best
    rem = max_time - (time.time() - start)
    if rem > 10 and best_P is not None:
        P_final = _local_optimize(best_P, rem)
        s_final = _score(P_final)
        if s_final > best_score:
            best_P = P_final
            best_score = s_final
    
    # Ensure finite and inside
    if best_P is None:
        # fallback: simple grid
        best_P = _build_hex(14)
    
    # Final safety: clip and ensure non-overlap? We can't guarantee, but clip.
    best_P[:, 0] = np.clip(best_P[:, 0], -0.5, 0.5)
    best_P[:, 1] = np.clip(best_P[:, 1], -0.5, 0.5)
    # Remove any non-finite
    if not np.all(np.isfinite(best_P)):
        best_P = _build_hex(14)
    
    return best_P

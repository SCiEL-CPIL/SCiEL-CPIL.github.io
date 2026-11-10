# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 15.0

def _count_convex7_fast(points):
    """
    Count convex 7-gons using vectorized dynamic programming.
    """
    n = len(points)
    order = np.lexsort((points[:, 1], points[:, 0]))
    p_sorted = points[order]

    total = 0

    for i in range(n - 6):
        p = p_sorted[i]
        later = p_sorted[i + 1:]
        m = len(later)
        if m < 6:
            continue

        dx = later[:, 0] - p[0]
        dy = later[:, 1] - p[1]
        angles = np.arctan2(dy, dx)
        idx = np.argsort(angles)
        q = later[idx]

        # Precompute turns vectorized
        # turn[k, p_idx, c] = (q[p_idx]-q[k]) x (q[c]-q[p_idx])
        # dx1[k, p_idx] = q[p_idx, 0] - q[k, 0]
        # dy1[k, p_idx] = q[p_idx, 1] - q[k, 1]
        # dx2[p_idx, c] = q[c, 0] - q[p_idx, 0]
        # dy2[p_idx, c] = q[c, 1] - q[p_idx, 1]
        
        q0 = q[:, 0]
        q1 = q[:, 1]
        
        # dx1 shape (m, m) [k, p_idx]
        dx1 = q0[None, :] - q0[:, None]
        dy1 = q1[None, :] - q1[:, None]
        
        # dx2 shape (m, m) [p_idx, c]
        dx2 = q0[None, :] - q0[:, None] # Wait, q[c] - q[p_idx]
        # dx2[p_idx, c] = q0[c] - q0[p_idx]
        dx2 = q0[None, :] - q0[:, None] # This is q[c] - q[p_idx] if row is p_idx?
        # q0[None, :] is (1, m) q[c]
        # q0[:, None] is (m, 1) q[p_idx]
        # Result (m, m) [p_idx, c] = q0[c] - q0[p_idx]. Correct.
        
        dy2 = q1[None, :] - q1[:, None]
        
        # term1[k, p_idx, c] = dx1[k, p_idx] * dy2[p_idx, c]
        # term2[k, p_idx, c] = dy1[k, p_idx] * dx2[p_idx, c]
        
        term1 = dx1[:, :, None] * dy2[None, :, :]
        term2 = dy1[:, :, None] * dx2[None, :, :]
        turn = term1 - term2
        
        # Mask for k < p_idx
        k_idx = np.arange(m)
        lower_mask = (k_idx[:, None] < k_idx[None, :]) # (m, m) [k, p_idx]
        
        # valid_turns[k, p_idx, c] = (turn[k, p_idx, c] > 1e-9) and (k < p_idx)
        valid_turns = (turn > 1e-9) & (lower_mask[:, :, None])
        
        # turn_p[s, c] = (q[s]-p) x (q[c]-q[s])
        # v1 = q[s] - p
        # v2 = q[c] - q[s]
        dx_p = q0 - p[0]
        dy_p = q1 - p[1]
        
        # dx2_ps[s, c] = q[c, 0] - q[s, 0]
        # dy2_ps[s, c] = q[c, 1] - q[s, 1]
        dx2_ps = q0[None, :] - q0[:, None]
        dy2_ps = q1[None, :] - q1[:, None]
        
        turn_p = dx_p[:, None] * dy2_ps - dy_p[:, None] * dx2_ps
        
        # close1[p_idx, c] = (q[c]-q[p_idx]) x (p-q[c])
        # v1 = q[c] - q[p_idx]
        # v2 = p - q[c]
        # dx1_c = q[c, 0] - q[p_idx, 0]
        # dy1_c = q[c, 1] - q[p_idx, 1]
        # dx2_c = p[0] - q[c, 0]
        # dy2_c = p[1] - q[c, 1]
        
        close1_dx = q0[None, :] - q0[:, None] # [p_idx, c]
        close1_dy = q1[None, :] - q1[:, None]
        
        close2_dx = p[0] - q0
        close2_dy = p[1] - q1
        
        close1 = close1_dx * close2_dy[None, :] - close1_dy * close2_dx[None, :]
        
        # close2[c, s] = (p-q[c]) x (q[s]-p)
        # v1 = p - q[c]
        # v2 = q[s] - p
        # dx1 = p[0] - q[c, 0]
        # dy1 = p[1] - q[c, 1]
        # dx2 = q[s, 0] - p[0]
        # dy2 = q[s, 1] - p[1]
        
        close2_dx = p[0] - q0
        close2_dy = p[1] - q1
        
        close2_dx2 = q0 - p[0]
        close2_dy2 = q1 - p[1]
        
        close2 = close2_dx[:, None] * close2_dy2[None, :] - close2_dy[:, None] * close2_dx2[None, :]
        
        # DP
        dp = np.zeros((m, m, m), dtype=np.int64)
        
        # Init length 2
        # dp[s, s, c] = 1 if turn_p[s, c] > 1e-9
        # Note: s must be < c? The chain starts at s, goes to s (pivot to s), then to c.
        # The first step is p -> q[s]. The second is q[s] -> q[c].
        # So turn_p[s, c] is the turn at q[s].
        # We need s < c in the angular order?
        # The points are sorted by angle.
        # A convex chain p -> q[s] -> q[c] -> ... requires s < c in the sorted order?
        # Yes, because we are building a fan from p.
        init_mask = (turn_p > 1e-9)
        # dp[s, s, c]
        # We can do this with advanced indexing or broadcasting
        s_idx = np.arange(m)
        c_idx = np.arange(m)
        # We want to set dp[s, s, c] where init_mask[s, c] is True
        # Since s and c are independent in the mask, but we need s=c in the first two indices?
        # dp[s, s, c]
        # We can loop s, or use broadcasting.
        # dp[s, s, :] = init_mask[s, :] * (s < something?)
        # Actually, for the chain to be valid, we just need the turn to be convex.
        # The indices s and c must be such that the points are distinct.
        # Since they are from a set, s != c is guaranteed if we don't pick same point.
        # But in the array, s and c are indices.
        # We set dp[s, s, c] = 1 if turn_p[s, c] > 1e-9.
        
        for s in range(m):
            dp[s, s, :] = init_mask[s, :].astype(np.int64)
            
        # Extend
        for length in range(3, 7):
            new_dp = np.zeros((m, m, m), dtype=np.int64)
            # new_dp[s, p_idx, c] = sum_k dp[s, k, p_idx] * valid_turns[k, p_idx, c]
            # dp[s] is (m, m) [k, p_idx]
            # valid_turns is (m, m, m) [k, p_idx, c]
            
            # dp_s = dp[s]
            # prod = dp_s[:, :, None] * valid_turns
            # new_dp_s = sum(prod, axis=1)
            
            # We can do this for all s at once?
            # dp is (m, m, m) [s, k, p_idx]
            # We want to sum over k.
            # prod[s, k, p_idx, c] = dp[s, k, p_idx] * valid_turns[k, p_idx, c]
            # new_dp[s, p_idx, c] = sum_k prod[s, k, p_idx, c]
            
            # dp_expanded = dp[:, :, :, None] # (m, m, m, 1)
            # vt_expanded = valid_turns[None, :, :, :] # (1, m, m, m)
            # prod = dp_expanded * vt_expanded # (m, m, m, m)
            # new_dp = prod.sum(axis=1) # (m, m, m)
            
            dp_expanded = dp[:, :, :, None]
            vt_expanded = valid_turns[None, :, :, :]
            prod = dp_expanded * vt_expanded
            new_dp = prod.sum(axis=1)
            
            dp = new_dp
            
        # Final sum
        # total += sum_{s, p_idx, c} dp[s, p_idx, c] * (close1[p_idx, c] > 1e-9) * (close2[c, s] > 1e-9)
        
        c1_mask = (close1 > 1e-9)
        c2_mask = (close2 > 1e-9)
        
        # FinalMask[s, p_idx, c] = c1_mask[p_idx, c] * c2_mask[c, s]
        # c1_mask is (m, m) [p_idx, c]
        # c2_mask is (m, m) [c, s]
        
        # We need to align dimensions.
        # c1_mask[None, :, :] -> (1, m, m) [_, p_idx, c]
        # c2_mask[:, None, :]? No, c2_mask[c, s].
        # We want c2_mask[c, s] to be at [s, p_idx, c].
        # So we need to transpose c2_mask to [s, c]?
        # c2_mask is [c, s].
        # c2_mask.T is [s, c].
        # c2_mask.T[None, :, :] -> (1, m, m) [_, s, c]
        
        # We need to multiply c1_mask[None, :, :] and c2_mask.T[None, :, :]?
        # No, they have different axes.
        # c1_mask[p_idx, c]
        # c2_mask[c, s]
        
        # Let's construct FinalMask[s, p_idx, c]
        # c1_part = c1_mask[None, :, :] # (1, m, m) [_, p_idx, c]
        # c2_part = c2_mask.T[None, :, :] # (1, m, m) [_, s, c]
        # This doesn't work directly because we need to broadcast to (m, m, m).
        
        # c1_part shape (1, m, m)
        # c2_part shape (1, m, m)
        # We want FinalMask[s, p_idx, c] = c1_mask[p_idx, c] * c2_mask[c, s]
        
        # c1_mask[None, :, :] broadcasts to (m, m, m) as [s, p_idx, c] taking p_idx, c from axes 1, 2.
        # c2_mask.T[None, :, :] broadcasts to (m, m, m) as [s, p_idx, c] taking s, c from axes 0, 2.
        
        c1_b = c1_mask[None, :, :]
        c2_b = c2_mask.T[None, :, :]
        
        # c1_b[s, p_idx, c] = c1_mask[p_idx, c]
        # c2_b[s, p_idx, c] = c2_mask.T[s, c] = c2_mask[c, s]
        
        final_mask = c1_b * c2_b
        
        total += np.sum(dp * final_mask)

    return total

def _is_valid_fast(pts):
    if pts.shape != (33, 2):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts < -1e-6) or np.any(pts > 1 + 1e-6):
        return False

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(d2, 1.0)
    if np.any(d2 < 1e-12):
        return False

    n = 33
    # Vectorized collinearity check is tricky for all triples, 
    # but n=33 is small enough for a loop if optimized, or we can sample.
    # However, the problem requires ALL triples to be non-collinear.
    # We must check all.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i, 0] * (pts[j, 1] - pts[k, 1]) + \
                    pts[j, 0] * (pts[k, 1] - pts[i, 1]) + \
                    pts[k, 0] * (pts[i, 1] - pts[j, 1])
                if abs(a) < 1e-9:
                    return False
    return True

def _generate_sparse_grid(rng):
    """
    Generate points on a sparse high-symmetry grid, then perturb to break degeneracies.
    This follows the directive's suggestion to maximize initial collinearity/density.
    """
    # Use a 6x6 grid of candidate positions, pick 33 of them
    # Scale to [0.05, 0.95]
    grid_size = 6
    margin = 0.05
    span = 1.0 - 2 * margin
    coords = np.linspace(margin, 1 - margin, grid_size)
    
    # All 36 grid positions
    xx, yy = np.meshgrid(coords, coords)
    all_pts = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Randomly select 33 out of 36
    selected = rng.choice(36, size=33, replace=False)
    pts = all_pts[selected].copy()
    
    # Add small perturbations to break exact collinearities
    pts += rng.normal(0, 0.008, (33, 2))
    pts = np.clip(pts, 0.01, 0.99)
    
    # Ensure validity
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 100:
        pts += rng.normal(0, 0.003, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    
    return pts

def _generate_corner_cluster(rng):
    """
    Generate points clustered near the four corners.
    4 corners * 8 points = 32, + 1 center = 33.
    """
    pts = []
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for cx, cy in corners:
        for _ in range(8):
            x = cx + rng.normal(0, 0.03)
            y = cy + rng.normal(0, 0.03)
            pts.append((x, y))
    # 1 point in center
    x = 0.5 + rng.normal(0, 0.02)
    y = 0.5 + rng.normal(0, 0.02)
    pts.append((x, y))
    
    pts = np.array(pts)
    pts = np.clip(pts, 0.01, 0.99)
    
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 100:
        pts += rng.normal(0, 0.003, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    
    return pts

def _generate_clusters(rng):
    """
    Generate points in 4 corner clusters.
    """
    pts = []
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for cx, cy in corners:
        for _ in range(8):  # 32 points
            x = cx + rng.normal(0, 0.03)
            y = cy + rng.normal(0, 0.03)
            pts.append((x, y))
    # 1 point in center
    x = 0.5 + rng.normal(0, 0.02)
    y = 0.5 + rng.normal(0, 0.02)
    pts.append((x, y))
    
    pts = np.array(pts)
    pts = np.clip(pts, 0.01, 0.99)
    
    attempts = 0
    while not _is_valid_fast(pts) and attempts < 50:
        pts += rng.normal(0, 0.005, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
    return pts

def _hill_climb(pts, rng, time_limit):
    """
    Perform aggressive hill climbing to minimize the number of convex 7-gons.
    Uses adaptive step sizes and occasional multi-point moves.
    """
    best_pts = pts.copy()
    best_c = _count_convex7_fast(best_pts)
    start_time = time.time()
    
    no_improve = 0
    max_no_improve = 500
    step_size = 0.08
    best_step = step_size
    
    while time.time() - start_time < time_limit and no_improve < max_no_improve:
        improved = False
        
        # Try moving 1 point
        i = rng.integers(0, 33)
        new_pts = best_pts.copy()
        
        # Occasionally jump to a random location to escape local minima
        if rng.random() < 0.05:
            new_pts[i] = rng.random(2) * 0.84 + 0.08
        else:
            offset = rng.normal(0, step_size, 2)
            new_pts[i] = best_pts[i] + offset
            new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
        
        if _is_valid_fast(new_pts):
            c = _count_convex7_fast(new_pts)
            if c < best_c:
                best_c = c
                best_pts = new_pts
                improved = True
                no_improve = 0
                step_size = max(0.01, step_size * 0.97)
            elif c == best_c and best_c > 1:
                # Accept neutral moves to explore
                if rng.random() < 0.2:
                    best_pts = new_pts
                    no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # If stuck, try moving 2-3 points simultaneously
        if not improved and no_improve % 5 == 0:
            n_move = int(rng.integers(2, 4))
            idx = rng.choice(33, size=n_move, replace=False)
            new_pts = best_pts.copy()
            for i in idx:
                offset = rng.normal(0, step_size * 1.5, 2)
                new_pts[i] = best_pts[i] + offset
                new_pts[i] = np.clip(new_pts[i], 0.01, 0.99)
            if _is_valid_fast(new_pts):
                c = _count_convex7_fast(new_pts)
                if c < best_c:
                    best_c = c
                    best_pts = new_pts
                    no_improve = 0
                    step_size = max(0.01, step_size * 0.95)
        
        # Adapt step size when stuck
        if no_improve % 30 == 0:
            if no_improve > 100:
                step_size = min(0.25, step_size * 1.3)
            else:
                step_size = max(0.01, step_size * 0.9)
    
    return best_pts, best_c

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)

    # Generate initial points using multiple strategies
    candidates = []
    
    # Strategy 1: Sparse grid (directive-inspired)
    pts1 = _generate_sparse_grid(rng)
    c1 = _count_convex7_fast(pts1)
    candidates.append((c1, pts1))
    
    # Strategy 2: Corner/edge clusters
    pts2 = _generate_corner_cluster(rng)
    c2 = _count_convex7_fast(pts2)
    candidates.append((c2, pts2))
    
    # Strategy 3: Original clusters
    pts3 = _generate_clusters(rng)
    c3 = _count_convex7_fast(pts3)
    candidates.append((c3, pts3))
    
    # Pick the best candidate
    best_c, best_pts = min(candidates, key=lambda x: x[0])
    
    # Perform hill climbing on the best candidate
    while time_left() > 20:
        # Run hill climb for a chunk of time
        chunk_time = min(45, time_left() - 20)
        best_pts, best_c = _hill_climb(best_pts, rng, chunk_time)
        
        if best_c <= 1:
            break
        
        # If we have time, try a restart from a slightly perturbed version
        if time_left() > 40 and rng.random() < 0.15:
            perturbed = best_pts + rng.normal(0, 0.04, (33, 2))
            perturbed = np.clip(perturbed, 0.01, 0.99)
            if _is_valid_fast(perturbed):
                c_pert = _count_convex7_fast(perturbed)
                if c_pert < best_c:
                    best_c = c_pert
                    best_pts = perturbed
    
    return best_pts
# EVOLVE-BLOCK-END

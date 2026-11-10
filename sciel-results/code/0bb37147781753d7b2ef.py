# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 630.0
_SAFE_MARGIN = 10.0

def _count_convex7_fast(points):
    """
    Count convex 7-gons using dynamic programming.
    Points are sorted lexicographically, for each pivot the later points are
    sorted by atan2 angle around it, and a turn p -> q -> r counts as convex
    only if (qx-px)(ry-qy) - (qy-py)(rx-qx) > 1e-9.
    """
    n = len(points)
    if n < 7:
        return 0
        
    order = np.lexsort((points[:, 1], points[:, 0]))
    p_sorted = points[order]

    total = 0
    TOL = 1e-9

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

        # Precompute turn(a,b,c) for all a<b<c
        # turn_ok[a,b,c] = (q[b]-q[a]) x (q[c]-q[b]) > TOL
        turn_ok = np.zeros((m, m, m), dtype=bool)
        for a in range(m):
            for b in range(a + 1, m):
                v1x = q[b, 0] - q[a, 0]
                v1y = q[b, 1] - q[a, 1]
                v2x = q[:, 0] - q[b, 0]
                v2y = q[:, 1] - q[b, 1]
                cross_vals = v1x * v2y - v1y * v2x
                turn_ok[a, b, b+1:] = cross_vals[b+1:] > TOL

        # Precompute turn from pivot p to q[s] to q[c]
        turn_p = np.zeros((m, m), dtype=bool)
        for s in range(m):
            v1x = q[s, 0] - p[0]
            v1y = q[s, 1] - p[1]
            v2x = q[:, 0] - q[s, 0]
            v2y = q[:, 1] - q[s, 1]
            cross_vals = v1x * v2y - v1y * v2x
            turn_p[s, s+1:] = cross_vals[s+1:] > TOL

        # dp[l][s][p_idx][c_idx] = number of convex chains of length l
        # starting at index s, with previous point at p_idx, ending at c_idx
        # We only need l=2 to 6.
        # dp_prev[s, s, c] for l=2
        dp_prev = np.zeros((m, m, m), dtype=np.int64)
        dp_curr = np.zeros((m, m, m), dtype=np.int64)

        # Initialize dp[2]
        for s in range(m):
            valid_c = np.where(turn_p[s, s+1:])[0]
            if len(valid_c) > 0:
                dp_prev[s, s, s+1:][valid_c] = 1

        # Fill dp[3..6]
        for l in range(3, 7):
            dp_curr.fill(0)
            for s in range(m):
                for p_idx in range(s + 1, m):
                    # Check if any chain of length l-1 ends at p_idx starting from s
                    # dp_prev[s, s:p_idx, p_idx]
                    slice_prev = dp_prev[s, s:p_idx, p_idx]
                    if not np.any(slice_prev > 0):
                        continue
                    
                    # We need to sum over k in [s, p_idx-1]
                    # dp_curr[s, p_idx, c_idx] += dp_prev[s, k, p_idx] * turn_ok[k, p_idx, c_idx]
                    # turn_ok slice: [s:p_idx, p_idx, p_idx+1:]
                    turn_slice = turn_ok[s:p_idx, p_idx, p_idx+1:]
                    
                    # Vectorized sum: (dp_slice[:, None] * turn_slice).sum(axis=0)
                    contrib = (slice_prev[:, np.newaxis] * turn_slice).sum(axis=0)
                    dp_curr[s, p_idx, p_idx+1:] = contrib
            
            dp_prev, dp_curr = dp_curr, dp_prev

        # Sum up results for l=6
        # We need to check closing turns for chains of length 6
        # Chain: p -> q[s] -> ... -> q[p_idx] -> q[c_idx]
        # Closing 1: q[p_idx] -> q[c_idx] -> p
        # Closing 2: q[c_idx] -> p -> q[s]
        
        for s in range(m):
            for p_idx in range(s + 1, m):
                # Get counts for chains ending at p_idx, starting at s, length 6
                # dp_prev[s, s:p_idx, p_idx] contains counts for chains of length 6?
                # No, dp_prev is length 6 in the last iteration.
                # The indices are: dp_prev[start, prev, curr]
                # So for fixed s, p_idx, the counts are in dp_prev[s, s:p_idx, p_idx]
                
                slice_counts = dp_prev[s, s:p_idx, p_idx]
                if not np.any(slice_counts > 0):
                    continue
                
                # We need to sum over c_idx > p_idx
                # For each c_idx, check closing conditions
                # closing1: (q[p_idx], q[c_idx], p)
                # closing2: (q[c_idx], p, q[s])
                
                # Vectorize over c_idx
                c_indices = np.arange(p_idx + 1, m)
                if len(c_indices) == 0:
                    continue
                
                # v1 for closing1: q[c] - q[p_idx]
                v1x = q[c_indices, 0] - q[p_idx, 0]
                v1y = q[c_indices, 1] - q[p_idx, 1]
                # v2 for closing1: p - q[c]
                v2x = p[0] - q[c_indices, 0]
                v2y = p[1] - q[c_indices, 1]
                cross1 = v1x * v2y - v1y * v2x
                ok1 = cross1 > TOL
                
                # v1 for closing2: p - q[c]
                # v1x is same as v2x above? No.
                # closing2: (q[c], p, q[s])
                # v1 = p - q[c]
                # v2 = q[s] - p
                v2x_closing2 = p[0] - q[c_indices, 0]
                v2y_closing2 = p[1] - q[c_indices, 1]
                v3x = q[s, 0] - p[0]
                v3y = q[s, 1] - p[1]
                cross2 = v2x_closing2 * v3y - v2y_closing2 * v3x
                ok2 = cross2 > TOL
                
                valid_c_mask = ok1 & ok2
                if not np.any(valid_c_mask):
                    continue
                
                # For each valid c_idx, we need to sum the counts from slice_counts
                # slice_counts[k] corresponds to chain ending at p_idx with prev=k
                # The count for a specific c_idx is sum(slice_counts) * (1 if valid else 0)?
                # No, the DP value dp[s, p_idx, c_idx] was computed as sum over k.
                # But here we are iterating over c_idx in the final step.
                # Wait, my DP loop filled dp_curr[s, p_idx, c_idx] for all c_idx.
                # So I should just look at dp_prev[s, p_idx, c_idx]?
                # No, dp_prev is 3D: [start, prev, curr].
                # In the last iteration (l=6), dp_prev[s, k, p_idx] is the count of chains of length 6 ending at p_idx with prev k.
                # But I need chains of length 6 ending at c_idx?
                # Let's re-read the DP definition.
                # dp[l][s][p_idx][c_idx] = number of chains of length l starting at s, ending at c_idx, with prev p_idx.
                # My code: dp_prev[s, s:p_idx, p_idx] ... this is confusing.
                
                # Let's trace:
                # l=2: dp_prev[s, s, c] = 1 if turn(p, q[s], q[c]) ok.
                # l=3: dp_curr[s, p_idx, c_idx] = sum_{k < p_idx} dp_prev[s, k, p_idx] * turn_ok[k, p_idx, c_idx]
                #      Here dp_prev[s, k, p_idx] is count of chains of length 2 ending at p_idx with prev k.
                #      So dp_curr[s, p_idx, c_idx] is count of chains of length 3 ending at c_idx with prev p_idx.
                # ...
                # l=6: dp_prev[s, p_idx, c_idx] is count of chains of length 6 ending at c_idx with prev p_idx.
                
                # So in the final sum, we iterate over s, p_idx, c_idx.
                # count = dp_prev[s, p_idx, c_idx]
                # Check closing turns for (s, p_idx, c_idx).
                
                # My previous loop structure was:
                # for s:
                #   for p_idx:
                #     for c_idx:
                #        count = dp_prev[s, p_idx, c_idx]
                
                # Let's rewrite the final sum part correctly.
                
                # The vectorization above was for a specific s, p_idx.
                # We need to sum over c_idx.
                # The count for a specific c_idx is dp_prev[s, p_idx, c_idx].
                
                # So:
                counts_for_c = dp_prev[s, p_idx, c_indices]
                total += np.sum(counts_for_c[valid_c_mask])

    return total

def _is_fast_valid(pts):
    """Check bounds and pairwise distance. Skips collinearity."""
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

    return True

def _is_full_valid(pts):
    """Check bounds, distance, and collinearity."""
    if not _is_fast_valid(pts):
        return False
    n = 33
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = pts[i, 0] * (pts[j, 1] - pts[k, 1]) + \
                    pts[j, 0] * (pts[k, 1] - pts[i, 1]) + \
                    pts[k, 0] * (pts[i, 1] - pts[j, 1])
                if abs(a) < 1e-9:
                    return False
    return True

def _generate_initial_points(rng):
    """
    Generate points in dense clusters near corners and center.
    """
    pts = []
    centers = [
        (0.05, 0.05), (0.95, 0.05), (0.05, 0.95), (0.95, 0.95),
        (0.5, 0.5)
    ]
    
    # 7 per corner, 5 in center
    for i, (cx, cy) in enumerate(centers):
        n_pts = 7 if i < 4 else 5
        for _ in range(n_pts):
            x = cx + rng.normal(0, 0.01)
            y = cy + rng.normal(0, 0.01)
            pts.append((x, y))
    
    pts = np.array(pts)
    pts = np.clip(pts, 0.01, 0.99)
    
    # Ensure fast validity
    attempts = 0
    while not _is_fast_valid(pts) and attempts < 100:
        pts += rng.normal(0, 0.001, (33, 2))
        pts = np.clip(pts, 0.01, 0.99)
        attempts += 1
        
    return pts

def erdos_szekeres_problem7() -> np.ndarray:
    start_time = time.time()
    rng = np.random.default_rng(42)

    def time_left():
        return _DEADLINE - _SAFE_MARGIN - (time.time() - start_time)

    # Initial configuration
    best_pts = _generate_initial_points(rng)
    best_c = _count_convex7_fast(best_pts)
    
    print(f"Initial C: {best_c}")

    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    alpha = 0.995
    iterations_per_temp = 50
    
    current_pts = best_pts.copy()
    current_c = best_c
    
    while time_left() > 5:
        if T < T_min:
            T = T_min
            
        for _ in range(iterations_per_temp):
            if time_left() < 2:
                break
                
            # Propose move
            i = rng.integers(0, 33)
            new_pts = current_pts.copy()
            offset = rng.normal(0, 0.02, 2)
            new_pts[i] = current_pts[i] + offset
            new_pts[i] = np.clip(new_pts[i], 0.005, 0.995)
            
            # Fast validity check
            if not _is_fast_valid(new_pts):
                continue
                
            new_c = _count_convex7_fast(new_pts)
            
            delta = new_c - current_c
            
            if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
                current_pts = new_pts
                current_c = new_c
                
                if current_c < best_c:
                    best_c = current_c
                    best_pts = current_pts.copy()
                    if best_c == 0:
                        print("Found 0 convex 7-gons!")
                        return best_pts
                        
        T *= alpha
        if T % 0.1 < 0.01: # Print occasionally
             print(f"T: {T:.4f}, Best C: {best_c}, Current C: {current_c}, Time: {time.time()-start_time:.1f}")

    # Final validity check and fix
    if not _is_full_valid(best_pts):
        print("Final points invalid, fixing...")
        attempts = 0
        while not _is_full_valid(best_pts) and attempts < 1000:
            # Small perturbations
            best_pts += rng.normal(0, 0.0001, (33, 2))
            best_pts = np.clip(best_pts, 0.005, 0.995)
            attempts += 1
        if not _is_full_valid(best_pts):
             # Last resort: regenerate
             best_pts = _generate_initial_points(rng)
             while not _is_full_valid(best_pts):
                 best_pts += rng.normal(0, 0.001, (33, 2))
                 best_pts = np.clip(best_pts, 0.01, 0.99)

    return best_pts
# EVOLVE-BLOCK-END

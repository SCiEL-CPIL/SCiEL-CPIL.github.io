import numpy as np
import time
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 15.0
    N = 16

    def get_ratio(pts):
        """Calculate d_min / d_max. Higher is better."""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max <= 1e-12:
            return 0.0
        return d_min / d_max

    def get_ratio_sq(pts):
        """Calculate (d_min / d_max)^2. Higher is better. This is the actual metric R."""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max <= 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def ring_init(n_inner, n_outer, r_inner, r_outer):
        """Points on two concentric rings."""
        pts = []
        if n_inner > 0:
            for i in range(n_inner):
                angle = 2 * np.pi * i / n_inner
                pts.append([r_inner * np.cos(angle), r_inner * np.sin(angle)])
        if n_outer > 0:
            for i in range(n_outer):
                angle = 2 * np.pi * i / n_outer + np.pi / n_outer
                pts.append([r_outer * np.cos(angle), r_outer * np.sin(angle)])
        return np.array(pts)

    def grid_init():
        """4x4 square grid."""
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i, j])
        pts = np.array(pts, dtype=float)
        pts = pts - np.mean(pts, axis=0)
        return pts

    def hex_init():
        """Hexagonal arrangement."""
        pts = []
        a = 1.0
        b = np.sqrt(3) / 2
        pts.append([0, 0])
        for i in range(6):
            angle = 2 * np.pi * i / 6
            pts.append([a * np.cos(angle), a * np.sin(angle)])
        ring2_pts = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * a + j * 0.5 * a
                y = j * b
                r = np.sqrt(x*x + y*y)
                if 1.5 * a - 0.1 < r < 2.5 * a + 0.1:
                    ring2_pts.append([x, y])
        ring2_pts = np.array(ring2_pts)
        if len(ring2_pts) >= 9:
            r_target = 2.0 * a
            dists_to_target = np.abs(np.linalg.norm(ring2_pts, axis=1) - r_target)
            idx = np.argsort(dists_to_target)[:9]
            ring2_pts = ring2_pts[idx]
        pts = np.array(pts)
        pts = np.vstack([pts, ring2_pts])
        pts = pts - np.mean(pts, axis=0)
        return pts

    def circle_init(n=16):
        """Points on a circle."""
        pts = []
        for i in range(n):
            angle = 2 * np.pi * i / n
            pts.append([np.cos(angle), np.sin(angle)])
        return np.array(pts)

    def square_ring_init():
        """Points on boundary of a square with some interior."""
        pts = []
        for i in range(4):
            pts.append([-1 + 2*i/3, 1.0])
        for i in range(1, 4):
            pts.append([1.0, 1 - 2*i/3])
        for i in range(1, 4):
            pts.append([-1 + 2*i/3, -1.0])
        for i in range(1, 3):
            pts.append([-1.0, 1 - 2*i/3])
        pts.append([0, 0])
        pts.append([0.5, 0.5])
        pts.append([-0.5, 0.5])
        pts.append([0.5, -0.5])
        pts = np.array(pts[:16], dtype=float)
        pts = pts - np.mean(pts, axis=0)
        return pts

    def generate_candidates():
        candidates = []
        candidates.append(ring_init(1, 15, 0, 1.0))
        candidates.append(ring_init(4, 12, 0.5, 1.0))
        candidates.append(ring_init(7, 9, 0.4, 1.0))
        candidates.append(ring_init(9, 7, 0.4, 1.0))
        candidates.append(ring_init(1, 3, 0, 0.5))
        for i, c in enumerate(candidates):
            c = c - np.mean(c, axis=0)
            max_r = np.max(np.linalg.norm(c, axis=1))
            if max_r > 0:
                c = c / max_r
            candidates[i] = c
        candidates.append(grid_init())
        candidates.append(hex_init())
        candidates.append(circle_init(16))
        candidates.append(square_ring_init())
        return candidates

    best_pts = None
    best_ratio = 0.0

    for c in generate_candidates():
        if len(c) != 16:
            continue
        ratio = get_ratio(c)
        if ratio > best_ratio:
            best_ratio = ratio
            best_pts = c.copy()

    # Sequential Convex Programming approach:
    # Minimize max pairwise distance subject to min pairwise distance >= lower_bound
    # Use bisection on lower_bound to find optimal packing density
    
    def scp_optimize(pts, time_budget=300.0):
        """
        Sequential Convex Programming: 
        1. Start with a lower bound on min distance
        2. Minimize max distance subject to min distance >= bound
        3. Bisection search on the bound to find optimal
        """
        pts = pts.copy()
        t_start = time.time()
        
        # Initial bounds for bisection
        d = pdist(pts)
        d_min_init = np.min(d)
        d_max_init = np.max(d)
        
        # We'll search for the optimal min_distance bound
        # The ratio is d_min/d_max, so we want to maximize this
        # Let's parameterize by d_min_bound and minimize d_max subject to d_min >= d_min_bound
        
        # First, get a good initial solution using coordinate descent
        cd_pts, cd_ratio = coordinate_descent(pts, max_iter=5000, time_budget=min(60.0, time_budget/4))
        
        d = pdist(cd_pts)
        d_min = np.min(d)
        d_max = np.max(d)
        
        # Bisection search on d_min_bound
        # We want to find the largest d_min_bound such that we can achieve d_max/d_min_bound <= some target
        # Actually, let's just directly optimize using SCP
        
        # SCP approach: 
        # Objective: minimize max_{i,j} ||p_i - p_j||^2
        # Constraint: min_{i,j} ||p_i - p_j||^2 >= t^2
        # We can linearize the constraints iteratively
        
        # For simplicity, let's use a penalty-based SCP
        # Minimize: max_dist + lambda * max(0, t - min_dist)
        
        best_sc = cd_pts.copy()
        best_sc_ratio = cd_ratio
        
        # Try different penalty weights and target distances
        for target_ratio in [cd_ratio * 0.9, cd_ratio, cd_ratio * 1.05, cd_ratio * 1.1]:
            if time.time() - t_start > time_budget:
                break
            
            t_target = target_ratio * d_max_init
            
            # SCP iterations
            current_pts = cd_pts.copy()
            lam = 10.0  # Penalty weight
            
            for it in range(200):
                if time.time() - t_start > time_budget:
                    break
                
                # Compute distances
                d_mat = pdist(current_pts, metric='euclidean')
                # Reconstruct full distance matrix
                n = N
                D = np.zeros((n, n))
                idx = np.triu_indices(n, k=1)
                D[idx] = d_mat
                D = D + D.T
                
                d_min_cur = np.min(D)
                d_max_cur = np.max(D)
                
                # Penalty for violating min distance constraint
                violation = max(0, t_target - d_min_cur)
                
                # Gradient of max distance w.r.t. each point
                # The max distance pair
                max_idx = np.unravel_index(np.argmax(D), D.shape)
                i_max, j_max = max_idx
                
                # Gradient of max distance
                grad_max = np.zeros((n, 2))
                grad_max[i_max] = 2 * (current_pts[i_max] - current_pts[j_max])
                grad_max[j_max] = 2 * (current_pts[j_max] - current_pts[i_max])
                
                # Gradient of min distance (if violated)
                if violation > 0:
                    min_idx = np.unravel_index(np.argmin(D), D.shape)
                    i_min, j_min = min_idx
                    grad_min = np.zeros((n, 2))
                    grad_min[i_min] = -2 * (current_pts[i_min] - current_pts[j_min])
                    grad_min[j_min] = -2 * (current_pts[j_min] - current_pts[i_min])
                else:
                    grad_min = np.zeros((n, 2))
                
                # Total gradient
                grad = grad_max + lam * grad_min
                
                # Take a step in the negative gradient direction
                step_size = 0.01
                for k in range(n):
                    current_pts[k] -= step_size * grad[k]
                
                # Center the points
                current_pts = current_pts - np.mean(current_pts, axis=0)
                
                # Check improvement
                cur_ratio = get_ratio(current_pts)
                if cur_ratio > best_sc_ratio:
                    best_sc_ratio = cur_ratio
                    best_sc = current_pts.copy()
                
                # Adaptive step size
                if it % 10 == 0:
                    step_size *= 0.95
        
        return best_sc, best_sc_ratio

    # Coordinate descent for local refinement
    def coordinate_descent(pts, max_iter=5000, time_budget=30.0):
        """Improve points using coordinate descent."""
        pts = pts.copy()
        t_start = time.time()
        step = 0.05
        
        for it in range(max_iter):
            if time.time() - t_start > time_budget:
                break
            
            improved = False
            for i in range(N):
                best_pos = pts[i].copy()
                best_r = get_ratio(pts)
                
                # Try small perturbations
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        old = pts[i].copy()
                        pts[i] += np.array([dx, dy])
                        r = get_ratio(pts)
                        if r > best_r + 1e-10:
                            best_r = r
                            best_pos = pts[i].copy()
                            improved = True
                        pts[i] = old
                
                pts[i] = best_pos
            
            if not improved:
                step *= 0.9
                if step < 1e-6:
                    break
        
        return pts, get_ratio(pts)

    # Run SCP optimization from best candidate
    time_elapsed = time.time() - start_time
    scp_budget = min(400.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN)
    if scp_budget > 30:
        scp_pts, scp_ratio = scp_optimize(best_pts, time_budget=scp_budget)
        if scp_ratio > best_ratio:
            best_pts = scp_pts
            best_ratio = scp_ratio

    # Final polish with coordinate descent
    time_elapsed = time.time() - start_time
    if TIME_LIMIT - time_elapsed - SAFETY_MARGIN > 10:
        cd_pts, cd_ratio = coordinate_descent(best_pts, max_iter=20000,
                                               time_budget=min(60.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN))
        if cd_ratio > best_ratio:
            best_pts = cd_pts
            best_ratio = cd_ratio

    # Ensure finite and correct shape
    final_pts = best_pts
    if final_pts is None or not np.all(np.isfinite(final_pts)) or final_pts.shape != (16, 2):
        final_pts = grid_init()

    return final_pts

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

    def compute_grad_ratio(pts):
        """
        Compute gradient of (d_min/d_max)^2 w.r.t. point positions.
        Returns (value, grad) where grad has shape (16, 2).
        """
        n = pts.shape[0]
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        dist = np.sqrt(np.maximum(dist_sq, 1e-30))
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(dist_sq, np.inf)
        
        d_min = np.min(dist)
        d_max = np.max(dist)
        
        if d_max <= 1e-12:
            return 0.0, np.zeros_like(pts)
        
        ratio_sq = (d_min / d_max) ** 2
        
        # Find which pairs give min and max
        # For gradient, we need the pairs that achieve min and max
        min_pairs = np.argwhere(dist == d_min)
        max_pairs = np.argwhere(dist == d_max)
        
        # Gradient of d_min^2 / d_max^2
        # Let f = (d_min/d_max)^2 = d_min^2 / d_max^2
        # df/dx = (2*d_min/d_max^2) * (d/dx d_min) - (2*d_min^2/d_max^3) * (d/dx d_max)
        # d/dx d_min = (1/(2*d_min)) * d/dx d_min^2
        # d/dx d_max = (1/(2*d_max)) * d/dx d_max^2
        
        grad = np.zeros_like(pts)
        
        # Contribution from d_min term
        if len(min_pairs) > 0:
            # Average over all min pairs (use only first few for stability)
            for (i, j) in min_pairs[:10]:
                if i == j:
                    continue
                # d/dp_i d_ij^2 = 2*(p_i - p_j)
                # d/dp_j d_ij^2 = 2*(p_j - p_i)
                diff_ij = pts[i] - pts[j]
                # Gradient of d_min^2 w.r.t. p_i is 2*diff_ij, w.r.t. p_j is -2*diff_ij
                # Gradient of ratio w.r.t. p_i: (2*d_min/d_max^2) * (1/(2*d_min)) * 2*diff_ij = (1/d_max^2) * 2*diff_ij
                # Actually let's be more careful:
                # f = d_min^2 / d_max^2
                # df/dp_i = (2 * (p_i - p_j) / d_max^2) - (d_min^2 / d_max^3) * (2 * (p_i - p_j) / d_max)
                # = 2*(p_i-p_j) * [1/d_max^2 - d_min^2/d_max^4]
                # = 2*(p_i-p_j) * (d_max^2 - d_min^2) / d_max^4
                
                coeff = 2.0 * (d_max ** 2 - d_min ** 2) / (d_max ** 4)
                grad[i] += coeff * diff_ij
                grad[j] -= coeff * diff_ij
        
        # Contribution from d_max term
        if len(max_pairs) > 0:
            for (i, j) in max_pairs[:10]:
                if i == j:
                    continue
                diff_ij = pts[i] - pts[j]
                # df/dp_i from d_max term: -(d_min^2/d_max^3) * (1/(2*d_max)) * 2*diff_ij
                # = -d_min^2 * diff_ij / d_max^4
                coeff = -d_min ** 2 / (d_max ** 4)
                grad[i] += coeff * diff_ij
                grad[j] -= coeff * diff_ij
        
        return ratio_sq, grad

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

    # Gradient-based optimization with line search
    def gradient_optimize(pts, time_budget=400.0):
        """
        Improve points using gradient ascent with adaptive line search.
        Uses analytical gradient of (d_min/d_max)^2.
        """
        pts = pts.copy()
        t_start = time.time()
        
        current_val = get_ratio_sq(pts)
        best_val = current_val
        best_pts = pts.copy()
        
        lr = 0.05  # Learning rate
        min_lr = 1e-8
        max_lr = 0.5
        patience = 0
        max_patience = 1000
        no_improve = 0
        
        it = 0
        while time.time() - t_start < time_budget:
            it += 1
            
            # Check time limit
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Compute gradient
            val, grad = compute_grad_ratio(pts)
            
            if not np.all(np.isfinite(grad)):
                break
            
            # Normalize gradient for stability
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                # Converged
                break
            grad_normalized = grad / grad_norm
            
            # Line search: try to find a step that improves
            found_improve = False
            for trial in range(20):
                step_size = lr * (0.5 ** trial)
                if step_size < min_lr:
                    break
                
                new_pts = pts + step_size * grad_normalized
                
                # Quick check: ensure finite
                if not np.all(np.isfinite(new_pts)):
                    continue
                
                new_val = get_ratio_sq(new_pts)
                
                if new_val > val + 1e-12:
                    pts = new_pts
                    val = new_val
                    found_improve = True
                    no_improve = 0
                    
                    if val > best_val:
                        best_val = val
                        best_pts = pts.copy()
                    
                    # Increase learning rate if we found improvement
                    if trial == 0:
                        lr = min(lr * 1.2, max_lr)
                    break
                else:
                    # Try smaller step
                    continue
            
            if not found_improve:
                # No improvement found; reduce learning rate
                lr *= 0.5
                no_improve += 1
                
                if no_improve >= max_patience:
                    # Try a random restart with small perturbation
                    pts = pts + np.random.randn(N, 2) * 0.01
                    val = get_ratio_sq(pts)
                    no_improve = 0
                    lr = 0.05
                    patience = 0
                elif no_improve % 100 == 0:
                    # Periodic perturbation to escape local minima
                    pts = pts + np.random.randn(N, 2) * 0.005
                    val = get_ratio_sq(pts)
        
        return best_pts, best_val

    # Run gradient optimization from best candidate
    time_elapsed = time.time() - start_time
    grad_budget = min(450.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN)
    if grad_budget > 10:
        grad_pts, grad_ratio = gradient_optimize(best_pts, time_budget=grad_budget)
        if grad_ratio > best_ratio:
            best_pts = grad_pts
            best_ratio = grad_ratio

    # Final polish with a second gradient pass
    def gradient_polish(pts, time_budget=60.0):
        """Final gradient polish with smaller steps."""
        pts = pts.copy()
        t_start = time.time()
        
        current_val = get_ratio_sq(pts)
        best_val = current_val
        best_pts = pts.copy()
        
        lr = 0.01
        
        it = 0
        while time.time() - t_start < time_budget:
            it += 1
            
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            val, grad = compute_grad_ratio(pts)
            
            if not np.all(np.isfinite(grad)):
                break
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-14:
                break
            grad_normalized = grad / grad_norm
            
            # Simple line search
            improved = False
            for trial in range(15):
                step_size = lr * (0.7 ** trial)
                if step_size < 1e-10:
                    break
                
                new_pts = pts + step_size * grad_normalized
                if not np.all(np.isfinite(new_pts)):
                    continue
                
                new_val = get_ratio_sq(new_pts)
                if new_val > val + 1e-13:
                    pts = new_pts
                    val = new_val
                    improved = True
                    if val > best_val:
                        best_val = val
                        best_pts = pts.copy()
                    break
            
            if not improved:
                lr *= 0.8
                if lr < 1e-10:
                    break
        
        return best_pts, best_val

    time_elapsed = time.time() - start_time
    if TIME_LIMIT - time_elapsed - SAFETY_MARGIN > 15:
        polish_pts, polish_ratio = gradient_polish(best_pts, 
                                                   time_budget=min(60.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN))
        if polish_ratio > best_ratio:
            best_pts = polish_pts
            best_ratio = polish_ratio

    # Ensure finite and correct shape
    final_pts = best_pts
    if final_pts is None or not np.all(np.isfinite(final_pts)) or final_pts.shape != (16, 2):
        final_pts = grid_init()

    return final_pts

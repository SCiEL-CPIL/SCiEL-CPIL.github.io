import numpy as np
import time
import itertools
from scipy.optimize import minimize
from scipy.spatial import Delaunay

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    N = 8
    n_points = N
    
    def diameter(pts):
        """Compute the diameter of the point set."""
        if len(pts) < 2:
            return 0.0
        d_mat = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
        return np.max(d_mat)
    
    def score(pts):
        """
        Compute the score: Delta / 8^8 where Delta is product of ordered pair distances
        after scaling the set to diameter 2.
        """
        if len(pts) < 2:
            return 0.0
        d_mat = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
        d_max = np.max(d_mat)
        if d_max < 1e-12:
            return 0.0
        
        # Scale to diameter 2
        pts_scaled = pts * (2.0 / d_max)
        
        # Compute distances in scaled set
        d_mat2 = np.sqrt(np.sum((pts_scaled[:, np.newaxis, :] - pts_scaled[np.newaxis, :, :])**2, axis=2))
        
        # Product of all ordered pair distances (i != j)
        # log(product) = sum_{i != j} log(d_ij)
        mask = ~np.eye(n_points, dtype=bool)
        d_safe = np.where(mask, np.maximum(d_mat2, 1e-15), 1.0)
        log_delta = np.sum(np.log(d_safe) * mask)
        delta = np.exp(log_delta)
        
        score = delta / (n_points ** n_points)
        return score
    
    def log_product_and_grad(pts):
        """
        Returns log product of all ordered pair distances and its gradient.
        For maximizing, we want to maximize sum_{i!=j} log|p_i - p_j|.
        """
        pts = np.array(pts)
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist = np.sqrt(dist_sq)
        
        mask = ~np.eye(n, dtype=bool)
        dist_safe = np.where(mask, np.maximum(dist, 1e-15), 1.0)
        
        log_prod = np.sum(np.log(dist_safe) * mask)
        
        # Gradient: d/dp_i sum_{j!=i} log|p_i - p_j| = sum_{j!=i} (p_i - p_j) / |p_i - p_j|^2
        # = sum_{j!=i} diff[i,j] / dist_sq[i,j]
        grad = np.zeros_like(pts)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d2 = dist_sq[i, j]
                if d2 < 1e-12:
                    continue
                grad[i] += diff[i, j] / d2
        
        return log_prod, grad
    
    def diameter_grad(pts):
        """
        Returns gradient of the diameter (max pairwise distance).
        For the pair (i,j) that achieves the max, grad_i = (p_i - p_j)/d, grad_j = (p_j - p_i)/d.
        """
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        
        # Find max pair (excluding diagonal)
        mask = ~np.eye(n, dtype=bool)
        dist_masked = np.where(mask, dist, -np.inf)
        idx = np.unravel_index(np.argmax(dist_masked), dist_masked.shape)
        i, j = idx
        
        grad = np.zeros_like(pts)
        d = dist[i, j]
        if d < 1e-12:
            return grad
        grad[i] = (pts[i] - pts[j]) / d
        grad[j] = (pts[j] - pts[i]) / d
        return grad
    
    def refine_local(pts0, max_iters=30):
        """
        Local refinement using projected gradient ascent on log product,
        subject to diameter <= 2 constraint.
        Uses a penalty method.
        """
        pts = np.array(pts0, dtype=float)
        n = len(pts)
        
        # Center and scale to diameter 2
        pts = pts - np.mean(pts, axis=0)
        d_max = diameter(pts)
        if d_max > 1e-12:
            pts = pts * (2.0 / d_max)
        
        best_pts = pts.copy()
        best_score = score(best_pts)
        
        step_size = 0.05
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            log_prod, grad = log_product_and_grad(pts)
            
            # Check diameter
            d_mat = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
            d_max = np.max(d_mat)
            
            # Penalty for diameter violation
            penalty = 0.0
            grad_pen = np.zeros_like(pts)
            if d_max > 2.0:
                # Find all pairs exceeding 2
                mask = ~np.eye(n, dtype=bool)
                violations = np.maximum(d_mat - 2.0, 0.0) * mask
                penalty = np.sum(violations)
                
                # Gradient of penalty
                for i in range(n):
                    for j in range(n):
                        if i < j and d_mat[i, j] > 2.0:
                            diff = pts[i] - pts[j]
                            dist = d_mat[i, j]
                            if dist > 1e-12:
                                term = diff / dist
                                grad_pen[i] += term
                                grad_pen[j] -= term
                
                penalty = 10.0 * penalty
                grad_pen = 100.0 * grad_pen
            
            total_grad = grad - grad_pen
            pts_new = pts + step_size * total_grad
            
            # Project to diameter <= 2
            d_mat_new = np.sqrt(np.sum((pts_new[:, np.newaxis, :] - pts_new[np.newaxis, :, :])**2, axis=2))
            d_max_new = np.max(d_mat_new)
            if d_max_new > 2.0:
                # Scale towards centroid
                centroid = np.mean(pts_new, axis=0)
                pts_new = centroid + (pts_new - centroid) * (2.0 / d_max_new)
            
            # Center
            pts_new = pts_new - np.mean(pts_new, axis=0)
            
            sc_new = score(pts_new)
            if sc_new > best_score:
                best_score = sc_new
                best_pts = pts_new.copy()
                step_size = min(step_size * 1.05, 0.2)
            else:
                step_size = max(step_size * 0.8, 0.005)
                if sc_new < best_score * 0.95:
                    pts = best_pts.copy()
                else:
                    pts = pts_new
            pts = pts_new
        
        return best_pts
    
    def solve_topology(pts0, max_iters=50):
        """
        Given an initial configuration, refine it using local optimization.
        Returns the best configuration found.
        """
        return refine_local(pts0, max_iters)
    
    # --- Generate diverse starting configurations ---
    
    starts = []
    
    # 1. Regular octagon
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    pts_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    starts.append(pts_oct)
    
    # 2. Concentric squares with various parameters
    for r_inner in np.linspace(0.1, 0.9, 10):
        for rot_inner in np.linspace(0, np.pi/4, 6):
            for rot_outer in np.linspace(0, np.pi/4, 6):
                a_inner = np.linspace(0, 2*np.pi, 4, endpoint=False) + rot_inner
                a_outer = np.linspace(0, 2*np.pi, 4, endpoint=False) + rot_outer
                p_inner = r_inner * np.column_stack([np.cos(a_inner), np.sin(a_inner)])
                p_outer = np.column_stack([np.cos(a_outer), np.sin(a_outer)])
                starts.append(np.vstack([p_inner, p_outer]))
    
    # 3. Two squares with different rotations and scales
    for scale in np.linspace(0.3, 0.9, 5):
        for rot in np.linspace(0, np.pi/4, 5):
            a1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
            a2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + rot
            p1 = scale * np.column_stack([np.cos(a1), np.sin(a1)])
            p2 = np.column_stack([np.cos(a2), np.sin(a2)])
            starts.append(np.vstack([p1, p2]))
    
    # 4. Perturbed octagons
    np.random.seed(42)
    for _ in range(30):
        radii = 1.0 + np.random.normal(0, 0.3, 8)
        radii = np.abs(radii) + 0.1
        angles_p = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.random.normal(0, 0.15, 8)
        pts = np.column_stack([radii * np.cos(angles_p), radii * np.sin(angles_p)])
        starts.append(pts)
    
    # 5. Random points in unit disk
    for _ in range(30):
        r = np.sqrt(np.random.rand(8))
        theta = np.random.rand(8) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        starts.append(pts)
    
    # 6. "Butterfly" configurations - two triangles sharing an edge or vertex
    # Triangle 1 at origin, Triangle 2 at some offset
    for offset_r in np.linspace(0.5, 1.5, 5):
        for offset_theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
            # Triangle 1
            a1 = np.linspace(0, 2*np.pi, 3, endpoint=False)
            p1 = np.column_stack([np.cos(a1), np.sin(a1)])
            # Triangle 2 offset
            a2 = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
            p2 = offset_r * np.column_stack([np.cos(a2) + np.cos(offset_theta), np.sin(a2) + np.sin(offset_theta)])
            # 5th point at center or elsewhere
            p5 = np.array([[0, 0], [offset_r*np.cos(offset_theta), offset_r*np.sin(offset_theta)]])
            # Actually we need 8 points, so add more
            # Let's just use 8 random-ish points
            pts = np.vstack([p1, p2, p5])
            if len(pts) < 8:
                # Pad with random points
                extra = np.random.randn(8 - len(pts), 2) * 0.5
                pts = np.vstack([pts, extra])
            starts.append(pts[:8])
    
    # 7. Random configurations with some structure
    for _ in range(20):
        # Mix of structured and random
        pts = np.random.randn(8, 2)
        pts = pts / np.max(np.sqrt(np.sum(pts**2, axis=1)))
        starts.append(pts)
    
    # --- Evaluate and refine each start ---
    
    best_score_global = 0.0
    best_pts_global = None
    
    for i, s in enumerate(starts):
        if time.time() > deadline:
            break
        
        # Quick score check
        sc_init = score(s)
        
        # Refine
        refined_pts = solve_topology(s, max_iters=25)
        sc_ref = score(refined_pts)
        
        if sc_ref > best_score_global:
            best_score_global = sc_ref
            best_pts_global = refined_pts.copy()
            
            # Try additional refinement from the best found
            if time.time() < deadline - 15:
                for extra_iter in range(5):
                    if time.time() > deadline - 5:
                        break
                    refined_pts2 = solve_topology(best_pts_global, max_iters=15)
                    sc_ref2 = score(refined_pts2)
                    if sc_ref2 > best_score_global:
                        best_score_global = sc_ref2
                        best_pts_global = refined_pts2.copy()
                    else:
                        break
    
    # Fallback
    if best_pts_global is None:
        best_pts_global = pts_oct
    
    # Final scaling to diameter 2 and clipping
    d_max_final = diameter(best_pts_global)
    if d_max_final > 1e-12:
        best_pts_global = best_pts_global * (2.0 / d_max_final)
    
    best_pts_global = np.clip(best_pts_global, -1e6, 1e6)
    
    return best_pts_global.tolist()

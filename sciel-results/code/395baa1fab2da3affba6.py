import numpy as np
import time
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Step 1: Generate hexagonal lattice points
    def hex_lattice_init(n):
        best_points = None
        best_min_dist = 0
        for trial in range(30):
            spacing = 2.0 / (np.sqrt(n) + 1.0) * (1.0 - trial * 0.015)
            if spacing <= 0:
                continue
            k = int(1.0 / spacing) + 2
            points = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    r = np.hypot(x, y)
                    if r <= 1.0 - spacing / 2.0:
                        points.append([x, y])
            if len(points) < n:
                continue
            points = np.array(points)
            dists = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists)[:n]
            pts_sel = points[idx]
            # Compute min pair distance for this selection
            tree = KDTree(pts_sel)
            pairs = tree.query_pairs(spacing * 1.5)
            if len(pairs) > 0:
                pair_dists = [np.linalg.norm(pts_sel[a] - pts_sel[b]) for a, b in pairs]
                min_pair = min(pair_dists)
            else:
                min_pair = spacing
            min_boundary = 1.0 - np.max(dists[idx])
            min_r = min(min_pair / 2.0, min_boundary)
            if best_points is None or min_r > best_min_dist:
                best_min_dist = min_r
                best_points = pts_sel.copy()
        return best_points
    
    pts = hex_lattice_init(N)
    
    # Step 2: Compute initial radius
    def compute_radius(pts):
        min_pair = np.min(pdist(pts)) / 2.0
        min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
        return min(min_pair, min_boundary)
    
    best_radius = compute_radius(pts)
    best_pts = pts.copy()
    
    # Step 3: Fast iterative optimization using KDTree for neighbor queries
    # Use a more efficient approach: only compute forces from nearby points
    
    # Precompute some parameters
    current_spacing = best_radius * 2.0  # approximate spacing
    
    # Optimization loop with KDTree-based neighbor search
    for it in range(20000):
        if time.time() > deadline:
            break
        
        # Build KDTree for neighbor queries
        tree = KDTree(pts)
        
        # For each point, find neighbors within a cutoff distance
        cutoff = max(current_spacing * 3.0, 0.1)
        
        # Compute repulsion forces from neighbors only
        forces = np.zeros_like(pts)
        for i in range(N):
            # Query neighbors within cutoff
            neighbors = tree.query_ball_point(pts[i], cutoff)
            for j in neighbors:
                if j == i:
                    continue
                diff = pts[i] - pts[j]
                d = np.linalg.norm(diff)
                if d < 1e-10:
                    continue
                # Coulomb-like repulsion: force ~ 1/d^2, direction = diff/d
                force_mag = 1.0 / (d * d + 1e-10)
                forces[i] += diff / d * force_mag
        
        # Boundary force: push inward if too close to edge
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        boundary_margin = 1.0 - r_pts
        boundary_force = np.zeros_like(pts)
        for i in range(N):
            if r_pts[i] > 1e-10:
                excess = max(0.0, 0.05 - boundary_margin[i])
                if excess > 0:
                    boundary_force[i] = -pts[i] / r_pts[i] * excess * excess * 50.0
        
        total_force = forces + boundary_force
        
        # Adaptive step size
        base_step = 0.002 * (1.0 + 0.5 * np.exp(-it / 500.0))
        force_mag = np.linalg.norm(total_force, axis=1)
        max_step = 0.02
        scale = np.minimum(base_step, max_step / (force_mag + 1e-10))
        pts_new = pts + total_force * scale[:, np.newaxis]
        
        # Keep points inside the disk
        r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
        mask = r_new > 0.999
        if np.any(mask):
            pts_new[mask] = pts_new[mask] / r_new[mask, np.newaxis] * 0.999
        
        # Check if this improves the radius
        r_new_val = compute_radius(pts_new)
        if r_new_val > best_radius:
            best_radius = r_new_val
            best_pts = pts_new.copy()
            pts = pts_new.copy()
            current_spacing = best_radius * 2.0
        else:
            # Still accept for exploration but with smaller effective step
            pts = pts_new.copy()
    
    # Final polish: use soft-min surrogate to refine
    # Use log-sum-exp based soft-min to get smooth gradients
    def soft_min_radius(pts, temperature=0.001):
        """Compute a smooth approximation of the minimum radius using log-sum-exp."""
        # Pairwise distances
        dists = pdist(pts)
        # Boundary distances
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        boundary_dists = 1.0 - r_pts
        
        # Combine all constraints
        all_dists = np.concatenate([dists, boundary_dists])
        
        # Soft-min: -log(sum(exp(-d/t))) * t
        # To avoid overflow, use the log-sum-exp trick
        min_val = np.min(all_dists)
        if min_val < 0:
            min_val = 0
        shifted = all_dists - min_val
        log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
        soft_min = min_val - temperature * log_sum
        
        return soft_min / 2.0
    
    # Polish with gradient ascent on soft-min radius
    if time.time() < deadline - 10:
        # Use numerical gradients for the soft-min objective
        current = best_pts.copy()
        current_radius = soft_min_radius(current)
        
        for polish_it in range(500):
            if time.time() > deadline - 5:
                break
            
            # Compute gradient numerically
            eps = 1e-6
            grad = np.zeros_like(current)
            for i in range(N):
                for d in range(2):
                    old_val = current[i, d]
                    current[i, d] = old_val + eps
                    f_plus = soft_min_radius(current)
                    current[i, d] = old_val - eps
                    f_minus = soft_min_radius(current)
                    current[i, d] = old_val
                    grad[i, d] = (f_plus - f_minus) / (2 * eps)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad /= grad_norm
            
            # Take a step in the gradient direction
            step_size = 0.005 * (1.0 + 0.5 * np.exp(-polish_it / 100.0))
            current += grad * step_size
            
            # Keep inside disk
            r_cur = np.hypot(current[:, 0], current[:, 1])
            mask = r_cur > 0.999
            if np.any(mask):
                current[mask] = current[mask] / r_cur[mask, np.newaxis] * 0.999
            
            # Check improvement
            new_soft_radius = soft_min_radius(current)
            if new_soft_radius > current_radius:
                current_radius = new_soft_radius
        
        # Final check: compute actual radius and update best if improved
        final_radius = compute_radius(current)
        if final_radius > best_radius:
            best_radius = final_radius
            best_pts = current.copy()
    
    # Final: return best
    return best_pts

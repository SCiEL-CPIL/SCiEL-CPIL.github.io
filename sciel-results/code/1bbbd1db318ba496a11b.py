"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def compute_radius(pts):
        """Compute the maximum common radius for given centers."""
        n = len(pts)
        if n == 0:
            return 0.0
        norms = np.linalg.norm(pts, axis=1)
        boundary_r = 1.0 - np.max(norms)
        if boundary_r <= 0:
            return 0.0
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        return min(boundary_r, min_pair_dist / 2.0)

    def hex_initial(spacing, n_points):
        """Generate hexagonal lattice points and select n_points closest to center."""
        sqrt3 = np.sqrt(3.0)
        k = int(1.0 / spacing) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * sqrt3 / 2.0
                if np.hypot(x, y) <= 1.0:
                    points.append((x, y))
        if len(points) < n_points:
            return None
        pts = np.array(points)
        norms = np.linalg.norm(pts, axis=1)
        idx = np.argsort(norms)[:n_points]
        return pts[idx]

    def gradient_ascent(pts, time_limit):
        """Smooth gradient ascent on the packing radius using soft-min surrogate."""
        n = len(pts)
        if n == 0:
            return pts
        
        best_pts = pts.copy()
        best_r = compute_radius(best_pts)
        
        # Soft-min parameter for smooth approximation
        beta = 50.0  # Higher = sharper min
        step_size = 0.005
        patience = 0
        max_patience = 300
        
        for iteration in range(20000):
            if time.time() > time_limit:
                break
            
            # Compute current radius and forces
            norms = np.linalg.norm(pts, axis=1)
            boundary_r = 1.0 - np.max(norms)
            
            # Pairwise distances
            dists = cdist(pts, pts)
            np.fill_diagonal(dists, np.inf)
            
            # Soft-min of pairwise distances
            # Using log-sum-exp trick for numerical stability
            min_d = np.min(dists)
            # Soft min: -1/beta * log(sum(exp(-beta * d)))
            # But we want gradient of min(d_ij/2, 1-|c_i|)
            
            # Gradient approach: 
            # r = min(min_ij d_ij/2, 1 - max_i |c_i|)
            # Use smooth approximation: soft_min of (d_ij/2 for all i<j) and (1-|c_i| for all i)
            
            # Collect all constraint values
            # For efficiency, only consider nearby pairs
            threshold = 2.0 * max(best_r, 0.01) + 0.1
            close_mask = dists < threshold
            np.fill_diagonal(close_mask, False)
            
            # Compute gradient of soft-min
            # For pairwise: contribution from pair (i,j) is proportional to exp(-beta * d_ij/2)
            # For boundary: contribution from point i is proportional to exp(-beta * (1-|c_i|))
            
            forces = np.zeros_like(pts)
            
            # Pairwise repulsion (soft)
            for i in range(n):
                for j in range(i + 1, n):
                    if close_mask[i, j]:
                        d = dists[i, j]
                        if d < 1e-10:
                            forces[i] += np.random.randn(2) * 0.01
                            forces[j] -= np.random.randn(2) * 0.01
                        else:
                            # Soft repulsion: stronger when closer
                            weight = np.exp(-beta * d / 2.0)
                            direction = (pts[i] - pts[j]) / d
                            forces[i] += direction * weight
                            forces[j] -= direction * weight
            
            # Boundary attraction (soft)
            for i in range(n):
                norm_i = norms[i]
                if norm_i > 1e-10:
                    boundary_val = 1.0 - norm_i
                    weight = np.exp(-beta * boundary_val)
                    direction = -pts[i] / norm_i
                    forces[i] += direction * weight
            
            # Normalize forces
            force_norm = np.linalg.norm(forces)
            if force_norm > 1e-10:
                forces = forces / force_norm
            
            # Update
            pts_new = pts + step_size * forces
            
            # Project to unit disk
            norms_new = np.linalg.norm(pts_new, axis=1)
            max_allowed = 1.0 - 1e-10
            for i in range(n):
                if norms_new[i] > max_allowed:
                    pts_new[i] = pts_new[i] / norms_new[i] * max_allowed
            
            # Evaluate
            new_r = compute_radius(pts_new)
            
            if new_r > best_r:
                best_r = new_r
                best_pts = pts_new.copy()
                pts = pts_new
                patience = 0
                step_size = min(step_size * 1.02, 0.02)
            else:
                patience += 1
                step_size *= 0.98
                if patience > max_patience:
                    break
        
        return best_pts

    def local_search(pts, time_limit):
        """Local search with random perturbations."""
        n = len(pts)
        best_pts = pts.copy()
        best_r = compute_radius(best_pts)
        
        step_size = 0.01
        patience = 0
        max_patience = 100
        
        for iteration in range(10000):
            if time.time() > time_limit:
                break
            
            # Random perturbation
            perturbation = np.random.randn(n, 2) * step_size
            pts_new = pts + perturbation
            
            # Project to unit disk
            norms_new = np.linalg.norm(pts_new, axis=1)
            max_allowed = 1.0 - 1e-10
            for i in range(n):
                if norms_new[i] > max_allowed:
                    pts_new[i] = pts_new[i] / norms_new[i] * max_allowed
            
            new_r = compute_radius(pts_new)
            
            if new_r > best_r:
                best_r = new_r
                best_pts = pts_new.copy()
                pts = pts_new
                patience = 0
                step_size = min(step_size * 1.05, 0.05)
            else:
                patience += 1
                step_size *= 0.95
                if patience > max_patience:
                    break
        
        return best_pts

    # Initial estimate of spacing
    s = np.sqrt(2.0 * np.pi / (np.sqrt(3.0) * N))
    
    best_points = None
    best_radius = 0.0
    
    # Try multiple initial configurations
    for spacing_factor in [1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15]:
        if time.time() > deadline - 20.0:
            break
            
        spacing = s * spacing_factor
        pts = hex_initial(spacing, N)
        if pts is None:
            continue
        
        # Gradient ascent
        pts_ga = gradient_ascent(pts, deadline - 15.0)
        r_ga = compute_radius(pts_ga)
        if r_ga > best_radius:
            best_radius = r_ga
            best_points = pts_ga.copy()
        
        # Local search from best
        if time.time() < deadline - 10.0:
            pts_ls = local_search(pts_ga, deadline - 5.0)
            r_ls = compute_radius(pts_ls)
            if r_ls > best_radius:
                best_radius = r_ls
                best_points = pts_ls.copy()
    
    # Final polish with gradient ascent if time permits
    if time.time() < deadline - 5.0 and best_points is not None:
        pts_final = gradient_ascent(best_points, deadline - 1.0)
        r_final = compute_radius(pts_final)
        if r_final > best_radius:
            best_points = pts_final
    
    if best_points is None or len(best_points) != N:
        # Fallback
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        pts = hex_initial(spacing, N)
        if pts is not None:
            return pts
        return np.zeros((N, 2))
    
    return best_points
# EVOLVE-BLOCK-END

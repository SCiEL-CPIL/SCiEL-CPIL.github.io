"""Improved circle packing for N=300 in unit circle using multi-restart gradient optimization."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s limit
    
    def compute_score(pts):
        """Compute the packing score r for given points."""
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        # Check all points are inside unit circle
        radii = np.hypot(pts[:, 0], pts[:, 1])
        if np.any(radii >= 1.0):
            return 0.0
        min_boundary = np.min(1.0 - radii)
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        return min(min_pair, min_boundary)

    def generate_hex_init(spacing_scale=1.0):
        """Generate a hexagonal lattice configuration scaled to fit."""
        # For N=300 circles, we need roughly sqrt(300) ~ 17-18 circles across
        # Hexagonal packing: each row has alternating offsets
        # Let's generate a hexagonal grid and select N points
        
        # Estimate spacing: area per circle ~ pi/(N * density), density ~ 0.9069 for hex
        # r ~ 0.053, so d = 2r ~ 0.106
        # Let's try different spacings
        
        best_pts = None
        best_score = 0.0
        
        for scale in [0.9, 0.95, 1.0, 1.05, 1.1]:
            d = 0.106 * scale  # base spacing
            # Generate hexagonal grid
            points = []
            # Range: need to cover roughly -1 to 1 in both x and y
            # Number of rows: ~ 2/d * sqrt(3)/2 * 2 ~ 20-25
            n_rows = int(2.0 / (d * np.sqrt(3) / 2)) + 5
            n_cols = int(2.0 / d) + 5
            
            for row in range(-n_rows//2, n_rows//2 + 1):
                for col in range(-n_cols//2, n_cols//2 + 1):
                    x = col * d
                    y = row * d * np.sqrt(3) / 2
                    if row % 2 != 0:
                        x += d / 2
                    points.append([x, y])
            
            points = np.array(points)
            
            # Filter to unit circle
            radii = np.hypot(points[:, 0], points[:, 1])
            mask = radii < 0.98
            candidates = points[mask]
            
            if len(candidates) < N:
                continue
            
            # Greedy selection: pick points that maximize minimum distance
            # Start with center
            selected = []
            remaining = candidates.copy()
            
            # Find point closest to origin
            dists_origin = np.hypot(remaining[:, 0], remaining[:, 1])
            idx = np.argmin(dists_origin)
            selected.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)
            
            while len(selected) < N and len(remaining) > 0:
                selected_arr = np.array(selected)
                diff = remaining[:, np.newaxis, :] - selected_arr[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                min_dists = np.min(dists, axis=1)
                idx = np.argmax(min_dists)
                selected.append(remaining[idx])
                remaining = np.delete(remaining, idx, axis=0)
                
                if time.time() > deadline - 5:
                    break
            
            if len(selected) < N:
                continue
            
            pts = np.array(selected[:N])
            score = compute_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts
        
        if best_pts is None:
            # Fallback: simple grid
            pts = np.random.uniform(-0.95, 0.95, (N, 2))
            radii = np.hypot(pts[:, 0], pts[:, 1])
            outside = radii > 0.95
            pts[outside] = pts[outside] * (0.95 / radii[outside, np.newaxis])
            best_pts = pts
            best_score = compute_score(pts)
        
        return best_pts, best_score

    def gradient_step(pts, r_target, step_size, n_steps=1):
        """
        Perform gradient ascent on the minimum distance metric.
        r_target is the target radius we're trying to achieve.
        """
        pts = pts.copy()
        
        for _ in range(n_steps):
            # Compute pairwise distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Current minimum distance
            min_dists = np.min(dists)
            current_r = min_dists / 2.0
            
            # Compute forces
            forces = np.zeros_like(pts)
            
            # Pairwise repulsion: push apart if d < 2*r_target
            overlap = 2.0 * r_target - dists
            overlap[overlap < 0] = 0.0
            
            dists_safe = np.where(dists > 1e-10, dists, 1.0)
            dir_vecs = diff / dists_safe[:, :, np.newaxis]
            
            weighted_dirs = dir_vecs * overlap[:, :, np.newaxis]
            forces += np.sum(weighted_dirs, axis=1)
            
            # Boundary constraint: push in if |p| > 1 - r_target
            radii = np.hypot(pts[:, 0], pts[:, 1])
            boundary_violation = radii - (1.0 - r_target)
            boundary_violation[boundary_violation < 0] = 0.0
            
            radii_safe = np.where(radii > 1e-10, radii, 1.0)
            boundary_dirs = -pts / radii_safe[:, np.newaxis]
            forces += boundary_dirs * boundary_violation[:, np.newaxis]
            
            # Update
            pts += forces * step_size
            
            # Project inside unit circle
            new_radii = np.hypot(pts[:, 0], pts[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                pts[outside] = pts[outside] / new_radii[outside, np.newaxis]
        
        return pts

    def optimize_single(pts, max_iters=5000, deadline_check=None):
        """
        Optimize a single configuration using adaptive gradient ascent.
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        step_size = 0.005
        r_target = best_score * 1.01 + 0.001
        
        for it in range(max_iters):
            if deadline_check is not None and time.time() > deadline_check:
                break
            
            # Compute current score
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
            radii = np.hypot(pts[:, 0], pts[:, 1])
            min_boundary = np.min(1.0 - radii)
            current_score = min(min_pair, min_boundary)
            
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
                # Increase target
                r_target = best_score * 1.01 + 0.001
            else:
                # No improvement, decrease step size
                step_size *= 0.995
            
            # Gradient step
            pts = gradient_step(pts, r_target, step_size, n_steps=1)
            
            # Check if step size is too small
            if step_size < 1e-7:
                break
            
            # Periodically recompute and adjust
            if it % 100 == 0:
                # Slight random perturbation to escape local minima
                if np.random.random() < 0.1:
                    pts += np.random.randn(N, 2) * 0.001
                    new_radii = np.hypot(pts[:, 0], pts[:, 1])
                    outside = new_radii > 1.0
                    if np.any(outside):
                        pts[outside] = pts[outside] / new_radii[outside, np.newaxis]
        
        return best_pts, best_score

    def local_search(pts, max_iters=3000, deadline_check=None):
        """
        Local search: perturb one point at a time, keep if better.
        Focus on points that are close to others or close to boundary.
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        temp = 0.008
        
        for it in range(max_iters):
            if deadline_check is not None and time.time() > deadline_check:
                break
            
            # Find the most constrained point (closest to another or boundary)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair_dists = np.min(dists, axis=1)
            
            radii = np.hypot(pts[:, 0], pts[:, 1])
            boundary_dists = 1.0 - radii
            
            # Combined constraint score
            constraint_score = np.minimum(min_pair_dists / 2.0, boundary_dists)
            
            # Bias toward more constrained points
            weights = 1.0 / (constraint_score + 0.01)
            weights /= np.sum(weights)
            
            idx = np.random.choice(N, p=weights)
            orig_pt = pts[idx].copy()
            
            # Perturb
            pts[idx] += np.random.randn(2) * temp
            
            # Project inside
            r = np.hypot(pts[idx, 0], pts[idx, 1])
            if r > 1.0:
                pts[idx] = pts[idx] / r
            
            new_score = compute_score(pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
                temp *= 0.99
            else:
                pts[idx] = orig_pt
                temp *= 0.9999
            
            # Reset temperature if too small
            if temp < 0.001:
                temp = 0.005
        
        return best_pts, best_score

    # Main execution
    # 1. Initialize with hexagonal lattice
    pts, init_score = generate_hex_init()
    
    best_pts = pts.copy()
    best_score = init_score
    
    # 2. Gradient optimization
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        pts, score = optimize_single(pts, max_iters=3000, deadline_check=deadline - 30)
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # 3. Local search for final refinement
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        pts, score = local_search(best_pts, max_iters=2000, deadline_check=deadline - 2)
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # 4. Second round of gradient if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        pts, score = optimize_single(best_pts, max_iters=2000, deadline_check=deadline - 5)
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # Final validation
    final_score = compute_score(best_pts)
    if final_score > best_score:
        best_score = final_score
    
    # Ensure validity
    radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
    if np.any(radii >= 1.0) or not np.all(np.isfinite(best_pts)):
        scale = 0.99 / np.max(radii) if np.max(radii) > 0 else 1.0
        best_pts = best_pts * scale
    
    return best_pts
# EVOLVE-BLOCK-END

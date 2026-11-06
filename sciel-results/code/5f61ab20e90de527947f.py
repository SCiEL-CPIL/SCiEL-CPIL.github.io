import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def compute_score(pts):
        """Compute the packing score r for given points."""
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        radii = np.hypot(pts[:, 0], pts[:, 1])
        if np.any(radii >= 1.0):
            return 0.0
        min_boundary = np.min(1.0 - radii)
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair = np.sqrt(np.min(dists_sq)) / 2.0
        return min(min_pair, min_boundary)

    def generate_spiral_init():
        """Generate a golden-angle spiral configuration."""
        ga = np.pi * (3.0 - np.sqrt(5.0))
        indices = np.arange(N)
        R_max = 0.95
        radii = np.sqrt(indices / N) * R_max
        angles = indices * ga
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack((x, y))

    def generate_hex_init():
        """Generate a hexagonal lattice configuration scaled to fit."""
        spacing = np.sqrt(4.0 * np.pi / (N * np.sqrt(3.0)))
        dx = spacing
        dy = spacing * np.sqrt(3.0) / 2.0
        range_x = int(np.ceil(1.0 / dx)) + 1
        range_y = int(np.ceil(1.0 / dy)) + 1
        pts = []
        for j in range(-range_y, range_y + 1):
            for i in range(-range_x, range_x + 1):
                x = i * dx + (j % 2) * dx / 2.0
                y = j * dy
                if x*x + y*y <= 1.0:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) > N:
            idx = np.random.choice(len(pts), N, replace=False)
            pts = pts[idx]
        elif len(pts) < N:
            while len(pts) < N:
                p = np.random.randn(2)
                p /= np.linalg.norm(p) * 0.9
                pts = np.vstack([pts, p])
        return pts[:N]

    def optimize_log_barrier(pts, deadline_check=None):
        """
        Optimize using a log-barrier objective:
        Maximize sum_i [sum_{j!=i} log(d_ij) + log(1 - |c_i|)]
        subject to d_ij > 0 and |c_i| < 1
        
        This is penalty-free and maintains feasibility via the log barrier.
        Uses adaptive step-size with annealing.
        """
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        # Initial parameters
        lr = 0.001
        lr_min = 1e-6
        lr_decay = 0.999
        n_iter = 0
        max_iters = 5000
        
        # Precompute initial distances for efficiency
        while n_iter < max_iters:
            if deadline_check is not None and time.time() > deadline_check:
                break
            
            # Compute pairwise distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dists_sq, 1e-10)
            dists = np.sqrt(dists_sq)
            
            # Log-barrier gradient:
            # For pair (i,j): d/dc_i [log(d_ij)] = (c_i - c_j) / d_ij^2
            # For boundary: d/dc_i [log(1 - |c_i|)] = -c_i / (|c_i| * (1 - |c_i|))
            
            # Pair gradient: sum_{j!=i} (c_i - c_j) / d_ij^2
            # = sum_{j!=i} diff[i,j] / dists_sq[i,j]
            weights = 1.0 / dists_sq
            np.fill_diagonal(weights, 0.0)
            grad_pair = np.sum(diff * weights[:, :, np.newaxis], axis=1)
            
            # Boundary gradient
            radii = np.hypot(pts[:, 0], pts[:, 1])
            # Avoid division by zero and log(0)
            radii_safe = np.maximum(radii, 1e-10)
            boundary_term = 1.0 / (radii_safe * (1.0 - radii_safe))
            grad_bound = -pts * boundary_term[:, np.newaxis]
            
            grad = grad_pair + grad_bound
            
            # Adaptive step-size: reduce if score doesn't improve
            pts_new = pts + grad * lr
            
            # Project inside unit circle (hard constraint)
            new_radii = np.hypot(pts_new[:, 0], pts_new[:, 1])
            outside = new_radii >= 1.0
            if np.any(outside):
                pts_new[outside] = pts_new[outside] * (1.0 - 1e-10) / new_radii[outside, np.newaxis]
            
            # Check validity
            if not np.all(np.isfinite(pts_new)):
                lr *= 0.5
                n_iter += 1
                continue
            
            score_new = compute_score(pts_new)
            
            if score_new > best_score:
                best_pts = pts_new.copy()
                best_score = score_new
                pts = pts_new
                lr = min(lr * 1.05, 0.01)  # Increase step if improving
            else:
                pts = pts_new  # Still move, but reduce step
                lr *= 0.95
            
            lr = max(lr, lr_min)
            lr *= lr_decay
            n_iter += 1
            
            # Early stopping if stuck
            if n_iter % 500 == 0 and lr < 1e-5:
                break
        
        return best_pts, best_score

    def optimize_multi_stage(pts, deadline_check=None):
        """
        Multi-stage optimization:
        1. Coarse optimization with large steps
        2. Fine optimization with small steps
        3. Final polish
        """
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        # Stage 1: Coarse
        remaining = deadline_check - time.time() if deadline_check else 100.0
        stage1_time = remaining * 0.4
        pts1, score1 = optimize_log_barrier(best_pts, deadline_check=time.time() + stage1_time)
        if score1 > best_score:
            best_pts = pts1
            best_score = score1
        
        # Stage 2: Fine
        remaining = deadline_check - time.time() if deadline_check else 100.0
        stage2_time = remaining * 0.4
        pts2, score2 = optimize_log_barrier(best_pts, deadline_check=time.time() + stage2_time)
        if score2 > best_score:
            best_pts = pts2
            best_score = score2
        
        # Stage 3: Polish
        remaining = deadline_check - time.time() if deadline_check else 100.0
        stage3_time = remaining * 0.3
        pts3, score3 = optimize_log_barrier(best_pts, deadline_check=time.time() + stage3_time)
        if score3 > best_score:
            best_pts = pts3
            best_score = score3
        
        return best_pts, best_score

    # Main execution
    # Try multiple initializations
    inits = [
        generate_spiral_init(),
        generate_hex_init(),
        np.random.randn(N, 2) * 0.5
    ]
    
    best_pts_overall = None
    best_score_overall = 0.0
    
    for init in inits:
        if time.time() > deadline - 15:
            break
        
        # Optimize
        pts, score = optimize_multi_stage(init, deadline_check=deadline - 5)
        
        if score > best_score_overall:
            best_score_overall = score
            best_pts_overall = pts
    
    # Final validation
    if best_pts_overall is None:
        best_pts_overall = generate_spiral_init()
    
    # Ensure validity
    radii = np.hypot(best_pts_overall[:, 0], best_pts_overall[:, 1])
    if np.any(radii >= 1.0) or not np.all(np.isfinite(best_pts_overall)):
        scale = 0.99 / np.max(radii) if np.max(radii) > 0 else 1.0
        best_pts_overall = best_pts_overall * scale
    
    return best_pts_overall

"""Improved packing of 300 equal circles in a unit circle using force-directed optimization."""
import numpy as np
import time
import math

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    def compute_score(points):
        """Compute the minimum radius for the entire configuration (O(N^2))."""
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        radii = np.linalg.norm(points, axis=1)
        min_boundary = np.min(1.0 - radii)
        
        return min(min_dist / 2.0, min_boundary)

    def initial_hex(n):
        """Generate initial hexagonal lattice points."""
        s_est = math.sqrt(2 * math.pi * 0.95**2 / (n * math.sqrt(3)))
        best_points = None
        best_score = -1
        
        for trial in range(10):
            spacing = s_est * (1.0 + trial * 0.01)
            points = []
            r_max = int(1.0 / (spacing * math.sqrt(3) / 2)) + 2
            for j in range(-r_max, r_max + 1):
                for i in range(-r_max, r_max + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * math.sqrt(3) / 2
                    if x * x + y * y <= (1.0 - spacing / 2) ** 2:
                        points.append((x, y))
            
            if len(points) >= n:
                points_arr = np.array(points)
                dists = np.hypot(points_arr[:, 0], points_arr[:, 1])
                idx = np.argsort(dists)[:n]
                candidate = points_arr[idx]
                score = compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate
            
            if time.time() > deadline - 10:
                break
        
        if best_points is None:
            best_points = np.random.randn(n, 2) * 0.8
            max_r = np.max(np.hypot(best_points[:, 0], best_points[:, 1]))
            if max_r > 0.9:
                best_points *= 0.9 / max_r
        return best_points

    def initial_rings(n):
        """Generate initial concentric ring points."""
        best_points = None
        best_score = -1
        rng = np.random.default_rng(42)
        
        for num_rings in range(10, 25):
            points = []
            # Center
            points.append((0.0, 0.0))
            
            # Determine points per ring
            # n_k proportional to k
            C = 2 * n / (num_rings * (num_rings + 1))
            max_r = 0.95
            
            for k in range(1, num_rings + 1):
                n_k = int(round(C * k))
                if k == num_rings:
                    # Adjust last ring to match total N
                    n_k = n - len(points)
                    if n_k < 1:
                        continue
                
                r_ring = max_r * math.sqrt(k / num_rings)
                offset = 0 if k % 2 == 0 else np.pi / n_k
                
                for m in range(n_k):
                    angle = 2 * np.pi * m / n_k
                    x = r_ring * math.cos(angle + offset)
                    y = r_ring * math.sin(angle + offset)
                    points.append((x, y))
            
            if len(points) != n:
                # Fallback if calculation was off
                points_arr = np.array(points)
                if len(points_arr) > n:
                    idx = rng.permutation(len(points_arr))[:n]
                    points_arr = points_arr[idx]
                elif len(points_arr) < n:
                    # Pad with random points near boundary
                    extra = n - len(points_arr)
                    new_pts = np.random.randn(extra, 2) * 0.9
                    points_arr = np.vstack([points_arr, new_pts])
            else:
                points_arr = np.array(points)
            
            score = compute_score(points_arr)
            if score > best_score:
                best_score = score
                best_points = points_arr
            
            if time.time() > deadline - 10:
                break
        
        if best_points is None:
            best_points = np.random.randn(n, 2) * 0.8
            max_r = np.max(np.hypot(best_points[:, 0], best_points[:, 1]))
            if max_r > 0.9:
                best_points *= 0.9 / max_r
        return best_points

    def force_optimize(points, max_time=60.0):
        """
        Force-directed optimization using active constraints.
        """
        start = time.time()
        pts = points.astype(np.float64).copy()
        best_score = compute_score(pts)
        best_pts = pts.copy()
        
        step_size = 0.01
        decay = 0.995
        eta = 0.01
        
        n = len(pts)
        
        while time.time() - start < max_time:
            current_score = compute_score(pts)
            
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
                
            # Compute forces
            forces = np.zeros_like(pts)
            
            # Pairwise forces
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            active_mask = dists < (2.0 * current_score + eta)
            np.fill_diagonal(active_mask, False)
            
            safe_dists = np.where(active_mask, dists, 1.0)
            force_vecs = np.where(active_mask[:, :, np.newaxis], diff / safe_dists[:, :, np.newaxis], 0.0)
            pair_forces = np.sum(force_vecs, axis=1)
            forces += pair_forces
            
            # Boundary forces
            norms = np.linalg.norm(pts, axis=1)
            active_bound = (1.0 - norms) < (current_score + eta)
            
            safe_norms = np.where(active_bound, norms, 1.0)
            bound_forces = np.where(active_bound[:, np.newaxis], -pts / safe_norms[:, np.newaxis], 0.0)
            forces += bound_forces
            
            # Update
            force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
            update = step_size * forces / np.where(force_norms > 1e-10, force_norms, 1.0)
            
            new_pts = pts + update
            
            # Project
            new_norms = np.linalg.norm(new_pts, axis=1)
            mask = new_norms > 0.999
            if np.any(mask):
                new_pts[mask] *= 0.999 / new_norms[mask]
            
            new_score = compute_score(new_pts)
            
            if new_score >= current_score - 1e-6:
                pts = new_pts
            else:
                step_size *= 0.9
                if step_size < 0.001:
                    idx = np.random.randint(n)
                    pts[idx] += np.random.randn(2) * 0.01
                    r = np.linalg.norm(pts[idx])
                    if r > 0.999:
                        pts[idx] *= 0.999 / r
                    step_size = 0.005
                    eta = 0.02
                    
            step_size *= decay
            
            if step_size < 1e-6:
                break
                
        return best_pts, best_score

    # Generate initial configurations
    pts_hex = initial_hex(N)
    score_hex = compute_score(pts_hex)
    
    pts_rings = initial_rings(N)
    score_rings = compute_score(pts_rings)
    
    if score_rings > score_hex:
        points = pts_rings
        score = score_rings
    else:
        points = pts_hex
        score = score_hex
        
    # Optimize
    if time.time() < deadline - 20:
        points, score = force_optimize(points, max_time=100.0)
        
    return points
# EVOLVE-BLOCK-END

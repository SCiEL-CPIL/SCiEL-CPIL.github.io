"""Improved packing of 300 equal circles in a unit circle using ring-based optimization."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def compute_score(points):
        """Compute the minimum radius for the entire configuration (O(N^2))."""
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        radii = np.linalg.norm(points, axis=1)
        min_boundary = np.min(1.0 - radii)
        
        return min(min_dist / 2.0, min_boundary)

    def build_ring_config(n_rings, points_per_ring):
        """
        Build a ring-based configuration.
        
        Parameters:
        -----------
        n_rings : int
            Number of rings (excluding center)
        points_per_ring : list of int
            Number of points on each ring
        
        Returns:
        --------
        points : ndarray of shape (N, 2)
            The 2D coordinates of all points
        """
        points = []
        
        # Add center point if we have room
        total_without_center = sum(points_per_ring)
        if total_without_center < N:
            points.append([0.0, 0.0])
        
        # Add points on each ring
        for k, n_k in enumerate(points_per_ring):
            if n_k <= 0:
                continue
            # Radius: distribute rings evenly from center to near boundary
            r_ring = 0.95 * math.sqrt((k + 1) / (n_rings + 1))
            for m in range(n_k):
                angle = 2 * np.pi * m / n_k
                x = r_ring * math.cos(angle)
                y = r_ring * math.sin(angle)
                points.append([x, y])
        
        points_arr = np.array(points)
        
        # Adjust if we have too many or too few points
        if len(points_arr) > N:
            points_arr = points_arr[:N]
        elif len(points_arr) < N:
            # Pad with random points near boundary
            extra = N - len(points_arr)
            rng = np.random.default_rng(42)
            new_pts = rng.randn(extra, 2) * 0.9
            points_arr = np.vstack([points_arr, new_pts])
        
        return points_arr

    def ring_parameterize(params, n_rings):
        """
        Convert ring parameters to 2D points.
        
        Parameters:
        -----------
        params : ndarray of shape (2 * n_rings + 1, )
            [r_1, theta_1, r_2, theta_2, ..., r_n, theta_n, center_offset]
            where r_k is the radius of ring k, theta_k is the angular offset
        n_rings : int
            Number of rings
        
        Returns:
        --------
        points : ndarray of shape (N, 2)
        """
        points = []
        
        # Determine points per ring based on optimal distribution
        # For a circle packing, the number of points on ring k should be 
        # proportional to the circumference, so n_k ~ 2*pi*r_k / (2*r_circle)
        # But we need exactly N-1 points (excluding center)
        
        # First, extract ring parameters
        ring_radii = params[0::2][:n_rings]
        ring_offsets = params[1::2][:n_rings]
        
        # Compute optimal points per ring based on radius
        # n_k should be proportional to r_k to maintain roughly equal spacing
        total_weight = np.sum(ring_radii)
        if total_weight > 0:
            n_per_ring = np.round((ring_radii / total_weight) * (N - 1)).astype(int)
        else:
            n_per_ring = np.ones(n_rings, dtype=int)
        
        # Adjust to ensure sum is exactly N-1
        diff = (N - 1) - np.sum(n_per_ring)
        if diff > 0:
            # Add points to rings with largest fractional parts
            frac_parts = (ring_radii / total_weight) * (N - 1) - n_per_ring
            idx = np.argsort(frac_parts)[::-1][:diff]
            n_per_ring[idx] += 1
        elif diff < 0:
            # Remove points from rings with smallest counts (but keep at least 6 per ring)
            for i in range(abs(diff)):
                # Find ring with most points (but > 6)
                valid = np.where(n_per_ring > 6)[0]
                if len(valid) > 0:
                    idx = valid[np.argmax(n_per_ring[valid])]
                    n_per_ring[idx] -= 1
        
        # Ensure minimum 6 points per ring
        n_per_ring = np.maximum(n_per_ring, 6)
        
        # Re-adjust if needed
        diff = (N - 1) - np.sum(n_per_ring)
        if diff > 0:
            idx = np.argsort(n_per_ring)[::-1][:diff]
            n_per_ring[idx] += 1
        elif diff < 0:
            idx = np.argsort(n_per_ring)[::-1][:abs(diff)]
            n_per_ring[idx] -= 1
        
        # Build points
        for k in range(n_rings):
            n_k = n_per_ring[k]
            r_k = ring_radii[k]
            theta_k = ring_offsets[k]
            for m in range(n_k):
                angle = 2 * np.pi * m / n_k + theta_k
                x = r_k * math.cos(angle)
                y = r_k * math.sin(angle)
                points.append([x, y])
        
        # Add center
        points.append([0.0, 0.0])
        
        points_arr = np.array(points)
        
        # Ensure exactly N points
        if len(points_arr) > N:
            points_arr = points_arr[:N]
        elif len(points_arr) < N:
            extra = N - len(points_arr)
            rng = np.random.default_rng(123)
            new_pts = rng.randn(extra, 2) * 0.9
            points_arr = np.vstack([points_arr, new_pts])
        
        return points_arr

    def ring_optimize(n_rings):
        """
        Optimize ring configuration using parameterized approach.
        """
        # Initial parameters
        ring_radii = np.linspace(0.1, 0.95, n_rings)
        ring_offsets = np.zeros(n_rings)
        params0 = np.zeros(2 * n_rings)
        params0[0::2] = ring_radii
        params0[1::2] = ring_offsets
        
        def objective(params):
            points = ring_parameterize(params, n_rings)
            # Penalize points outside container
            norms = np.linalg.norm(points, axis=1)
            if np.any(norms > 1.0):
                return -1.0
            score = compute_score(points)
            return -score  # Minimize negative score
        
        # Optimize
        result = minimize(objective, params0, method='L-BFGS-B', 
                         bounds=[(0.01, 0.99)] * n_rings + [(-np.pi, np.pi)] * n_rings,
                         options={'maxiter': 100, 'ftol': 1e-10})
        
        points = ring_parameterize(result.x, n_rings)
        score = compute_score(points)
        
        return points, score, result

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

    # Try different numbers of rings
    best_points = None
    best_score = -1
    
    # Try ring-based optimization with different numbers of rings
    for n_rings in [15, 20, 25, 30]:
        if time.time() > deadline - 30:
            break
            
        try:
            points, score, result = ring_optimize(n_rings)
            if score > best_score:
                best_score = score
                best_points = points
        except Exception as e:
            pass
    
    # Also try traditional hexagonal initialization as fallback
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

    if best_points is None or best_score < 0.05:
        pts_hex = initial_hex(N)
        score_hex = compute_score(pts_hex)
        if score_hex > best_score:
            best_score = score_hex
            best_points = pts_hex
    
    # Final force-directed optimization
    if time.time() < deadline - 20 and best_points is not None:
        best_points, best_score = force_optimize(best_points, max_time=100.0)
        
    return best_points
# EVOLVE-BLOCK-END

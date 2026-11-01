"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    def compute_score(pts):
        """Compute the packing radius for given centers using vectorized operations."""
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        
        # Radial distances
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return -1.0
        
        # Pairwise distances - use efficient chunked computation
        # For 300 points, we can compute all pairwise distances
        # But to save memory, compute in chunks
        min_dist = np.inf
        chunk_size = 100
        for i in range(0, N, chunk_size):
            j = min(i + chunk_size, N)
            # Compute distances from pts[i:j] to pts[j:]
            for ii in range(i, j):
                d = np.linalg.norm(pts[ii+1:] - pts[ii], axis=1)
                if len(d) > 0:
                    min_dist = min(min_dist, np.min(d))
        
        r = min(min_dist / 2.0, 1.0 - np.max(dists))
        return r
    
    def compute_score_fast(pts):
        """Faster score computation for optimization loops."""
        if not np.all(np.isfinite(pts)):
            return -1.0
        
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return -1.0
        
        # Use a more efficient pairwise distance computation
        # Only check a subset of pairs if needed
        min_dist = np.inf
        for i in range(N):
            d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
            if len(d) > 0:
                local_min = np.min(d)
                if local_min < min_dist:
                    min_dist = local_min
        
        r = min(min_dist / 2.0, 1.0 - np.max(dists))
        return r
    
    def generate_hexagonal_packing():
        """Generate a hexagonal lattice packing."""
        # Estimate spacing for N circles in unit circle
        # Area of unit circle = pi, area per circle ~ pi/N
        # Hexagonal packing density ~ 0.9069
        # So spacing ~ sqrt(2*sqrt(3)*pi/(N*0.9069))
        spacing = np.sqrt(2 * np.sqrt(3) * np.pi / (N * 0.9069))
        
        points = []
        # Generate hexagonal lattice points
        max_r = 1.0 / spacing + 2
        for j in range(int(-max_r), int(max_r) + 1):
            for i in range(int(-max_r), int(max_r) + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3) / 2
                if np.hypot(x, y) <= 1.0 - spacing/2:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) == 0:
            return None
        
        # If we have too many, keep the ones closest to center
        if len(points) > N:
            radii = np.linalg.norm(points, axis=1)
            idx = np.argsort(radii)[:N]
            points = points[idx]
        
        if len(points) < N:
            return None
        
        return points
    
    def generate_ring_packing(ring_counts, ring_radii, angular_offsets):
        """Generate points from ring configuration."""
        points = []
        for i, (n_circ, r_ring, offset) in enumerate(zip(ring_counts, ring_radii, angular_offsets)):
            if n_circ == 0:
                continue
            if n_circ == 1:
                if r_ring < 1e-9:
                    points.append([0.0, 0.0])
                else:
                    points.append([r_ring * np.cos(offset), r_ring * np.sin(offset)])
            else:
                angles = offset + np.linspace(0, 2*np.pi, n_circ, endpoint=False)
                for a in angles:
                    points.append([r_ring * np.cos(a), r_ring * np.sin(a)])
        
        points = np.array(points)
        if len(points) != N:
            return None
        return points
    
    def optimize_with_auxiliary_t(pts_init, max_iter=500):
        """Optimize using auxiliary variable t with smooth penalty.
        
        We want to maximize t subject to:
        - |c_i - c_j| >= 2t for all i < j
        - |c_i| <= 1 - t for all i
        
        Using penalty method: minimize -t + lambda * sum(max(0, 2t - |c_i-c_j|)^2) + lambda * sum(max(0, |c_i| + t - 1)^2)
        """
        if time.time() > deadline - 5:
            return pts_init, compute_score(pts_init)
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            
            t = params[0]
            pts = params[1:].reshape(-1, 2)
            
            if not np.all(np.isfinite(pts)) or not np.isfinite(t):
                return 1e10
            
            if t < 0:
                t = 0
            
            # Radial constraint penalty
            dists = np.linalg.norm(pts, axis=1)
            radial_violation = np.maximum(0, dists + t - 1.0)
            radial_penalty = np.sum(radial_violation ** 2)
            
            # Pairwise distance constraint penalty
            pair_violations = []
            for i in range(N):
                d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
                if len(d) > 0:
                    viol = np.maximum(0, 2*t - d)
                    pair_violations.append(np.sum(viol ** 2))
            
            pair_penalty = sum(pair_violations) if pair_violations else 0
            
            # Total objective: minimize -t + penalty
            lambda_val = 100.0
            obj = -t + lambda_val * (radial_penalty + pair_penalty)
            
            return obj
        
        # Initial t estimate
        pts = pts_init.copy()
        dists = np.linalg.norm(pts, axis=1)
        min_pair_dist = np.inf
        for i in range(N):
            d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
            if len(d) > 0:
                min_pair_dist = min(min_pair_dist, np.min(d))
        
        t_init = min(min_pair_dist / 2.0, 1.0 - np.max(dists)) * 0.95
        t_init = max(0.01, min(t_init, 0.08))
        
        x0 = np.concatenate([[t_init], pts.flatten()])
        
        # Bounds: t in [0.001, 0.08], coords in [-0.999, 0.999]
        bounds = [(0.001, 0.08)] + [(-0.999, 0.999)] * (2 * N)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': max_iter, 'maxfun': max_iter * 5, 
                                      'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < 1e10:
                t_opt = result.x[0]
                pts_opt = result.x[1:].reshape(-1, 2)
                score = compute_score(pts_opt)
                return pts_opt, score
        except Exception:
            pass
        
        return pts_init, compute_score(pts_init)
    
    def polish(pts, max_iter=200):
        """Final polish with tighter tolerance."""
        if time.time() > deadline - 3:
            return pts, compute_score(pts)
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            
            pts = params.reshape(-1, 2)
            
            if not np.all(np.isfinite(pts)):
                return 1e10
            
            dists = np.linalg.norm(pts, axis=1)
            if np.any(dists > 1.0):
                return 1e10
            
            min_dist = np.inf
            for i in range(N):
                d = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
                if len(d) > 0:
                    min_dist = min(min_dist, np.min(d))
            
            r = min(min_dist / 2.0, 1.0 - np.max(dists))
            return -r
        
        x0 = pts.flatten()
        bounds = [(-0.999, 0.999)] * (2 * N)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': max_iter, 'maxfun': max_iter * 3,
                                      'ftol': 1e-14, 'gtol': 1e-12})
            if result.fun < 1e10:
                new_pts = result.x.reshape(-1, 2)
                new_score = compute_score(new_pts)
                if new_score > compute_score(pts):
                    return new_pts, new_score
        except Exception:
            pass
        
        return pts, compute_score(pts)
    
    # Strategy 1: Start with hexagonal lattice
    best_pts = generate_hexagonal_packing()
    if best_pts is None:
        # Fallback: simple grid
        best_pts = np.zeros((N, 2))
        n_per_row = int(np.sqrt(N))
        spacing = 0.03
        for i in range(N):
            row = i // n_per_row
            col = i % n_per_row
            best_pts[i, 0] = (col - n_per_row/2) * spacing
            best_pts[i, 1] = (row - n_per_row/2) * spacing
    
    best_score = compute_score(best_pts)
    
    # Strategy 2: Try ring configurations
    ring_configs = [
        # (counts, radii, offsets)
        (np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29]), 
         np.linspace(0.02, 0.98, 11), np.zeros(11)),
        (np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 29]),
         np.linspace(0.02, 0.98, 12), np.zeros(12)),
        (np.array([1, 8, 16, 24, 32, 40, 48, 56, 75]),
         np.linspace(0.05, 0.98, 9), np.zeros(9)),
    ]
    
    for counts, radii, offsets in ring_configs:
        if time.time() > deadline - 30:
            break
        
        # Adjust counts to sum to N
        counts = counts.copy()
        diff = N - np.sum(counts)
        for i in range(abs(diff)):
            if diff > 0:
                counts[i % len(counts)] += 1
            else:
                counts[i % len(counts)] -= 1
        counts = np.maximum(1, counts)
        
        pts = generate_ring_packing(counts, radii, offsets)
        if pts is not None:
            score = compute_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts
    
    # Strategy 3: Optimize with auxiliary t variable
    if time.time() < deadline - 20:
        opt_pts, opt_score = optimize_with_auxiliary_t(best_pts, max_iter=300)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts
    
    # Strategy 4: Try multiple random restarts if time permits
    if time.time() < deadline - 15:
        np.random.seed(42)
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            # Create perturbed version of best solution
            pts_perturbed = best_pts + np.random.randn(N, 2) * 0.01
            # Project back inside circle
            dists = np.linalg.norm(pts_perturbed, axis=1)
            mask = dists > 0.99
            pts_perturbed[mask] = pts_perturbed[mask] / dists[mask, None] * 0.99
            
            score = compute_score(pts_perturbed)
            if score > best_score:
                best_score = score
                best_pts = pts_perturbed
            
            # Quick optimization
            if time.time() < deadline - 8:
                opt_pts, opt_score = optimize_with_auxiliary_t(pts_perturbed, max_iter=100)
                if opt_score > best_score:
                    best_score = opt_score
                    best_pts = opt_pts
    
    # Final polish
    if time.time() < deadline - 5:
        final_pts, final_score = polish(best_pts, max_iter=150)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts
    
    # Final validation
    final_score = compute_score(best_pts)
    if final_score <= 0 or not np.all(np.isfinite(best_pts)):
        # Return a valid fallback
        best_pts = np.zeros((N, 2))
        n_per_row = int(np.sqrt(N))
        spacing = 0.03
        for i in range(N):
            row = i // n_per_row
            col = i % n_per_row
            best_pts[i, 0] = (col - n_per_row/2) * spacing
            best_pts[i, 1] = (row - n_per_row/2) * spacing
    
    return best_pts
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # First, get a good initial packing using hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
    
    # Vectorized score computation
    def compute_score(pts):
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return -1.0
        # Vectorized pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, np.inf)
        min_dist = np.sqrt(np.min(d2))
        r = min(min_dist / 2.0, 1.0 - np.max(dists))
        return r
    
    # Fast score using only critical constraints (k smallest pair dists + max radial)
    def compute_score_fast(pts, k=10):
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return -1.0
        # Only compute k smallest pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, np.inf)
        # Get k smallest off-diagonal entries
        flat_d2 = d2.ravel()
        # Exclude diagonal (N entries)
        mask = ~np.eye(N, dtype=bool).ravel()
        off_diag = flat_d2[mask]
        if len(off_diag) < k:
            min_dist = np.sqrt(np.min(off_diag)) if len(off_diag) > 0 else 0.0
        else:
            kth_smallest = np.partition(off_diag, k)[:k]
            min_dist = np.sqrt(np.min(kth_smallest))
        r = min(min_dist / 2.0, 1.0 - np.max(dists))
        return r
    
    initial_score = compute_score(points)
    if initial_score <= 0:
        return points
    
    best_points = points.copy()
    best_score = initial_score
    
    # Smooth soft-min objective using log-sum-exp for the min operation
    # The true objective is min(min_pair_dist/2, 1 - max_radius)
    # We use a smooth approximation: -softmin(beta, min_pair_dist/2, 1 - max_radius)
    # where softmin uses log-sum-exp on the negative values
    
    def soft_min_beta(x, y, beta):
        """Smooth approximation to min(x, y) using log-sum-exp."""
        # min(x,y) = -max(-x, -y)
        # max(a,b) = (1/beta) * log(exp(beta*a) + exp(beta*b))
        a, b = -x, -y
        m = np.maximum(a, b)
        if m > 500:
            return -m
        if m < -500:
            return -(m + 1.0/beta)
        return -(m + np.log(np.exp(beta*(a-m)) + np.exp(beta*(b-m))) / beta)
    
    def smooth_objective(params, pts_fixed, beta):
        if time.time() > deadline:
            return 1e10
        pts = pts_fixed.copy()
        n_opt = len(params) // 2
        pts[:n_opt] = params.reshape(-1, 2)
        
        # Check inside container
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return 1e10
        
        # Use fast computation with only critical constraints
        k_crit = 5  # Track 5 smallest pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, np.inf)
        
        # Get k smallest off-diagonal squared distances
        flat_d2 = d2.ravel()
        mask = ~np.eye(N, dtype=bool).ravel()
        off_diag = flat_d2[mask]
        if len(off_diag) < k_crit:
            min_d2 = np.min(off_diag) if len(off_diag) > 0 else 0.0
        else:
            kth_smallest = np.partition(off_diag, k_crit)[:k_crit]
            min_d2 = np.min(kth_smallest)
        
        min_dist = np.sqrt(max(min_d2, 1e-15))
        
        x = min_dist / 2.0
        y = 1.0 - np.max(dists)
        
        # Return negative of soft-min (we're minimizing)
        return -soft_min_beta(x, y, beta)
    
    # Multi-stage optimization with annealing beta
    # Start with low beta (smooth landscape) and increase to high beta (sharp)
    betas = [5.0, 20.0, 100.0, 500.0, 2000.0]
    
    # Optimize all N points (vectorized distance makes this feasible)
    n_opt = N
    opt_indices = np.arange(n_opt)
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
        
        x0 = points.flatten()
        bounds = [(-0.99, 0.99)] * (2 * n_opt)
        
        try:
            result = minimize(
                smooth_objective, x0, args=(points, beta),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 30, 'maxfun': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < 1e10:
                new_points = result.x.reshape(-1, 2)
                new_score = compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                    points = new_points.copy()
        except Exception:
            pass
    
    # If we have time, try a more global approach with different ring configurations
    if time.time() < deadline - 30:
        # Try radial ring approach
        best_ring_points = None
        best_ring_score = best_score
        
        for n_rings in range(5, 15):
            if time.time() > deadline - 10:
                break
            
            ring_radii = np.linspace(0.1, 0.95, n_rings)
            ring_points = []
            
            for r_ring in ring_radii:
                est_r = 0.05
                n_circ = max(1, int(2 * np.pi * r_ring / (2 * est_r)))
                angles = np.linspace(0, 2*np.pi, n_circ, endpoint=False)
                for a in angles:
                    x = r_ring * np.cos(a)
                    y = r_ring * np.sin(a)
                    ring_points.append([x, y])
            
            ring_points.append([0.0, 0.0])
            ring_points = np.array(ring_points)
            
            if len(ring_points) > N:
                dists = np.linalg.norm(ring_points, axis=1)
                idx = np.argsort(dists)[:N]
                ring_points = ring_points[idx]
            elif len(ring_points) < N:
                continue
            
            score = compute_score(ring_points)
            if score > best_ring_score:
                best_ring_score = score
                best_ring_points = ring_points.copy()
        
        if best_ring_points is not None:
            best_points = best_ring_points
            best_score = best_ring_score
    
    # Final polish with very high beta on the best points found
    if time.time() < deadline - 10:
        points = best_points.copy()
        # Use progressively higher beta and tighter tolerances
        polish_betas = [500.0, 2000.0, 5000.0, 10000.0]
        for beta in polish_betas:
            if time.time() > deadline - 5:
                break
            x0 = points.flatten()
            bounds = [(-0.99, 0.99)] * (2 * N)
            try:
                result = minimize(
                    smooth_objective, x0, args=(points, beta),
                    method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 50, 'maxfun': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if result.fun < 1e10:
                    new_points = result.x.reshape(-1, 2)
                    new_score = compute_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
                        points = new_points.copy()
            except Exception:
                pass
        
        # Final exact local solve: iteratively push critical constraints
        if time.time() < deadline - 3:
            for _ in range(5):
                if time.time() > deadline - 2:
                    break
                pts = best_points.copy()
                # Compute current score
                score = compute_score(pts)
                if score <= 0:
                    break
                
                # Find critical constraints
                dists = np.linalg.norm(pts, axis=1)
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                d2 = np.sum(diff * diff, axis=2)
                np.fill_diagonal(d2, np.inf)
                
                # Radial constraint violation
                radial_violation = np.max(dists) - (1.0 - score)
                if radial_violation > 0:
                    # Push points inward
                    norm_dists = np.linalg.norm(pts, axis=1, keepdims=True)
                    norm_dists[norm_dists < 1e-10] = 1.0
                    pts = pts - 0.5 * radial_violation * pts / norm_dists
                
                # Pairwise constraint
                min_d2 = np.min(d2)
                min_dist = np.sqrt(min_d2)
                pair_violation = score - min_dist / 2.0
                if pair_violation > 0:
                    # Find closest pair and separate them
                    min_idx = np.argmin(d2)
                    i, j = np.unravel_index(min_idx, (N, N))
                    if i != j:
                        d_vec = pts[j] - pts[i]
                        d_norm = np.linalg.norm(d_vec)
                        if d_norm > 1e-10:
                            push = 0.5 * pair_violation
                            pts[i] -= push * d_vec / d_norm
                            pts[j] += push * d_vec / d_norm
                
                new_score = compute_score(pts)
                if new_score > best_score:
                    best_score = new_score
                    best_points = pts.copy()
    
    # Final validation
    final_score = compute_score(best_points)
    if final_score <= 0:
        return points
    
    return best_points
# EVOLVE-BLOCK-END

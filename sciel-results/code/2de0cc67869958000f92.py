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
        
        # Vectorized pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, np.inf)
        min_dist = np.sqrt(np.min(d2))
        
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
    if time.time() < deadline - 5:
        points = best_points.copy()
        for beta in [1000.0, 5000.0]:
            if time.time() > deadline - 3:
                break
            x0 = points.flatten()
            bounds = [(-0.99, 0.99)] * (2 * N)
            try:
                result = minimize(
                    smooth_objective, x0, args=(points, beta),
                    method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 20, 'maxfun': 100, 'ftol': 1e-12, 'gtol': 1e-10}
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
    
    # Final validation
    final_score = compute_score(best_points)
    if final_score <= 0:
        return points
    
    return best_points
# EVOLVE-BLOCK-END

import numpy as np
import time
from scipy.optimize import minimize
import itertools

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 10.0
    N = 16

    def get_cost(pts):
        """Calculate (d_max / d_min)^2. Lower is better."""
        # Use pdist for efficiency, or manual calculation
        # Manual is fast enough for N=16
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        dist_sq = np.where(mask, dist_sq, np.inf)
        
        d_min_sq = np.min(dist_sq)
        d_max_sq = np.max(dist_sq)
        
        if d_min_sq <= 1e-12 or d_max_sq <= 1e-12:
            return np.inf
            
        return d_max_sq / d_min_sq

    def get_cost_smooth(pts, k=10.0):
        """
        Smoothed cost function for gradient-based methods.
        Approximates (d_max / d_min)^2 using soft min/max.
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        
        mask = ~np.eye(N, dtype=bool)
        dist_sq = np.where(mask, dist_sq, 0)
        dist = np.sqrt(dist_sq)
        
        # Soft min and max of distances
        # min(d) ~ -1/k * ln(sum(exp(-k*d)))
        # max(d) ~ 1/k * ln(sum(exp(k*d)))
        
        # To avoid overflow, subtract max/min
        d_max = np.max(dist)
        d_min = np.min(dist[dist > 1e-9]) if np.any(dist > 1e-9) else 1e-9
        
        # If points are coincident, cost is inf
        if d_min < 1e-6:
            return 1e10
            
        # Normalized soft min/max
        # We want to minimize (d_max / d_min)^2
        # log R = 2 * (log d_max - log d_min)
        
        # Soft log d_max = log( (1/k) * sum(exp(k*d)) ) = log(sum(exp(k*d))) - log(k)
        # Soft log d_min = log( (1/k) * sum(exp(-k*d)) ) ... wait, min is -max(-d)
        # log d_min ~ -log( (1/k) * sum(exp(-k*d)) ) = -log(sum(exp(-k*d))) + log(k)
        
        # This is getting complicated. Let's just use Nelder-Mead on the exact cost.
        # It is robust for N=16.
        pass

    def hexagonal_lattice_init(n_points=16):
        """Generate a hexagonal lattice and select the n_points closest to the centroid."""
        R = 5
        points = []
        a = 1.0
        b = np.sqrt(3) / 2
        for i in range(-R, R + 1):
            for j in range(-R, R + 1):
                x = i * a + j * 0.5 * a
                y = j * b
                points.append([x, y])
        
        points = np.array(points)
        
        centroid = np.mean(points, axis=0)
        dists = np.linalg.norm(points - centroid, axis=1)
        idx = np.argsort(dists)
        selected = points[idx[:n_points]]
        
        selected = selected - np.mean(selected, axis=0)
        return selected

    def grid_init(n_points=16):
        """Generate a square grid."""
        side = int(np.ceil(np.sqrt(n_points)))
        g = np.array([(i, j) for i in range(side) for j in range(side)], dtype=float)
        g = g - np.mean(g, axis=0)
        if len(g) > n_points:
            centroid = np.mean(g, axis=0)
            dists = np.linalg.norm(g - centroid, axis=1)
            idx = np.argsort(dists)
            g = g[idx[:n_points]]
        return g

    def force_relaxed_init(n_points=16, iterations=500, seed=42):
        """
        Initialize points using a repulsive force simulation.
        This helps escape bad local minima of the grid/lattice.
        """
        np.random.seed(seed)
        pts = np.random.randn(n_points, 2)
        
        # Repulsion constant
        k_rep = 1.0
        # Attraction to center constant
        k_att = 0.1
        # Learning rate
        lr = 0.1
        
        for i in range(iterations):
            # Calculate pairwise distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=-1)
            dist = np.sqrt(dist_sq + 1e-9) # avoid div by zero
            
            # Forces
            forces = np.zeros_like(pts)
            
            # Repulsive forces (Coulomb-like)
            # F_ij = k_rep * (p_i - p_j) / d_ij^3
            # We sum forces on i
            # dist_sq is (N, N)
            # We want to compute sum_j k_rep * (p_i - p_j) / d_ij^3
            # This is equivalent to:
            # F_i = sum_j k_rep * p_i / d_ij^3 - sum_j k_rep * p_j / d_ij^3
            
            # Avoid self-interaction
            np.fill_diagonal(dist_sq, np.inf)
            dist_cubed = np.power(dist, 3)
            
            # Sum of p_j / d_ij^3
            # Weight matrix W_ij = k_rep / d_ij^3
            W = k_rep / dist_cubed
            W = np.where(np.isfinite(W), W, 0)
            
            # F_rep_i = sum_j W_ij * (p_i - p_j) = p_i * sum_j W_ij - sum_j W_ij * p_j
            row_sums = np.sum(W, axis=1)
            rep_force = row_sums[:, np.newaxis] * pts - (W @ pts)
            
            # Attractive force to center
            att_force = -k_att * pts
            
            total_force = rep_force + att_force
            
            # Update positions
            pts += lr * total_force
            
            # Normalize to prevent explosion
            pts = pts - np.mean(pts, axis=0)
            norm = np.linalg.norm(pts, axis=1)
            norm = np.max(norm)
            if norm > 0:
                pts = pts / norm
                
        return pts

    def polish(pts, time_budget):
        """
        Refine points using Nelder-Mead.
        """
        def objective(params):
            p = params.reshape(N, 2)
            return get_cost(p)
            
        initial_params = pts.flatten()
        
        try:
            res = minimize(objective, initial_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if hasattr(res, 'fun') and np.isfinite(res.fun):
                res_pts = res.x.reshape(N, 2)
                res_cost = get_cost(res_pts)
                return res_pts, res_cost
        except Exception:
            pass
            
        return pts, get_cost(pts)

    # Generate initial candidates
    candidates = [
        hexagonal_lattice_init(N),
        grid_init(N),
        np.random.randn(N, 2),
        force_relaxed_init(N, iterations=200, seed=1),
        force_relaxed_init(N, iterations=200, seed=2),
        force_relaxed_init(N, iterations=200, seed=3),
    ]
    
    # Normalize initial candidates
    for i in range(len(candidates)):
        c = candidates[i]
        c = c - np.mean(c, axis=0)
        norm = np.linalg.norm(c, axis=1)
        max_norm = np.max(norm)
        if max_norm > 0:
            c = c / max_norm
        candidates[i] = c

    best_pts = None
    best_cost = np.inf
    
    # Evaluate initial candidates
    for c in candidates:
        cost = get_cost(c)
        if cost < best_cost:
            best_cost = cost
            best_pts = c.copy()

    time_elapsed = time.time() - start_time
    time_per_polish = 10.0 # Estimate time for one Nelder-Mead run
    
    # Polish candidates one by one
    for i, c in enumerate(candidates):
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Set a time limit for this polish
        # We can't easily interrupt minimize, so we rely on maxiter
        # Or we can run in a separate process? No, too complex.
        # We will just run it and hope it's fast. N=16 is small.
        
        polished_pts, polished_cost = polish(c, time_per_polish)
        
        if polished_cost < best_cost:
            best_cost = polished_cost
            best_pts = polished_pts.copy()

    # If we still have time, try more random starts
    time_elapsed = time.time() - start_time
    remaining_time = TIME_LIMIT - time_elapsed - SAFETY_MARGIN
    
    if remaining_time > 20:
        # Try differential evolution if time permits?
        # DE is slow. Let's do more local searches from random points.
        
        n_extra_starts = int(remaining_time / 5.0) # 5s per start estimate
        n_extra_starts = min(n_extra_starts, 50) # Cap at 50 extra starts
        
        for k in range(n_extra_starts):
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
                
            # Random start
            c = np.random.randn(N, 2)
            c = c - np.mean(c, axis=0)
            norm = np.linalg.norm(c, axis=1)
            max_norm = np.max(norm)
            if max_norm > 0:
                c = c / max_norm
                
            polished_pts, polished_cost = polish(c, 5.0)
            
            if polished_cost < best_cost:
                best_cost = polished_cost
                best_pts = polished_pts.copy()

    # Final check
    if best_pts is None:
        best_pts = grid_init(N)
        
    if not np.all(np.isfinite(best_pts)):
        best_pts = grid_init(N)

    return best_pts

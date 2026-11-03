"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _voronoi_inradius(pts, t=1.0):
    """
    Compute a smooth approximation of the minimum inradius of the Voronoi cells.
    For each point i, the inradius is the minimum distance to:
    1. The walls of the square [-0.5, 0.5]^2
    2. The perpendicular bisectors with other points j (approximated by d_ij/2)
    
    We use a soft-min (log-sum-exp) to make the objective differentiable.
    The 'radius' of the packing is min_i (inradius_i).
    We maximize min_i (inradius_i).
    To do this, we minimize -min_i (inradius_i).
    Using soft-min: -softmin_i (inradius_i) approximates -min_i (inradius_i).
    """
    n = len(pts)
    if n == 0:
        return 0.0
    
    # Distances to walls for each point
    # x in [-0.5, 0.5], y in [-0.5, 0.5]
    dx = np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5)
    dy = np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    dist_walls = np.minimum(dx, dy)
    
    # Distances to other points (bisector distance is half the distance between centers)
    # We need the minimum over j != i of dist(i, j) / 2
    # Vectorized: compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity so it doesn't affect the min
    np.fill_diagonal(dists, np.inf)
    
    # The constraint for point i is min( dist_wall_i, min_j(dist_ij/2) )
    # Let r_i = min( dist_wall_i, min_j(dist_ij/2) )
    # We want to maximize min_i r_i
    
    # Combine wall distance and neighbor distances for each point
    # For each i, we have a vector of constraints: [dist_wall_i, dist_i1/2, dist_i2/2, ...]
    # We want min of these.
    # Create a matrix C where C[i, 0] = dist_wall_i, C[i, 1:] = dists[i, :] / 2
    C = np.zeros((n, n + 1))
    C[:, 0] = dist_walls
    C[:, 1:] = dists / 2.0
    
    # Soft min for each row i: -1/t * log(sum(exp(-t * C[i, :])))
    # This approximates min(C[i, :])
    # We want to maximize the minimum of these soft mins.
    # So we compute soft_min_i = -1/t * log(sum(exp(-t * C[i, :])))
    # Then we want to maximize min_i(soft_min_i).
    # To make it differentiable, we use another soft min? Or just sum?
    # Actually, maximizing min_i(r_i) is equivalent to maximizing the smallest radius.
    # A common trick is to maximize the sum of radii with a large weight on the minimum,
    # or use a barrier.
    # However, a simpler differentiable proxy for "maximize min" is to maximize the
    # soft-min of the radii themselves.
    # Let R_i = soft_min_j(C[i, j]).
    # We want to maximize min_i(R_i).
    # Let's use a soft-min over i as well.
    # Objective to minimize: - soft_min_i(R_i)
    
    # Compute R_i
    # To avoid overflow, subtract max
    # C can be negative if points are outside? No, we enforce bounds or assume valid.
    # But during optimization, points might go outside.
    # If a point is outside, dist_wall is negative.
    # This is a feature: it penalizes going outside.
    
    # t controls sharpness. Larger t -> closer to true min.
    # Start with moderate t.
    
    # For numerical stability in exp:
    # soft_min = -1/t * log(sum(exp(-t * x)))
    # = -1/t * ( -t * max(x) + log(sum(exp(-t * (x - max(x))))))
    # = max(x) - 1/t * log(sum(exp(-t * (x - max(x)))))
    
    max_C = np.max(C, axis=1, keepdims=True)
    exp_term = np.exp(-t * (C - max_C))
    log_sum = np.log(np.sum(exp_term, axis=1))
    R = max_C[:, 0] - (1.0 / t) * log_sum
    
    # Now maximize min(R).
    # Use soft-min again for the final objective.
    # obj = - soft_min(R)
    max_R = np.max(R)
    exp_R = np.exp(-t * (R - max_R))
    log_sum_R = np.log(np.sum(exp_R))
    soft_min_R = max_R - (1.0 / t) * log_sum_R
    
    return -soft_min_R

def _make_hex_grid(n):
    """Generate a hexagonal grid of n points centered at origin."""
    # Approximate number of rows/cols
    # Hex packing density is higher.
    # For n=50, a 7x7 hex grid has 49 points. 8x7 has 56.
    # Let's generate a grid and take the first n.
    pts = []
    # Hex spacing
    # We want to fit in [-0.5, 0.5]
    # Let's just generate a large grid and center it.
    rows = int(math.ceil(math.sqrt(n * 2 / math.sqrt(3)))) + 2
    cols = rows
    # Spacing
    # If we have 'rows' rows, height is rows * sqrt(3)/2 * s
    # Width is cols * s
    # We want max(width, height) <= 1.0
    # s * cols <= 1.0 and s * rows * sqrt(3)/2 <= 1.0
    # s <= min(1.0/cols, 1.0/(rows * sqrt(3)/2))
    s = min(1.0 / cols, 1.0 / (rows * math.sqrt(3) / 2)) * 0.9 # 0.9 for margin
    
    for r in range(rows):
        for c in range(cols):
            x = (c - cols / 2.0) * s
            y = (r - rows / 2.0) * s * math.sqrt(3) / 2
            if r % 2 == 1:
                x += s / 2.0
            pts.append([x, y])
    
    pts = np.array(pts)
    # Sort by distance to center to keep the most central ones?
    # Or just take first N? Taking first N from a grid generation order might be biased.
    # Better to generate more, then select the N closest to origin.
    dists_to_origin = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_origin)[:n]
    return pts[indices]

def construct():
    start_time = time.time()
    timeout = 110.0  # Leave 10s margin
    
    # Initialize with hex grid
    pts = _make_hex_grid(N)
    
    # Optimize using L-BFGS-B
    # Objective: minimize -soft_min_radius
    # Constraints: none explicitly, but the objective penalizes overlap/out-of-bounds
    # We can add bounds to L-BFGS-B to keep points inside [-0.5, 0.5]
    # This helps stability.
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    from scipy.optimize import minimize
    
    def objective(x):
        p = x.reshape(N, 2)
        return _voronoi_inradius(p, t=50.0)
    
    def objective_grad(x):
        # Finite difference gradient for simplicity and robustness
        # Or implement analytical? Finite diff is safer for this complex soft-min.
        # 100 dimensions. 100 evaluations per step.
        # L-BFGS-B might take 100-200 steps. 100*200 = 20000 evals.
        # Each eval is O(N^2) = 2500 ops. 20000 * 2500 = 50M ops.
        # Should be fast enough in Python/NumPy?
        # Let's check time.
        h = 1e-5
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * h)
        return grad
    
    # Use numerical gradient in minimize to save code complexity, 
    # but it's slow. Let's try to implement a faster gradient or just use 
    # a different optimizer that doesn't need gradients? 
    # Nelder-Mead is slow in 100D.
    # Powell is okay but no gradient.
    # Let's stick with L-BFGS-B and finite diff, but limit iterations.
    
    best_pts = pts
    best_score = _voronoi_inradius(pts, t=50.0) # This is negative of radius proxy
    
    try:
        res = minimize(objective, pts.flatten(), method='L-BFGS-B', 
                       jac=objective_grad, bounds=bounds, 
                       options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5})
        
        if res.x is not None:
            candidate = res.x.reshape(N, 2)
            # Check validity
            if np.all(np.isfinite(candidate)):
                score = _voronoi_inradius(candidate, t=50.0)
                if score < best_score: # Remember we are minimizing negative radius
                    best_score = score
                    best_pts = candidate
    except Exception:
        pass
    
    # Local refinement with smaller t (sharper min) if time permits
    if time.time() - start_time < timeout - 10:
        try:
            def objective_sharp(x):
                p = x.reshape(N, 2)
                return _voronoi_inradius(p, t=200.0)
            
            def objective_grad_sharp(x):
                h = 1e-6
                grad = np.zeros_like(x)
                for i in range(len(x)):
                    x_plus = x.copy()
                    x_plus[i] += h
                    x_minus = x.copy()
                    x_minus[i] -= h
                    grad[i] = (objective_sharp(x_plus) - objective_sharp(x_minus)) / (2 * h)
                return grad
            
            res2 = minimize(objective_sharp, best_pts.flatten(), method='L-BFGS-B',
                            jac=objective_grad_sharp, bounds=bounds,
                            options={'maxiter': 30, 'ftol': 1e-10, 'gtol': 1e-6})
            
            if res2.x is not None:
                candidate = res2.x.reshape(N, 2)
                if np.all(np.isfinite(candidate)):
                    score = _voronoi_inradius(candidate, t=200.0)
                    if score < best_score:
                        best_score = score
                        best_pts = candidate
        except Exception:
            pass
    
    # Final check and return
    # Ensure points are strictly inside or on boundary
    # The objective penalizes going outside, but L-BFGS-B bounds should keep them in.
    # Clip to be safe.
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts

# EVOLVE-BLOCK-END

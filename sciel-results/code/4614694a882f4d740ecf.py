"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _hexagonal_lattice(N):
    """Generate a hexagonal lattice of points inside the unit circle."""
    points = []
    # Estimate spacing
    # Area of unit circle = pi. Area of hex cell ~ 2.6 * r^2.
    # r ~ sqrt(pi / (2.6 * N))
    # Distance between centers d ~ 2r.
    # Let's just iterate spacing.
    spacing = 0.1
    while True:
        k = int(1.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        points = np.array(rows)
        if len(points) >= N:
            # Sort by radius to take the most central ones
            radii = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(radii, kind="stable")[:N]
            return points[idx]
        spacing *= 0.95

def _compute_radius(pts):
    """Compute the max packing radius for a given set of points."""
    if len(pts) < 2:
        return 1.0
    # Distances between pairs
    # This is O(N^2), but N=300 is small enough.
    # Using broadcasting
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Distance to boundary
    radii = np.linalg.norm(pts, axis=-1)
    min_boundary = np.min(1.0 - radii)
    
    return min(min_dist / 2.0, min_boundary)

def _soft_min_objective_and_grad(pts_flat, beta):
    """
    Objective: Minimize f(x) = (1/beta) * ln(sum(exp(-beta * d_k)))
    where d_k are distances to other points and boundary.
    This approximates -min(d_k). Minimizing it maximizes min(d_k).
    """
    pts = pts_flat.reshape(-1, 2)
    N = len(pts)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d_pairs = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(d_pairs, np.inf)
    
    # Boundary distances
    r_pts = np.linalg.norm(pts, axis=-1)
    d_boundary = 1.0 - r_pts
    
    # Avoid division by zero in gradient
    eps = 1e-12
    d_pairs_safe = np.where(d_pairs < eps, eps, d_pairs)
    r_pts_safe = np.where(r_pts < eps, eps, r_pts)
    
    # Weights
    # To prevent overflow/underflow in exp, we can shift, but for beta up to 1000 and d~0.1, exp(-100) is small.
    # Sum of exps might be small.
    w_pairs = np.exp(-beta * d_pairs_safe)
    w_boundary = np.exp(-beta * d_boundary)
    
    Z = np.sum(w_pairs) + np.sum(w_boundary)
    
    if Z == 0:
        return 0.0, np.zeros_like(pts_flat)
        
    obj = (1.0 / beta) * np.log(Z)
    
    # Gradient
    # grad_i = - sum_j (w_ij / Z) * (x_i - x_j) / d_ij + (w_i_boundary / Z) * x_i / ||x_i||
    
    # Unit vectors for pairs
    # diff is x_i - x_j
    # We want (x_i - x_j) / d_ij
    unit_vecs = diff / d_pairs_safe[:, :, np.newaxis]
    
    # Weighted sum of unit vectors for each i
    # Sum over j for each i
    # w_pairs is N x N
    # We need sum_j w_ij * unit_vec_ij
    # Note: unit_vec_ij = (x_i - x_j)/d_ij
    # The term in gradient is - (w_ij / Z) * unit_vec_ij
    grad_pairs = -np.sum(w_pairs[:, :, np.newaxis] * unit_vecs, axis=1) / Z
    
    # Boundary term
    # (w_i / Z) * x_i / ||x_i||
    unit_boundary = pts / r_pts_safe[:, np.newaxis]
    grad_boundary = (w_boundary[:, np.newaxis] * unit_boundary) / Z
    
    grad = grad_pairs + grad_boundary
    
    return obj, grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Initial guess
    pts = _hexagonal_lattice(N)
    best_pts = pts.copy()
    best_radius = _compute_radius(pts)
    
    # 2. Optimization loop with increasing beta
    # We use L-BFGS-B
    # Bounds: [-1.1, 1.1] for each coordinate to allow some movement but prevent flying away
    bounds = [(-1.1, 1.1)] * (2 * N)
    
    # Betas to try
    betas = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    
    current_pts = pts.copy()
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
            
        # Prepare objective and gradient
        def obj_grad(x):
            o, g = _soft_min_objective_and_grad(x, beta)
            return o, g
        
        try:
            res = minimize(obj_grad, current_pts.flatten(), method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < 1e-10:
                new_pts = res.x.reshape(-1, 2)
                new_radius = _compute_radius(new_pts)
                
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_pts = new_pts.copy()
                    current_pts = new_pts.copy()
                    
        except Exception:
            pass
            
        # Check time
        if time.time() > deadline - 5.0:
            break
            
    # Final check
    final_radius = _compute_radius(best_pts)
    if final_radius > best_radius:
        best_radius = final_radius
        
    # Ensure validity: if any point is outside, clip? 
    # The objective penalizes going outside, but L-BFGS-B might land on boundary.
    # We should ensure the returned points are strictly inside or on the boundary.
    # If a point is outside, the score is 0.
    # Let's clip to the unit circle if necessary, but this might cause overlaps.
    # Better to rely on the optimization.
    
    return best_pts

# EVOLVE-BLOCK-END

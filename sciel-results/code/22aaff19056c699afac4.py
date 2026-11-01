"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _repulsion_energy(x):
    """
    x: shape (N, 2)
    Returns repulsion energy sum 1/d^2
    """
    # Compute pairwise distances
    # x is (N, 2)
    # diff = x[:, np.newaxis, :] - x[np.newaxis, :, :] -> (N, N, 2)
    # This is memory intensive for large N, but N=300 is 90k pairs, 180k floats. Fine.
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero on diagonal
    # Use 1/d^2, but clip small distances to avoid infinity
    # If dist_sq is very small, 1/dist_sq is huge, which is good for repulsion
    # But numerical stability:
    dist_sq = np.maximum(dist_sq, 1e-12)
    energy = np.sum(1.0 / dist_sq)
    
    # Boundary penalty: keep points inside unit circle
    # r_i = norm(x_i)
    # penalty if r_i > 1 - margin
    # We want to maximize min(1 - r_i).
    # So we want to minimize max(r_i).
    # Or just penalize r_i > 0.99
    r = np.linalg.norm(x, axis=1)
    # Soft penalty for exceeding radius 1
    # (r - 1)^2 if r > 1, else 0?
    # But we want them to stay away from boundary to allow for circle radius.
    # The score is min(d_min/2, 1 - r_max).
    # So we want to balance d_min and r_max.
    # Minimizing repulsion naturally pushes them to boundary.
    # We should add a penalty that grows as r approaches 1.
    # Let's add (r^2) to the energy? No, that pulls to center.
    # The repulsion term 1/d^2 pushes them apart.
    # If we don't constrain boundary, they fly out.
    # We MUST constrain or penalize.
    # Let's use a strong penalty for r > 1.
    over = np.maximum(0, r - 1.0)
    boundary_penalty = np.sum(over ** 2) * 1000.0
    
    return energy + boundary_penalty

def _repulsion_gradient(x):
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-12)
    
    # Gradient of sum 1/d^2 wrt x_i
    # d(1/d^2)/dx_i = -2/d^4 * (x_i - x_j)
    # Sum over j
    # Let's compute numerically or analytically.
    # Analytical:
    # inv_dist_sq = 1.0 / dist_sq
    # grad = -2.0 * np.sum(inv_dist_sq[:, np.newaxis, np.newaxis] * diff, axis=1)
    # Wait, diff[i,j] = x_i - x_j.
    # d/dx_i (1/|x_i - x_j|^2) = -2 (x_i - x_j) / |x_i - x_j|^4
    # So grad_i = sum_j -2 (x_i - x_j) / |x_i - x_j|^4
    
    inv_dist_sq_2 = 1.0 / (dist_sq ** 2)
    # Mask diagonal
    np.fill_diagonal(inv_dist_sq_2, 0.0)
    
    grad = -2.0 * np.sum(inv_dist_sq_2[:, np.newaxis, :] * diff, axis=1)
    
    # Boundary gradient
    r = np.linalg.norm(x, axis=1)
    over = np.maximum(0, r - 1.0)
    # grad of (r-1)^2 * 1000 = 2000 * (r-1) * x/r
    # only if r > 1
    mask = r > 1.0
    grad_boundary = np.zeros_like(x)
    if np.any(mask):
        grad_boundary[mask] = 2.0 * 1000.0 * (r[mask] - 1.0)[:, np.newaxis] * x[mask] / r[mask][:, np.newaxis]
    
    return grad + grad_boundary

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_score = -1.0
    best_points = None
    
    # Initialization 1: Hexagonal Lattice
    def init_hex():
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        return points[:N] # Fallback

    # Initialization 2: Random
    def init_random(seed=42):
        np.random.seed(seed)
        # Generate random points in unit disk
        # Rejection sampling
        points = []
        while len(points) < N:
            r = np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi
            points.append([r * np.cos(theta), r * np.sin(theta)])
        return np.array(points)
    
    # Initialization 3: Spiral (Fermat)
    def init_spiral():
        # Sunflower pattern
        n = N
        c = 1.0
        points = []
        for i in range(n):
            r = np.sqrt(i / n)
            theta = i * 2.39996322972865332 # golden angle
            points.append([r * np.cos(theta), r * np.sin(theta)])
        return np.array(points)

    inits = [
        ("hex", init_hex()),
        ("random", init_random(1)),
        ("random2", init_random(2)),
        ("spiral", init_spiral())
    ]
    
    for name, x0 in inits:
        if time.time() > deadline:
            break
            
        # Ensure x0 is valid (inside circle)
        r0 = np.linalg.norm(x0, axis=1)
        if np.any(r0 > 1.0):
            x0 = x0 / np.maximum(r0[:, np.newaxis], 1e-6) * 0.99
        
        # Run optimization
        try:
            res = minimize(
                _repulsion_energy, 
                x0, 
                jac=_repulsion_gradient, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'gtol': 1e-5, 'disp': False}
            )
            x_opt = res.x
            
            # Ensure validity: project back to circle if needed
            r_opt = np.linalg.norm(x_opt, axis=1)
            mask_out = r_opt > 1.0
            if np.any(mask_out):
                x_opt[mask_out] = x_opt[mask_out] / r_opt[mask_out, np.newaxis] * 0.9999
                
            # Calculate score
            # Score = min( min_{i<j} d_ij / 2, min_i (1 - |c_i|) )
            
            # Pairwise distances
            diff = x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(dist_sq, 1e10)
            min_dist = np.sqrt(np.min(dist_sq))
            
            # Boundary distance
            r = np.linalg.norm(x_opt, axis=1)
            min_boundary_dist = 1.0 - np.max(r)
            
            score = min(min_dist / 2.0, min_boundary_dist)
            
            if score > best_score:
                best_score = score
                best_points = x_opt.copy()
                
        except Exception as e:
            pass

    if best_points is None:
        # Fallback to hex
        best_points = init_hex()
        
    return best_points
# EVOLVE-BLOCK-END

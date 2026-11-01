"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 300

# EVOLVE-BLOCK-START

def _hex_lattice_init(n):
    """Generate a hexagonal lattice of points inside the unit circle."""
    # Estimate spacing
    # Area of hexagon ~ sqrt(3)/2 * s^2. N circles ~ N * (s^2/2) roughly?
    # Better: just try a few spacings.
    # For N=300, r~0.05, d~0.1.
    # Lattice spacing s.
    # Number of points in radius 1 hex lattice with spacing s is approx pi/s^2 * (2/sqrt(3)) ?
    # Let's just iterate.
    spacing = 0.15
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= n:
            # Select n points closest to center to keep symmetry and compactness
            idx = np.argsort(radius, kind="stable")[:n]
            return points[idx]
        spacing *= 0.95
        if spacing < 0.01:
             # Fallback: random
             return np.random.rand(n, 2) * 2 - 1

def _compute_penalty_and_grad(x, r):
    """
    Compute penalty and gradient for packing N circles of radius r in unit circle.
    Penalty = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, ||x_i|| - (1-r))^2
    """
    n = len(x)
    penalty = 0.0
    grad = np.zeros_like(x)
    
    # Pairwise distances
    # x[:, np.newaxis, :] - x[np.newaxis, :, :]
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    
    # Avoid division by zero
    dist[dist == 0] = 1e-8
    
    # Overlap penalty
    # Only consider i < j to avoid double counting and self-interaction
    i_idx, j_idx = np.triu_indices(n, k=1)
    d = dist[i_idx, j_idx]
    v = 2 * r - d
    mask = v > 0
    if np.any(mask):
        v_masked = v[mask]
        d_masked = d[mask]
        diff_masked = diff[i_idx[mask], j_idx[mask], :]
        
        # Term: (2r - d)^2
        # dP/dd = -2(2r - d)
        # dd/dx_i = (x_i - x_j) / d
        # dP/dx_i = -2(2r - d) * (x_i - x_j) / d
        # We want to MINIMIZE penalty. Gradient points to increase penalty.
        # So we subtract gradient in optimizer, or return gradient of penalty.
        # scipy minimize expects gradient of the function to minimize.
        # So we return dP/dx.
        
        coeff = -2.0 * v_masked / d_masked
        # coeff shape (M,), diff_masked shape (M, 2)
        # grad_i += coeff * diff_masked
        # grad_j -= coeff * diff_masked
        
        np.add.at(grad, i_idx[mask], coeff[:, np.newaxis] * diff_masked)
        np.add.at(grad, j_idx[mask], -coeff[:, np.newaxis] * diff_masked)
        
        penalty += np.sum(v_masked ** 2)

    # Boundary penalty
    dist_center = np.linalg.norm(x, axis=1)
    limit = 1.0 - r
    v_bound = dist_center - limit
    mask_bound = v_bound > 0
    if np.any(mask_bound):
        v_b = v_bound[mask_bound]
        d_c = dist_center[mask_bound]
        x_b = x[mask_bound]
        
        # Term: (dist - limit)^2
        # dP/ddist = 2(dist - limit)
        # ddist/dx = x / dist
        # dP/dx = 2(dist - limit) * x / dist
        
        coeff_b = 2.0 * v_b / d_c
        grad[mask_bound] += coeff_b[:, np.newaxis] * x_b
        
        penalty += np.sum(v_b ** 2)

    return penalty, grad

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin
    
    # 1. Initialization
    x0 = _hex_lattice_init(N)
    
    # Estimate initial radius
    # min dist / 2
    diff = x0[:, np.newaxis, :] - x0[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    i_idx, j_idx = np.triu_indices(N, k=1)
    min_dist = np.min(dist[i_idx, j_idx])
    min_bound = np.min(1.0 - np.linalg.norm(x0, axis=1))
    r_init = min(min_dist / 2.0, min_bound)
    
    # Binary search bounds
    r_low = r_init * 0.9
    r_high = 0.1  # Upper bound guess. Max possible is ~ 1/sqrt(300) ~ 0.057. 0.1 is safe.
    best_x = x0
    best_r = r_init
    
    # Warm start for minimization
    current_x = x0
    
    # Number of binary search steps
    # 20 steps give precision ~ (0.1-0.05)/2^20 which is overkill.
    # 15 steps is enough.
    n_steps = 20
    
    for i in range(n_steps):
        if time.time() - start_time > timeout:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Objective function for scipy
        def obj(x_flat):
            x = x_flat.reshape(-1, 2)
            p, g = _compute_penalty_and_grad(x, r_mid)
            return p, g.reshape(-1)
            
        # Minimize
        try:
            res = minimize(
                obj, 
                current_x.reshape(-1), 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 100, 'gtol': 1e-5}
            )
        except Exception:
            # If minimization fails, assume infeasible
            r_high = r_mid
            continue
            
        p_final = res.fun
        # Threshold for feasibility
        # Penalty is sum of squares. If penalty is very small, it's feasible.
        # Scale: distances are ~0.1, r~0.05. Overlap 0.01 -> penalty 1e-4.
        # Let's use 1e-6 as threshold for "feasible enough"
        if p_final < 1e-6:
            r_low = r_mid
            best_r = r_mid
            best_x = res.x.reshape(-1, 2)
            current_x = res.x.reshape(-1, 2) # Warm start with solution
        else:
            r_high = r_mid
            # Do not update current_x to avoid contaminating the warm start with a bad solution?
            # Actually, keeping the best feasible x is safer.
            
    return best_x

# EVOLVE-BLOCK-END

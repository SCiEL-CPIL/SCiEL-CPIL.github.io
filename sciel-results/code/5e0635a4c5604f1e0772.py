"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # 1. Initialization: Hexagonal lattice with jitter
    def generate_hex_lattice(r):
        d = 2.0 * r
        if d <= 0:
            return np.array([])
        
        k = int(1.0 / d) + 2
        points = []
        sqrt3 = np.sqrt(3.0)
        
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * d
                y = j * d * sqrt3 / 2.0
                dist = np.hypot(x, y)
                if dist + r <= 1.0:
                    points.append([x, y])
        
        if len(points) == 0:
            return np.array([])
            
        points = np.array(points)
        
        if len(points) >= N:
            dists = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(dists, kind="stable")[:N]
            return points[indices]
        else:
            return points

    # Find a good initial radius for hex lattice
    r_low = 0.01
    r_high = 0.08
    
    while True:
        pts = generate_hex_lattice(r_low)
        if len(pts) >= N:
            break
        r_low *= 0.8
        
    for _ in range(20):
        r_mid = (r_low + r_high) / 2.0
        pts = generate_hex_lattice(r_mid)
        if len(pts) >= N:
            r_low = r_mid
        else:
            r_high = r_mid
            
    points = generate_hex_lattice(r_low)
    
    if len(points) < N:
        # Fallback random
        angles = np.random.rand(N) * 2 * np.pi
        radii = np.sqrt(np.random.rand(N)) * 0.5
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists, kind="stable")[:N]
        points = points[indices]
        
    # Add small jitter to break symmetry
    points += np.random.randn(N, 2) * 0.002
    
    # 2. Smooth Surrogate Optimization
    # Objective: Maximize soft-min of (pairwise distances / 2, boundary distances)
    # Soft-min: S(x) = -1/beta * log( sum exp(-beta * v_k) )
    # We minimize -S(x).
    
    def soft_min_objective(x_flat, beta):
        x = x_flat.reshape(-1, 2)
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Values for soft min: d_pairs / 2 and 1 - |x|
        vals_pairs = dists / 2.0
        radii = np.hypot(x[:, 0], x[:, 1])
        vals_bound = 1.0 - radii
        
        # To avoid overflow in exp, we can shift values, but since we want min,
        # large negative values (violations) should have large exp weight.
        # exp(-beta * val). If val is negative, -beta*val is positive.
        
        # We only sum over i < j for pairs to avoid double counting in the log-sum-exp?
        # Actually, for min, double counting doesn't change the argmin, but it affects the gradient magnitude.
        # Standard soft-min uses all terms. Let's use all pairs i!=j.
        
        with np.errstate(over='ignore', invalid='ignore'):
            exp_pairs = np.exp(-beta * vals_pairs)
            exp_pairs[np.diag_indices(N)] = 0.0
            
            exp_bound = np.exp(-beta * vals_bound)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bound)
        
        if not np.isfinite(sum_exp) or sum_exp == 0:
            return 1e10, np.zeros_like(x_flat)
            
        S = -1.0 / beta * np.log(sum_exp)
        val = -S
        
        # Gradient
        # grad_S = 1/Z * sum w_k grad_v_k
        # grad_val = -grad_S
        
        # Pairs
        # grad_v_pairs[i, j] = 0.5 * (x[i] - x[j]) / d[i, j]
        with np.errstate(divide='ignore', invalid='ignore'):
            grad_v_pairs = 0.5 * diff / dists[:, :, np.newaxis]
            grad_v_pairs[:, :, :] = np.where(dists[:, :, np.newaxis] > 1e-8, grad_v_pairs, 0.0)
            
            # Weighted sum
            # w_matrix = exp_pairs
            grad_x_pairs = np.sum(exp_pairs[:, :, np.newaxis] * grad_v_pairs, axis=1)
            
        # Boundary
        # grad_v_bound[i] = -x[i] / |x[i]|
        with np.errstate(divide='ignore', invalid='ignore'):
            grad_v_bound = -x / radii[:, np.newaxis]
            grad_v_bound = np.where(radii[:, np.newaxis] > 1e-8, grad_v_bound, 0.0)
            
        grad_x_bound = exp_bound[:, np.newaxis] * grad_v_bound
        
        grad_S = grad_x_pairs + grad_x_bound
        
        Z = sum_exp
        if Z > 0:
            grad_S /= Z
            
        grad_val = -grad_S
        
        return val, grad_val.ravel()

    # Optimization Loop
    current_x = points.copy().ravel()
    
    # Start with low beta (smooth) and increase
    betas = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(
                soft_min_objective, 
                current_x, 
                args=(beta,), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < 1e9: # Valid result
                current_x = res.x
        except Exception:
            pass
            
    best_points = current_x.reshape(-1, 2)
    
    # 3. Final Polish: Local search to refine the exact minimum radius
    # We do a few steps of force-based relaxation on the best points to fix any minor overlaps
    # caused by the soft-min approximation not being exact at finite beta.
    
    def compute_radius(pts):
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_pair, min_boundary)

    best_r = compute_radius(best_points)
    best_pts = best_points.copy()
    
    # Simple force polish
    step_size = 0.001
    for _ in range(500):
        if time.time() > deadline - 2.0:
            break
            
        radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
        diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        current_r = min(np.min(dists) / 2.0, np.min(1.0 - radii))
        
        if current_r > best_r:
            best_r = current_r
            best_pts = best_pts.copy()
            
        forces = np.zeros_like(best_pts)
        
        # Repulsion for close pairs
        # Target radius slightly above current
        r_target = current_r * 1.001
        
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 2.0 * r_target:
                    diff_ij = best_pts[i] - best_pts[j]
                    if d > 1e-8:
                        force_mag = (2.0 * r_target - d) / d
                        force_vec = diff_ij * force_mag
                        forces[i] += force_vec
                        forces[j] -= force_vec
                        
            # Boundary
            if radii[i] > 1.0 - r_target:
                violation = radii[i] - (1.0 - r_target)
                if violation > 0:
                    forces[i] -= (best_pts[i] / radii[i]) * violation * 10.0
                    
        best_pts += forces * step_size
        
        # Project inside
        new_radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
        outside = new_radii > 1.0
        if np.any(outside):
            best_pts[outside] = best_pts[outside] / new_radii[outside, np.newaxis]
            
    return best_pts
# EVOLVE-BLOCK-END

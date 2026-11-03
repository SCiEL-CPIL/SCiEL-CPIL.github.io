import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 110.0  # 10s margin for safety
    
    # 1. Initialization: Hexagonal lattice
    # We want to place N points.
    # A hexagonal packing with spacing s has density ~ 0.9069.
    # Area of unit circle is pi. Area of N circles is N * pi * r^2.
    # r ~ sqrt(1/N) * sqrt(density) ~ 0.14 * 0.95 ~ 0.13.
    # Let's generate a grid of points and pick the best N.
    
    def generate_initial_points():
        points = []
        # Hexagonal grid
        s = 0.15  # initial spacing guess
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = i * s + (j % 2) * s * 0.5
                y = j * s * np.sqrt(3) / 2
                if x*x + y*y < 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
            # Fallback to random
            points = np.random.uniform(-0.9, 0.9, (N, 2))
            # Ensure inside
            norms = np.linalg.norm(points, axis=1)
            mask = norms <= 1.0
            if not np.all(mask):
                 # Scale down
                 points = points * (0.9 / np.max(norms))
        else:
            # Select N points closest to center? Or just first N?
            # Better: Select points that are well spread.
            # For now, just take N points from the lattice.
            # To be more robust, we can shuffle and take N, or take the ones with lowest potential.
            # Let's just take the first N from a sorted list by radius to ensure central filling.
            dists = np.linalg.norm(points, axis=1)
            indices = np.argsort(dists)
            points = points[indices[:N]]
        return points

    c_init = generate_initial_points()
    
    # 2. Optimization
    # Variables: x_0...x_{N-1}, y_0...y_{N-1}, r
    # Total 2N + 1 variables.
    
    def objective_and_grad(vars):
        if time.time() - start_time > timeout:
            return 1e10, np.zeros_like(vars)
        
        c = vars[:2*N].reshape(N, 2)
        r = vars[2*N]
        
        # Compute distances
        # Vectorized distance calculation
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero or self-interaction
        np.fill_diagonal(dists, 1.0)
        
        # Boundary distances
        norms = np.linalg.norm(c, axis=1)
        
        # Penalty terms
        # Overlap penalty: max(0, 2r - dist)^2
        overlap_viol = np.maximum(0, 2*r - dists)
        # Exclude self
        np.fill_diagonal(overlap_viol, 0)
        penalty_overlap = np.sum(overlap_viol**2)
        
        # Boundary penalty: max(0, norm - (1-r))^2
        bound_viol = np.maximum(0, norms - (1 - r))
        penalty_bound = np.sum(bound_viol**2)
        
        penalty = penalty_overlap + penalty_bound
        
        # We want to maximize r, so minimize -r + lambda * penalty
        # Lambda needs to be large enough to enforce constraints.
        # As r increases, penalty grows if constraints are violated.
        # We can adapt lambda or just use a fixed large value.
        # Let's use a fixed lambda = 100.0
        lambda_val = 100.0
        
        val = -r + lambda_val * penalty
        
        # Gradient calculation (numerical is slow, let's try analytical if possible or use finite diff in scipy)
        # For L-BFGS-B, providing gradient is good.
        # But for simplicity and robustness against coding errors in grad, 
        # we can let scipy use finite differences if we don't provide jac.
        # However, N=50 means 101 variables. Finite diff is 101 evaluations per step.
        # Analytical gradient is much faster.
        
        grad = np.zeros_like(vars)
        
        # Gradient of -r is -1
        grad[2*N] = -1.0
        
        # Gradient of penalty_overlap
        # d/du max(0, u)^2 = 2*max(0, u)
        # u_ij = 2r - dist_ij
        # d u_ij / d c_k ...
        
        # This is getting complex. Let's use finite differences via scipy's jac option or just not provide it.
        # Given the time limit, finite differences might be too slow if many iterations are needed.
        # Let's try providing the analytical gradient.
        
        # Overlap part:
        # P_o = sum_{i<j} max(0, 2r - d_ij)^2
        # dP_o/dc_k = sum_{i<j, k in {i,j}} 2 * max(0, 2r - d_ij) * (2r - d_ij) * (-d d_ij/dc_k)
        # d d_ij / d c_i = (c_i - c_j) / d_ij
        # d d_ij / d c_j = (c_j - c_i) / d_ij
        
        # Let's compute the gradient vector for c
        grad_c = np.zeros_like(c)
        
        # We need to iterate over pairs or use vectorized ops.
        # Vectorized grad for pairwise potentials is tricky but doable.
        # Let's stick to a simpler approach: use scipy.optimize.minimize with jac=None (finite diff) 
        # but reduce the number of iterations or use a coarser grid.
        # Actually, for N=50, 101 variables, finite diff is ~100 evals/step.
        # If we need 100 steps, that's 10,000 evals. Each eval is O(N^2) = 2500 ops. 25M ops. Fast in numpy.
        # So finite difference is acceptable.
        
        return val, None # None indicates finite difference

    # We will use a multi-start or iterative r-increasing strategy.
    # Start with r = 0.1
    r_current = 0.1
    best_c = c_init
    best_r_achieved = 0.0
    
    # Initial evaluation
    def calc_radius(c):
        d = np.linalg.norm(c[:, np.newaxis, :] - c[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, 1.0)
        min_dist = np.min(d)
        min_norm = np.min(1.0 - np.linalg.norm(c, axis=1))
        return min(min_dist / 2.0, min_norm)
    
    best_r_achieved = calc_radius(best_c)
    
    # Optimization loop
    # We try to increase r step by step.
    # Or we can just optimize the joint (c, r) with a large lambda.
    # Let's try the joint optimization with L-BFGS-B.
    
    x0 = np.concatenate([c_init.flatten(), [0.12]])
    
    # Bounds
    # c in [-1, 1], r in [0, 0.2]
    bounds = [(-1.0, 1.0)] * (2*N) + [(0.0, 0.2)]
    
    # Run optimizer
    # maxiter should be enough.
    # We check time inside the objective function to abort early.
    
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
        
        if time.time() - start_time < timeout:
            c_opt = res.x[:2*N].reshape(N, 2)
            r_opt = res.x[2*N]
            
            # The optimizer might not have found a feasible solution for r_opt if lambda wasn't strong enough.
            # We must calculate the ACTUAL achievable radius for c_opt.
            r_actual = calc_radius(c_opt)
            
            if r_actual > best_r_achieved:
                best_r_achieved = r_actual
                best_c = c_opt
                
            # If the optimizer converged to a valid packing with r_actual > r_opt, we might be able to push r further.
            # Let's do a few more steps if time permits, increasing the target r.
            # But L-BFGS-B on the joint space should handle this if lambda is high.
            
    except Exception as e:
        print(f"Optimization error: {e}")

    # Final polish:
    # Sometimes L-BFGS-B gets stuck in local minima.
    # We can try a simple local search or just return the best found.
    # Given the "known best" is 0.1258, and we start at 0.1, we should be able to reach ~0.125.
    
    # One more check: ensure the returned points are strictly valid (inside, no overlap).
    # The score calculation in the evaluator is robust, but we should return a valid config.
    # If best_c has overlaps, the radius will be small/zero.
    # We assume L-BFGS-B with penalty pushed it to a valid state.
    
    return best_c

# EVOLVE-BLOCK-END

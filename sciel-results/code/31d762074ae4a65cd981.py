import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin before 120s kill
    
    # Potential function and gradient
    def potential_and_grad(x):
        x = x.reshape(N, 3)
        # Pairwise distances
        # d[i,j] = ||x[i] - x[j]||
        # To save memory and time, we can iterate, but for N=50, 50x50 matrix is small.
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid self-interaction (diagonal)
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        # dist to -0.5 is 0.5 + x
        # dist to 0.5 is 0.5 - x
        d_left = 0.5 + x
        d_right = 0.5 - x
        # Nearest wall distance for each coordinate
        d_wall = np.minimum(d_left, d_right)
        
        eps = 1e-6
        
        # Potential
        # Sum of 1/(d+eps)^2 for pairs
        pair_pot = np.sum(1.0 / (dists + eps)**2)
        # Sum of 1/(d_wall+eps)^2 for walls
        wall_pot = np.sum(1.0 / (d_wall + eps)**2)
        
        total_pot = pair_pot + wall_pot
        
        # Gradient
        # grad_pair = sum_j -2 * (1/(dists[i,j]+eps)^3) * (x[i] - x[j]) / dists[i,j]
        # Note: dists[i,i] is inf, so term is 0.
        # We need to be careful with dists=0 for i!=j.
        # Using np.where to handle division by zero if any (shouldn't happen with eps, but safe).
        
        # Factor for pair potential: -2 * (dists+eps)^-3 / dists
        # If dists is 0, this is singular. But we have eps in denominator.
        # Actually, derivative of (d+eps)^-2 is -2(d+eps)^-3 * dd/dx.
        # dd/dx = (x_i - x_j) / d.
        # So term is -2 (d+eps)^-3 (x_i - x_j) / d.
        
        # Let's compute the matrix of coefficients
        # coeff[i,j] = -2 / ((dists[i,j] + eps)**3 * dists[i,j])
        # If dists[i,j] == 0, set coeff to 0? 
        # If two points coincide, d=0. The force is undefined. 
        # In practice, with random init and eps, d is rarely exactly 0.
        # If d is very small, force is huge, which is good for repulsion.
        
        with np.errstate(divide='ignore', invalid='ignore'):
            coeff = -2.0 / ((dists + eps)**3 * dists)
        
        # Set diagonal to 0
        np.fill_diagonal(coeff, 0.0)
        # Replace any inf/nan with 0
        coeff = np.nan_to_num(coeff, nan=0.0, posinf=0.0, neginf=0.0)
        
        # grad_pair[i] = sum_j coeff[i,j] * (x[i] - x[j])
        # diff[i,j] = x[i] - x[j]
        grad_pair = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        # Wall gradient
        # d_wall = min(0.5+x, 0.5-x)
        # If 0.5+x < 0.5-x (i.e., x < 0), d_wall = 0.5+x, d(d_wall)/dx = 1
        # If 0.5-x < 0.5+x (i.e., x > 0), d_wall = 0.5-x, d(d_wall)/dx = -1
        # If x == 0, derivative is undefined, but measure zero.
        
        d_wall_deriv = np.where(x < 0, 1.0, -1.0)
        
        # grad_wall[i,k] = -2 * (d_wall[i,k] + eps)^-3 * d_wall_deriv[i,k]
        grad_wall = -2.0 * (1.0 / (d_wall + eps)**3) * d_wall_deriv
        
        total_grad = grad_pair + grad_wall
        
        return total_pot, total_grad.flatten()

    best_radius = 0.0
    best_x = None
    
    # We will run multiple restarts
    # Estimate time per restart. L-BFGS-B on 150 vars is fast.
    # Let's try to run as many as possible.
    
    num_restarts = 0
    max_restarts = 500 # Hard cap to prevent infinite loop if time check fails
    
    while time.time() - start_time < time_limit and num_restarts < max_restarts:
        # Random initialization
        x0 = np.random.uniform(-0.4, 0.4, (N, 3)).flatten()
        
        try:
            res = minimize(
                potential_and_grad, 
                x0, 
                jac=True, 
                method='L-BFGS-B', 
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            
            if res.success:
                x_opt = res.x.reshape(N, 3)
                
                # Calculate the actual radius (min distance / 2 and min wall dist)
                diff = x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dists, np.inf)
                min_pair_dist = np.min(dists)
                
                d_left = 0.5 + x_opt
                d_right = 0.5 - x_opt
                d_wall = np.minimum(d_left, d_right)
                min_wall_dist = np.min(d_wall)
                
                r = min(min_pair_dist / 2.0, min_wall_dist)
                
                if r > best_radius:
                    best_radius = r
                    best_x = x_opt.copy()
                    
        except Exception:
            pass
        
        num_restarts += 1
        
        # Check time more frequently in later stages
        if num_restarts > 10 and (time.time() - start_time) > (time_limit - 5.0):
            break

    if best_x is None:
        # Fallback to simple grid if something went wrong
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
        
    return best_x
# EVOLVE-BLOCK-END

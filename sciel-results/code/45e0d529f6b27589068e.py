import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_score = np.inf
    best_points = None
    
    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    n_pairs = len(i_idx)
    
    def get_pairs(X):
        # X is (N, D)
        # Returns array of pairwise squared distances (n_pairs,)
        # and gradients for each point
        xi = X[i_idx]
        xj = X[j_idx]
        diff = xi - xj
        u = np.sum(diff**2, axis=1)
        return u, diff
        
    def objective_and_grad(X_flat, beta, mu):
        X = X_flat.reshape(N, D)
        u, diff = get_pairs(X)
        
        # Log-sum-exp for max
        # U_beta = 1/beta * log(sum(exp(beta * u)))
        # To avoid overflow, subtract max
        u_max = np.max(u)
        exp_terms = np.exp(beta * (u - u_max))
        sum_exp = np.sum(exp_terms)
        U_beta = u_max + (1.0 / beta) * np.log(sum_exp)
        
        # Weights for gradient of U_beta
        weights_u = exp_terms / sum_exp
        
        # Gradient of U_beta w.r.t X
        # dU_beta/dX_i = sum_{j} weights_{ij} * 2 * (x_i - x_j)
        grad_U = np.zeros_like(X)
        # Accumulate gradients for i_idx
        # diff = xi - xj
        # grad for xi is 2 * diff * weight
        # grad for xj is -2 * diff * weight
        np.add.at(grad_U, i_idx, 2.0 * weights_u[:, None] * diff)
        np.add.at(grad_U, j_idx, -2.0 * weights_u[:, None] * diff)
        
        # Penalty P(X) = sum max(0, 1 - u)^2
        # If u < 1, penalty is (1-u)^2
        # Grad is 2(1-u)(-1) * grad_u = -2(1-u) * grad_u
        # grad_u for xi is 2(xi-xj), for xj is -2(xi-xj)
        # So grad P for xi is -2(1-u) * 2(xi-xj) = -4(1-u)(xi-xj)
        # grad P for xj is -2(1-u) * (-2(xi-xj)) = 4(1-u)(xi-xj)
        
        mask = u < 1.0
        if np.any(mask):
            u_masked = u[mask]
            diff_masked = diff[mask]
            # Coefficient is -4 * (1 - u)
            coeff = -4.0 * (1.0 - u_masked)
            
            # Add to grad
            # grad_P = mu * grad_P_raw
            # We want total grad = grad_U + mu * grad_P
            # So we add mu * grad_P to grad_U
            
            # For i_idx
            np.add.at(grad_U, i_idx[mask], coeff[:, None] * diff_masked)
            # For j_idx
            np.add.at(grad_U, j_idx[mask], -coeff[:, None] * diff_masked)
            
            P_val = np.sum((1.0 - u_masked)**2)
        else:
            P_val = 0.0
            
        val = U_beta + mu * P_val
        grad = grad_U + mu * 0.0 # Wait, I already added mu*grad_P to grad_U? 
        # No, I added coeff which is -4(1-u).
        # The term in objective is mu * (1-u)^2.
        # Grad is mu * 2(1-u)(-1) * grad_u.
        # My coeff was -4(1-u).
        # grad_u for xi is 2 diff.
        # So term is mu * (-4(1-u)) * diff.
        # Yes, I added coeff * diff to grad_U.
        # So grad_U now contains grad_U_beta + mu * grad_P.
        # So val is U_beta + mu * P_val.
        # And grad is grad_U.
        
        return val, grad

    # Optimization loop
    num_restarts = 20
    restarts_done = 0
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize points
        # Random points in a square
        X0 = np.random.uniform(-2, 2, (N, D))
        
        # Scale to have reasonable initial distances
        # Maybe scale to min dist ~ 1?
        # Let's just start with random and let the optimizer handle it.
        # But if points are too close, penalty is huge.
        # If points are too far, U_beta is huge.
        
        # Better initialization: Grid + noise
        side = int(np.ceil(N ** (1 / D)))
        grid = np.array(list(__import__('itertools').product(range(side), repeat=D)))[:N]
        X0 = grid + np.random.normal(0, 0.1, (N, D))
        
        # Optimization parameters
        beta = 5.0
        mu = 10.0
        max_iter = 500
        
        res = minimize(
            objective_and_grad,
            X0.flatten(),
            args=(beta, mu),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'maxfun': 1000}
        )
        
        if res.success:
            X_opt = res.x.reshape(N, D)
            
            # Calculate actual score
            u, _ = get_pairs(X_opt)
            d_min = np.sqrt(np.min(u))
            d_max = np.sqrt(np.max(u))
            if d_min > 1e-6:
                score = (d_max / d_min) ** 2
            else:
                score = np.inf
                
            if score < best_score:
                best_score = score
                best_points = X_opt
        
        # Try a second phase with higher beta and mu if time permits
        if time.time() > deadline:
            break
            
        if res.success and best_points is not X_opt: # Only if we found a good one? Or just always try to refine
             # Let's just refine the best found so far if we have time
             pass

    # If we still have time, do a refinement pass on the best points
    if best_points is not None and time.time() < deadline - 10:
        X_refine = best_points
        beta = 50.0
        mu = 100.0
        res = minimize(
            objective_and_grad,
            X_refine.flatten(),
            args=(beta, mu),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'maxfun': 500}
        )
        if res.success:
            X_opt = res.x.reshape(N, D)
            u, _ = get_pairs(X_opt)
            d_min = np.sqrt(np.min(u))
            d_max = np.sqrt(np.max(u))
            if d_min > 1e-6:
                score = (d_max / d_min) ** 2
                if score < best_score:
                    best_score = score
                    best_points = X_opt

    if best_points is None:
        # Fallback to grid
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_points = np.array(grid, dtype=float)
        
    return best_points

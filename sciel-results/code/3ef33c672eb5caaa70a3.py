"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    
    def compute_overlaps(h):
        """Compute overlap at each shift k."""
        overlaps = np.zeros(N)
        for k in range(N):
            # h[i] * (1 - h[i+k]) for valid i
            n_terms = N - k
            if n_terms <= 0:
                overlaps[k] = 0.0
            else:
                overlaps[k] = np.sum(h[:n_terms] * (1 - h[k:]))
        return overlaps
    
    def objective_and_grad(h):
        """Objective: smooth max of overlaps using log-sum-exp."""
        overlaps = compute_overlaps(h)
        # Use log-sum-exp for smooth max approximation
        beta = 50.0
        max_overlap = np.max(overlaps)
        if max_overlap < 1e-10:
            return 0.0, np.zeros_like(h)
        # Soft max
        shifted = beta * overlaps
        max_shifted = np.max(shifted)
        log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        obj = log_sum_exp / beta
        
        # Gradient via finite differences or analytic
        # For simplicity, use numerical gradient
        grad = np.zeros_like(h)
        eps = 1e-7
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            overlaps_plus = compute_overlaps(h_plus)
            shifted_plus = beta * overlaps_plus
            max_shifted_plus = np.max(shifted_plus)
            log_sum_exp_plus = max_shifted_plus + np.log(np.sum(np.exp(shifted_plus - max_shifted_plus)))
            obj_plus = log_sum_exp_plus / beta
            grad[i] = (obj_plus - obj) / eps
        return obj, grad
    
    # Initial guess: all 0.5
    h0 = np.full(N, 0.5)
    
    # Use L-BFGS-B with bounds [0, 1]
    best_h = h0.copy()
    best_obj = np.max(compute_overlaps(h0))
    
    # Try multiple random restarts if time permits
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
            
        if restart == 0:
            h_init = h0.copy()
        else:
            # Perturb from best or random
            h_init = best_h + 0.1 * np.random.randn(N)
            h_init = np.clip(h_init, 0.01, 0.99)
            # Project to satisfy sum constraint approximately
            h_init = h_init / np.sum(h_init) * (N / 2)
            h_init = np.clip(h_init, 0.0, 1.0)
        
        result = minimize(
            lambda x: objective_and_grad(x)[0],
            h_init,
            jac=lambda x: objective_and_grad(x)[1],
            method='L-BFGS-B',
            bounds=[(0, 1)] * N,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        h_opt = result.x
        # Enforce sum constraint by scaling
        current_sum = np.sum(h_opt)
        if current_sum > 0:
            h_opt = h_opt * (N / 2) / current_sum
            h_opt = np.clip(h_opt, 0.0, 1.0)
        
        overlaps = compute_overlaps(h_opt)
        max_overlap = np.max(overlaps)
        
        if max_overlap < best_obj:
            best_obj = max_overlap
            best_h = h_opt.copy()
    
    # Final polish: ensure exact sum
    current_sum = np.sum(best_h)
    if current_sum > 0:
        best_h = best_h * (N / 2) / current_sum
        best_h = np.clip(best_h, 0.0, 1.0)
    
    # Verify sum is exactly N/2
    diff = N / 2 - np.sum(best_h)
    if abs(diff) > 1e-10:
        # Distribute difference evenly
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END

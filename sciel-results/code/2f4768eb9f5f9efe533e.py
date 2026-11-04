import numpy as np
from scipy.optimize import minimize
import sys
import time

def construct():
    N = 600
    L = 0.5  # Length of interval [-0.25, 0.25]
    M = 20   # Number of harmonics (reduced for stability)
    
    # Grids for optimization
    n_x = 400
    x_grid = np.linspace(-L/2, L/2, n_x, endpoint=False)
    
    n_t = 800
    t_grid = np.linspace(-L/2, L/2, n_t, endpoint=False)
    
    # Precompute basis matrices
    A = np.zeros((n_x, M + 1))
    A[:, 0] = np.sqrt(2.0)
    for k in range(1, M + 1):
        A[:, k] = 2.0 * np.cos(2.0 * np.pi * k * x_grid / L)
        
    B = np.zeros((n_t, M + 1))
    B[:, 0] = 1.0
    for k in range(1, M + 1):
        B[:, k] = np.cos(2.0 * np.pi * k * t_grid / L)
    
    # Smooth max approximation using log-sum-exp
    def smooth_max(g_vals, temperature=0.01):
        """Smooth approximation of max using log-sum-exp"""
        max_val = np.max(g_vals)
        if max_val < -1e10:
            return max_val, np.zeros_like(g_vals)
        shifted = g_vals - max_val
        log_sum = np.log(np.sum(np.exp(shifted / temperature)))
        result = max_val + temperature * log_sum
        # Gradient: softmax weights
        weights = np.exp(shifted / temperature) / np.sum(np.exp(shifted / temperature))
        return result, weights
    
    def objective_and_gradient(u, temperature=0.01):
        u_sq = u**2
        g_vals = 2.0 * (B @ u_sq)
        
        obj_val, weights = smooth_max(g_vals, temperature)
        
        # Gradient: sum of weighted gradients
        # d/du_k [2 * sum_j w_j * sum_l u_l^2 B[j,l]] = 4 * sum_j w_j * B[j,k] * u_k
        grad = 4.0 * u * (weights @ B)
        
        return obj_val, grad
    
    def constraint_energy(u):
        return np.dot(u, u) - 1.0
        
    def constraint_energy_grad(u):
        return 2.0 * u

    def constraint_nonneg(u):
        f_vals = A @ u
        return f_vals

    # Initial guess: Uniform function
    u0 = np.zeros(M + 1)
    u0[0] = 1.0
    
    # Multiple restarts with diverse initializations
    best_score = np.inf
    best_u = u0.copy()
    
    n_restarts = 8
    start_time = time.time()
    
    for r in range(n_restarts):
        if time.time() - start_time > 90:
            break
            
        if r == 0:
            u_start = u0.copy()
        elif r == 1:
            # Slightly perturbed uniform
            u_start = u0.copy()
            u_start[1] = 0.1
            u_start[2] = -0.05
            u_start = u_start / np.linalg.norm(u_start)
        elif r == 2:
            # Symmetric bump: u_k ~ cos(k * phi)
            phi = 0.3
            u_start = np.zeros(M + 1)
            for k in range(M + 1):
                u_start[k] = np.cos(k * phi)
            u_start = u_start / np.linalg.norm(u_start)
        else:
            # Random start with low-frequency bias
            u_rand = np.random.randn(M + 1)
            # Bias toward low frequencies
            for k in range(1, M + 1):
                u_rand[k] *= np.exp(-k * 0.1)
            u_start = u_rand / np.linalg.norm(u_start)
            
        # Define constraints
        cons = [
            {'type': 'eq', 'fun': constraint_energy, 'jac': constraint_energy_grad},
            {'type': 'ineq', 'fun': constraint_nonneg, 'jac': lambda u, jac: A}
        ]
        
        try:
            # Use SLSQP with smooth objective
            res = minimize(
                lambda u: objective_and_gradient(u)[0],
                u_start,
                method='SLSQP',
                jac=lambda u: objective_and_gradient(u)[1],
                constraints=cons,
                options={'maxiter': 150, 'ftol': 1e-8}
            )
            
            u_opt = res.x
            # Normalize
            energy = np.dot(u_opt, u_opt)
            if energy > 1e-10:
                u_opt = u_opt / np.sqrt(energy)
            
            # Check non-negativity with tolerance
            f_vals = A @ u_opt
            min_f = np.min(f_vals)
            if min_f >= -1e-4:
                # Calculate actual score (hard max)
                u_sq = u_opt**2
                g_vals = 2.0 * (B @ u_sq)
                score = np.max(g_vals)
                
                if score < best_score:
                    best_score = score
                    best_u = u_opt.copy()
                    
        except Exception as e:
            pass
    
    # If optimization failed or didn't improve, use best_u
    interval_width = L / N
    centers = -L/2 + (np.arange(N) + 0.5) * interval_width
    
    # Evaluate f at centers
    f_centers = np.zeros(N)
    f_centers += best_u[0] * np.sqrt(2.0)
    for k in range(1, M + 1):
        f_centers += best_u[k] * 2.0 * np.cos(2.0 * np.pi * k * centers / L)
        
    # Ensure non-negative
    f_centers = np.maximum(0, f_centers)
    
    return list(f_centers)

if __name__ == "__main__":
    heights = construct()
    assert len(heights) == 600
    assert all(h >= 0 for h in heights)

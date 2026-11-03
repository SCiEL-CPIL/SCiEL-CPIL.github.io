"""
Optimized program for the First Autocorrelation Inequality task.

Approach:
1. Represent the step function by its first M Fourier cosine coefficients (symmetric, even function).
2. The autoconvolution of a function f is related to the square of its Fourier transform.
3. For a periodic function on [-1/4, 1/4] with period 1/2, the autoconvolution's peak can be bounded
   by the ratio of the L2 norm of the Fourier transform to the DC component squared, but more directly,
   we minimize the maximum of the autoconvolution relative to the energy.
   
   Actually, the autoconvolution (f*f)(t) = ∫ f(τ)f(t-τ) dτ. For a function supported on [-1/4, 1/4],
   the autoconvolution is supported on [-1/2, 1/2].
   
   The task asks to minimize the "normalized maximum of its autoconvolution". 
   Let's assume the normalization is by the L2 norm of f, or by the area.
   Typically, C1 is related to the constant in the inequality ||f*f||_∞ ≤ C1 ||f||_2^2.
   So we want to minimize ||f*f||_∞ / ||f||_2^2.
   
   Using Parseval's theorem: ||f*f||_2^2 = (2π)^2 ||F||_2^2 ... wait, that's L2 norm of the convolution.
   We need the L∞ norm.
   
   However, if we restrict to band-limited functions (low frequency harmonics), the autoconvolution will also be band-limited.
   We can compute the autoconvolution in the time domain by summing the Fourier series of the product of coefficients.
   
   Let f(x) = sum_{k=-M}^{M} c_k e^{i 2π k x / L} where L = 1/2.
   Since f is real and even, c_{-k} = c_k and c_k is real. Let c_k for k>=0.
   f(x) = c_0 + 2 * sum_{k=1}^{M} c_k cos(2π k x / L).
   
   The autoconvolution g(t) = (f*f)(t).
   In the Fourier domain, G(ω) = F(ω)^2.
   Since f is band-limited to M harmonics, g is band-limited to 2M harmonics.
   We can compute g(t) at many points in time to estimate its maximum.
   
   Objective: minimize max_t g(t) / (∫ f^2 dx).
   By scaling invariance, we can fix ∫ f^2 dx = 1 and minimize max_t g(t).
   
   We will optimize the coefficients c_0, c_1, ..., c_M subject to:
   1. f(x) >= 0 for all x in [-L/2, L/2] (nonnegative step function constraint).
   2. ||f||_2^2 = 1.
   
   We use gradient descent with projection or penalty methods.
"""

import numpy as np
from scipy.optimize import minimize, LinearConstraint
import sys

N = 600

def construct():
    # Parameters
    M = 15  # Number of harmonics
    L = 0.5  # Length of the interval [-1/4, 1/4]
    
    # Discretization for checking constraints and computing objective
    # Use a fine grid for evaluating f(x) >= 0 and max of g(t)
    n_grid = 2048
    x = np.linspace(-L/2, L/2, n_grid, endpoint=False)
    dx = L / n_grid
    
    # Time grid for autoconvolution g(t)
    # g(t) is supported on [-L, L] = [-0.5, 0.5]
    n_t_grid = 4096
    t = np.linspace(-L, L, n_t_grid, endpoint=False)
    dt = 2*L / n_t_grid
    
    # Precompute basis matrices for efficiency
    # Phi_f: shape (n_grid, M+1), where Phi_f[j, k] = phi_k(x_j)
    # phi_0(x) = 1/sqrt(L)
    # phi_k(x) = sqrt(2/L) * cos(2π k x / L) for k >= 1
    Phi_f = np.zeros((n_grid, M+1))
    Phi_f[:, 0] = 1.0 / np.sqrt(L)
    for k in range(1, M+1):
        Phi_f[:, k] = np.sqrt(2.0/L) * np.cos(2*np.pi*k*x/L)
    
    # For g(t), we need to evaluate the cosine terms
    # g(t) = (1/L) * [ u_0^2 + sum_{k=1}^M u_k^2 cos(2π k t / L) ]
    # Let's precompute the cosine matrix for g
    # CosG: shape (n_t_grid, M+1), where CosG[j, k] = cos(2π k t_j / L) for k>=1, 1 for k=0
    CosG = np.zeros((n_t_grid, M+1))
    CosG[:, 0] = 1.0
    for k in range(1, M+1):
        CosG[:, k] = np.cos(2*np.pi*k*t/L)
    
    def compute_f(x_grid, u):
        """Compute f(x) for given grid and coefficients u using matrix multiplication."""
        # x_grid: (n,)
        # u: (M+1,)
        # f_vals = u_0/sqrt(L) + sum u_k sqrt(2/L) cos(...)
        # We can reuse the structure. Since x_grid might be different from x, we compute directly.
        # But for the optimization loop, we use the precomputed Phi_f.
        # This function is for final evaluation or if x changes.
        f_vals = np.zeros_like(x_grid)
        f_vals += u[0] / np.sqrt(L)
        for k in range(1, M+1):
            f_vals += u[k] * np.sqrt(2.0/L) * np.cos(2*np.pi*k*x_grid/L)
        return f_vals
    
    def compute_g(t_grid, u):
        """Compute g(t) for given grid and coefficients u."""
        # g(t) = (1/L) * [ u_0^2 + sum_{k=1}^M u_k^2 cos(2π k t / L) ]
        # We can vectorize this.
        # u_sq = u**2
        # g_vals = (1/L) * (u_sq[0] + sum_{k=1}^M u_sq[k] * CosG[:, k])
        # Let's use matrix multiplication.
        # Create a matrix where row j is [1, cos(...), cos(...), ...]
        # Then g_vals[j] = (1/L) * (u_sq[0] * 1 + sum_{k=1}^M u_sq[k] * CosG[j, k])
        # This is (1/L) * (u_sq[0] + CosG[:, 1:] @ u_sq[1:])
        # But CosG[:, 0] is 1.
        # So g_vals = (1/L) * (CosG @ u_sq)
        u_sq = u**2
        g_vals = (CosG @ u_sq) / L
        return g_vals
    
    # Initial guess: uniform function (u_0 = 1, others 0)
    u0 = np.zeros(M+1)
    u0[0] = 1.0
    
    best_score = np.inf
    best_u = u0.copy()
    
    # Optimization parameters
    n_restarts = 5
    n_iter = 200
    lr = 0.01
    penalty_weight = 100.0
    
    for restart in range(n_restarts):
        if restart == 0:
            u = u0.copy()
        else:
            u = np.random.randn(M+1)
            u /= np.linalg.norm(u)
            # Ensure initial feasibility or at least not too bad
            # If f is negative, the penalty will handle it.
        
        for i in range(n_iter):
            # Compute f and g using precomputed matrices
            f_vals = Phi_f @ u
            g_vals = compute_g(t, u)
            
            # Penalty for negative f
            neg_mask = f_vals < 0
            penalty = np.sum(f_vals[neg_mask]**2) if np.any(neg_mask) else 0.0
            
            # Objective
            idx_max = np.argmax(g_vals)
            obj = g_vals[idx_max] + penalty_weight * penalty
            
            # Gradient of max(g)
            # g(t) = (1/L) * sum_{k=0}^M u_k^2 * C_k(t) where C_0=1, C_k=cos(2πkt/L)
            # dg/dt_k = (2/L) * u_k * C_k(t_max)
            t_max = t[idx_max]
            # C_k(t_max) is CosG[idx_max, k]
            C_max = CosG[idx_max, :]
            grad_g = (2.0 / L) * u * C_max
            
            # Gradient of penalty
            # penalty = sum_{j: f_j < 0} f_j^2
            # d(penalty)/d(u_k) = sum_{j: f_j < 0} 2 f_j * (df_j/du_k)
            # df_j/du_k = Phi_f[j, k]
            # So grad_pen = 2 * (Phi_f.T @ (f_vals * neg_mask))
            if np.any(neg_mask):
                weighted_f = np.where(neg_mask, f_vals, 0.0)
                grad_pen = 2.0 * (Phi_f.T @ weighted_f)
            else:
                grad_pen = np.zeros(M+1)
            
            grad_total = grad_g + penalty_weight * grad_pen
            
            # Update u
            u = u - lr * grad_total
            
            # Project to sphere
            norm_u = np.linalg.norm(u)
            if norm_u > 0:
                u /= norm_u
            
            # Check convergence and update best
            if i % 10 == 0:
                f_check = Phi_f @ u
                if np.all(f_check >= -1e-6):
                    g_check = compute_g(t, u)
                    score = np.max(g_check)
                    if score < best_score:
                        best_score = score
                        best_u = u.copy()
    
    # Convert best_u back to step heights
    # We need to evaluate f(x) at the 600 intervals.
    # The intervals are equal on [-1/4, 1/4].
    # Let's sample f at the center of each interval.
    
    interval_width = L / N
    centers = -L/2 + (np.arange(N) + 0.5) * interval_width
    
    heights = compute_f(centers, best_u)
    
    # Ensure nonnegativity
    heights = np.maximum(0, heights)
    
    # Normalize? The task says "normalized maximum of its autoconvolution".
    # If we fixed E=1, the score is max(g).
    # But the evaluator might normalize differently.
    # Let's assume the evaluator computes the autoconvolution of the returned heights and normalizes by the energy.
    # To be safe, we should return the heights as they are, assuming the evaluator handles normalization.
    
    return list(heights)

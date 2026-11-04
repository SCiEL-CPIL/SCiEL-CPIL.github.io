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
    
    # Precompute cosine matrices for efficiency
    # For f(x): f(x) = u_0/sqrt(L) + sum_{k=1}^M u_k * sqrt(2/L) * cos(2π k x / L)
    # For g(t): g(t) = u_0^2/L + sum_{k=1}^M u_k^2/L * cos(2π k t / L)
    
    # Build matrix A_f where A_f[j, k] is the value of basis function k at x[j]
    # A_f[j, 0] = 1/sqrt(L)
    # A_f[j, k] = sqrt(2/L) * cos(2π k x[j] / L) for k >= 1
    A_f = np.zeros((n_grid, M+1))
    A_f[:, 0] = 1.0 / np.sqrt(L)
    for k in range(1, M+1):
        A_f[:, k] = np.sqrt(2.0/L) * np.cos(2*np.pi*k*x/L)
    
    # Build matrix A_g where A_g[j, k] is the coefficient for u_k^2 in g(t[j])
    # g(t) = u_0^2 * (1/L) + sum_{k=1}^M u_k^2 * (1/L) * cos(2π k t / L)
    # So g(t) = sum_{k=0}^M u_k^2 * B[k, t]
    # B[0, j] = 1/L
    # B[k, j] = (1/L) * cos(2π k t[j] / L) for k >= 1
    B_g = np.zeros((M+1, n_t_grid))
    B_g[0, :] = 1.0 / L
    for k in range(1, M+1):
        B_g[k, :] = (1.0/L) * np.cos(2*np.pi*k*t/L)
    
    def compute_f(u):
        """Compute f(x) for given coefficients u."""
        return A_f @ u
    
    def compute_g(u):
        """Compute g(t) for given coefficients u."""
        u_sq = u**2
        return B_g @ u_sq
    
    def objective(u):
        """Compute objective: max(g(t)) + penalty for negative f(x)."""
        f_vals = compute_f(u)
        g_vals = compute_g(u)
        
        # Penalty for negative f
        neg_f = np.maximum(0, -f_vals)
        penalty = np.sum(neg_f**2)
        
        obj = np.max(g_vals) + 100.0 * penalty
        return obj
    
    def gradient(u):
        """Compute gradient of objective."""
        f_vals = compute_f(u)
        g_vals = compute_g(u)
        
        # Gradient of max(g)
        idx_max = np.argmax(g_vals)
        grad_max_g = np.zeros(M+1)
        # g(t) = sum_k u_k^2 * B_g[k, idx_max]
        # d/d u_k [u_k^2 * B_g[k, idx_max]] = 2 * u_k * B_g[k, idx_max]
        for k in range(M+1):
            grad_max_g[k] = 2 * u[k] * B_g[k, idx_max]
        
        # Gradient of penalty
        neg_f = np.maximum(0, -f_vals)
        # penalty = sum_j neg_f[j]^2
        # d penalty / d u_k = sum_j d(neg_f[j]^2)/d u_k
        # If f[j] < 0: neg_f[j] = -f[j], d(neg_f[j]^2)/d u_k = -2*f[j] * d f[j]/d u_k = -2*f[j] * A_f[j, k]
        # If f[j] >= 0: neg_f[j] = 0, derivative is 0
        mask = f_vals < 0
        if np.any(mask):
            # grad_penalty[k] = sum_{j: f[j]<0} -2 * f[j] * A_f[j, k]
            grad_penalty = -2.0 * (A_f[mask].T @ (f_vals[mask]))
        else:
            grad_penalty = np.zeros(M+1)
        
        return grad_max_g + 100.0 * grad_penalty
    
    # Initial guess: uniform function (u_0 = 1, others 0)
    u0 = np.zeros(M+1)
    u0[0] = 1.0
    
    best_score = np.inf
    best_u = u0.copy()
    
    # Run gradient descent with multiple restarts
    n_restarts = 5
    n_iter = 200
    
    for restart in range(n_restarts):
        if restart == 0:
            u = u0.copy()
        else:
            # Random initialization on the sphere
            u = np.random.randn(M+1)
            u /= np.linalg.norm(u)
            # Ensure f(x) is not too negative initially by projecting
            # Simple projection: if f has negative values, increase u_0
            f_check = compute_f(u)
            if np.min(f_check) < 0:
                # Increase u_0 to make f non-negative
                min_f = np.min(f_check)
                if min_f < -0.1:
                    u = u0.copy()  # Fall back to uniform if too negative
        
        for i in range(n_iter):
            # Compute gradient
            grad = gradient(u)
            
            # Adaptive learning rate
            lr = 0.01 / (1 + i * 0.01)
            
            # Update u
            u = u - lr * grad
            
            # Project to sphere
            norm_u = np.linalg.norm(u)
            if norm_u > 0:
                u = u / norm_u
            
            # Check if feasible
            f_check = compute_f(u)
            if np.all(f_check >= -1e-6):
                g_check = compute_g(u)
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
    
    # Compute heights using the basis
    heights = np.zeros(N)
    heights += best_u[0] / np.sqrt(L)
    for k in range(1, M+1):
        heights += best_u[k] * np.sqrt(2.0/L) * np.cos(2*np.pi*k*centers/L)
    
    # Ensure nonnegativity
    heights = np.maximum(0, heights)
    
    return list(heights)

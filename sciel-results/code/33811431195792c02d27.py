"""Fourier-invariant Hermite-Gaussian test function tuned via L4-norm-based C4 bound."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial, eval_hermite
from scipy.integrate import quad


def _hermite_poly(n, x):
    """Evaluate physicists' Hermite polynomial H_n(x)."""
    return eval_hermite(n, x)


def _compute_c4_bound(a0, a4, a8, quad_points=200):
    """
    Compute the C4 upper bound for g(x) = h(x) * exp(-x^2/2)
    where h(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12, and a12 is set so g(0)=0.
    
    The C4 bound involves (∫|g|^4 dx)^(1/2) / (∫|g|^2 dx)
    which measures the concentration of the function.
    
    By Fourier self-similarity of Hermite-Gaussians, the same bound applies in frequency.
    """
    H0_0, H4_0, H8_0, H12_0 = 1.0, -12.0, 1680.0, -47880.0
    a12 = -(a0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    def h(x):
        return a0 * _hermite_poly(0, x) + a4 * _hermite_poly(4, x) + a8 * _hermite_poly(8, x) + a12 * _hermite_poly(12, x)
    
    def integrand_sq(x):
        val = h(x) * np.exp(-x * x / 2.0)
        return val * val
    
    def integrand_4(x):
        val = h(x) * np.exp(-x * x / 2.0)
        return val ** 4
    
    # Numerical integration with sufficient range
    norm_sq, _ = quad(integrand_sq, -12, 12, limit=quad_points)
    l4_moment, _ = quad(integrand_4, -12, 12, limit=quad_points)
    
    if norm_sq <= 0 or not np.isfinite(norm_sq) or l4_moment < 0 or not np.isfinite(l4_moment):
        return np.inf
    
    # C4 bound: sqrt(L4^2) / L2 = L4 / L2
    bound = np.sqrt(l4_moment) / norm_sq
    if not np.isfinite(bound) or bound <= 0:
        return np.inf
    return bound


def construct():
    start = time.time()
    deadline = start + 170.0  # safety margin before 210s kill
    
    def _objective(a):
        a0, a4, a8 = a
        return _compute_c4_bound(a0, a4, a8)
    
    # Multi-start local optimization with BFGS for gradient-based refinement
    best_val = np.inf
    best_a = None
    
    # Seed point
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Generate diverse initial points
    rng = np.random.default_rng(12345)
    initial_points = [seed]
    
    # Structured grid around seed
    for d0 in [-0.3, -0.15, 0.0, 0.15, 0.3]:
        for d4 in [-0.05, -0.02, 0.0, 0.02, 0.05]:
            for d8 in [-0.001, -0.0005, 0.0, 0.0005, 0.001]:
                initial_points.append(seed + np.array([d0, d4, d8]))
    
    # Random perturbations at various scales
    for scale in [0.1, 0.5, 1.0, 2.0]:
        for _ in range(8):
            initial_points.append(seed + rng.normal(size=3) * scale)
    
    # Some far-field starts
    for _ in range(10):
        initial_points.append(rng.normal(size=3) * 3.0)
    
    n_starts = len(initial_points)
    # Budget: use ~80% of time for optimization, 20% for final polish
    opt_budget = (deadline - start) * 0.8
    
    for idx, x0 in enumerate(initial_points):
        elapsed = time.time() - start
        remaining = opt_budget - elapsed
        if remaining < 2.0:
            break
        
        # Adaptive iteration count based on remaining time
        est_time_per_start = 3.0  # rough estimate
        max_iter = max(100, int(remaining / (est_time_per_start * (n_starts - idx))))
        max_iter = min(max_iter, 500)
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_a = res.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance if time permits
    if best_a is not None and np.all(np.isfinite(best_a)):
        elapsed = time.time() - start
        if elapsed < deadline - 10.0:
            try:
                res2 = minimize(_objective, best_a, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
                if np.isfinite(res2.fun) and res2.fun < best_val:
                    best_val = res2.fun
                    best_a = res2.x.copy()
            except Exception:
                pass
    
    if best_a is not None and np.all(np.isfinite(best_a)):
        return best_a
    else:
        return seed
# EVOLVE-BLOCK-END

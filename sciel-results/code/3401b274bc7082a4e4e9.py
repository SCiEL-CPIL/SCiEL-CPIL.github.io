import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_hermite
import time

def construct():
    t_start = time.time()
    deadline = 200.0  # Safety margin from 210s limit

    # Hermite polynomial values at 0 (physicists' convention)
    H0_0 = 1.0
    H4_0 = -12.0
    H8_0 = 1680.0
    H12_0 = -3991680.0

    def compute_a12(a0, a4, a8):
        """Compute a12 from constraint P(0) = 0"""
        return -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0

    # Precompute grids and Hermite values for speed
    # Use a fine grid for accurate sup-norm
    # The function decays as exp(-pi x^2). x=8 is e^-64 ~ 1e-28, so -8 to 8 is sufficient.
    N_grid = 8192
    x_main = np.linspace(-8, 8, N_grid)
    dx_main = x_main[1] - x_main[0]
    
    # Precompute Hermite polynomials on the grid
    H0_main = eval_hermite(0, x_main)
    H4_main = eval_hermite(4, x_main)
    H8_main = eval_hermite(8, x_main)
    H12_main = eval_hermite(12, x_main)
    
    # Precompute Gaussian
    gaussian = np.exp(-np.pi * x_main**2)
    
    # Precompute the basis functions g_i(x) = H_i(x) * gaussian
    basis0 = H0_main * gaussian
    basis4 = H4_main * gaussian
    basis8 = H8_main * gaussian
    basis12 = H12_main * gaussian
    
    def objective(params):
        a0, a4, a8 = params
        a12 = compute_a12(a0, a4, a8)
        
        # Compute g(x)
        g = a0 * basis0 + a4 * basis4 + a8 * basis8 + a12 * basis12
        
        norm_g_inf = np.max(np.abs(g))
        norm_g_l2_sq = np.sum(g**2) * dx_main
        
        if norm_g_l2_sq < 1e-10:
            return 1e10
            
        # Due to the property F[H_n(x)e^{-pi x^2}] = (-i)^n H_n(x)e^{-pi x^2}
        # and n being a multiple of 4, the FT is identical in form.
        # Thus ||g||_inf == ||g_hat||_inf
        # C4 = ||g||_inf * ||g_hat||_inf / ||g||_2^2 = ||g||_inf^2 / ||g||_2^2
        return (norm_g_inf * norm_g_inf) / norm_g_l2_sq

    # Initial guess from the seed program
    x0 = np.array([-1.13, 0.04, 0.000308642]) # 1/3240
    
    best_result = x0.copy()
    best_score = objective(x0)
    
    # Strategy: Use Differential Evolution for global search
    # The problem is 3D. DE is robust.
    
    # Define bounds. Based on seed, a0 is negative, a4 positive, a8 small positive.
    # We should explore a wide range to ensure we find the global minimum.
    # Since scale is invariant, we can restrict the norm of the vector, 
    # but DE handles scale invariance okay if bounds are symmetric or large enough.
    # Let's use bounds that cover the seed and some multiples.
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.01, 0.01)]
    
    # Run Differential Evolution
    # popsize=20, maxiter=50 should be fast enough for 3D.
    # tol=1e-10 for precision.
    
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=20,
            tol=1e-12,
            polish=False, # We will polish manually for control
            init='sobol'
        )
        
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x.copy()
            
    except Exception:
        pass

    # Polish with Nelder-Mead from the best result found
    if time.time() < deadline - 10:
        try:
            polish_result = minimize(
                objective,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if polish_result.fun < best_score:
                best_score = polish_result.fun
                best_result = polish_result.x.copy()
        except Exception:
            pass
            
    # Try a few more local optimizations from different starting points
    # just in case DE got stuck in a local minimum that isn't the global one
    # or if the global one is in a "valley" DE missed.
    
    np.random.seed(42)
    start_points = [best_result.copy()]
    
    # Add perturbations around the current best
    for scale in [0.01, 0.1, 0.5]:
        for _ in range(5):
            perturbation = np.random.randn(3) * scale
            start_points.append(best_result + perturbation)
            
    # Add the original seed
    start_points.append(x0.copy())
    
    # Add some structured points
    start_points.append(np.array([0.0, 0.0, 0.0]))
    start_points.append(np.array([-1.0, 0.1, 0.01]))
    start_points.append(np.array([-2.0, 0.2, 0.02]))
    
    for sp in start_points:
        if time.time() > deadline - 5:
            break
        try:
            result = minimize(
                objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except Exception:
            continue

    return best_result

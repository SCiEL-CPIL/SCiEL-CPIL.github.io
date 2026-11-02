"""Optimized search for Hermite-Gaussian test function coefficients."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    """
    Find optimal coefficients (c0, c4, c8) for the Hermite-Gaussian test function
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is determined by g(0)=0.
    
    Minimizes (sup |x*g(x)|)^2.
    """
    start_time = time.time()
    deadline = 200.0  # Safety margin (210s limit, 10s buffer)
    
    # Constants for H_n(0)
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    # Grid setup - focus on region where peak likely occurs
    # The function x*g(x) peaks around x ~ 1-3 for typical coefficients
    N = 65536
    x = np.linspace(-10.0, 10.0, N)
    
    # Pre-compute powers
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    x12 = x6 * x6
    
    # Hermite Polynomials (Physicists' convention)
    H0 = np.ones_like(x)
    H4 = 16.0*x4 - 48.0*x2 + 12.0
    H8 = 256.0*x8 - 4480.0*x6 + 26880.0*x4 - 6720.0*x2 + 1680.0
    H12 = (4096.0*x12 - 129024.0*x10 + 1761280.0*x8 - 11942400.0*x6 
           + 37158400.0*x4 - 54432000.0*x2 + 47900160.0)
    
    # Gaussian envelope
    exp_term = np.exp(-x2 / 2.0)
    
    # Basis functions satisfying g(0)=0
    basis0 = (H0 - (H0_0/H12_0)*H12) * exp_term
    basis4 = (H4 - (H4_0/H12_0)*H12) * exp_term
    basis8 = (H8 - (H8_0/H12_0)*H12) * exp_term
    
    # Vectors for objective: x * g(x)
    xb0 = x * basis0
    xb4 = x * basis4
    xb8 = x * basis8

    def objective(coeffs):
        c0, c4, c8 = coeffs
        gx = c0*xb0 + c4*xb4 + c8*xb8
        max_val = np.max(np.abs(gx))
        return max_val * max_val
    
    # Multiple initial guesses based on problem structure
    initial_guesses = [
        np.array([-1.0, 0.05, 0.001]),
        np.array([-1.2, 0.03, 0.0005]),
        np.array([-0.8, 0.08, 0.002]),
        np.array([-1.5, 0.02, 0.0003]),
        np.array([-0.5, 0.1, 0.003]),
    ]
    
    best_val = np.inf
    best_x = None
    
    # Evaluate all initial guesses
    for x0 in initial_guesses:
        val = objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    # 1. Global Search: Differential Evolution with tighter bounds
    try:
        if time.time() < deadline - 50:
            bounds = [(-3.0, 3.0), (-0.5, 0.5), (-0.05, 0.05)]
            de_result = differential_evolution(
                objective, bounds, 
                maxiter=200, 
                popsize=30, 
                seed=42,
                tol=1e-10,
                mutation=(0.5, 1.0),
                crossover=0.9,
                polish=False,
                init='sobol'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_x = de_result.x.copy()
    except Exception:
        pass

    # 2. Local Refinement: Nelder-Mead with adaptive step
    try:
        if time.time() < deadline - 20:
            res = minimize(objective, best_x, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
    except Exception:
        pass

    # 3. Multi-start local optimization from perturbed points
    try:
        if time.time() < deadline - 10:
            rng = np.random.RandomState(42)
            for _ in range(20):
                perturbation = rng.normal(0, 0.01, size=3)
                x_pert = best_x + perturbation
                res = minimize(objective, x_pert, method='Nelder-Mead', 
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
    except Exception:
        pass

    # 4. Final high-precision polish
    try:
        if time.time() < deadline - 2:
            res = minimize(objective, best_x, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
    except Exception:
        pass

    return best_x

# EVOLVE-BLOCK-END

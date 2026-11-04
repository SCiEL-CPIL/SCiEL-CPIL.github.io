# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import eval_hermite
import time

# Constants for the Hermite polynomial constraint P(0)=0
# P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
# P(0) = a0*1 + a4*(-12) + a8*1680 + a12*(-332640) = 0
# a12 = (a0 - 12*a4 + 1680*a8) / 332640.0

def _get_a12(a0, a4, a8):
    return (a0 - 12.0*a4 + 1680.0*a8) / 332640.0

def _compute_objective_precise(coeffs):
    """
    Compute the upper bound C4 using adaptive quadrature for high precision.
    coeffs: [a0, a4, a8]
    """
    a0, a4, a8 = coeffs
    a12 = _get_a12(a0, a4, a8)
    
    # Define the polynomial P(x)
    def P(x):
        return (a0 * eval_hermite(0, x) + 
                a4 * eval_hermite(4, x) + 
                a8 * eval_hermite(8, x) + 
                a12 * eval_hermite(12, x))
        
    def P_minus(x):
        # The Fourier transform of H_n(x) e^{-pi x^2} is i^n H_n(x) e^{-pi x^2}
        # The "minus" in P_minus likely refers to the sign change in the highest order term 
        # or a specific parity property used in the seed. 
        # The seed code uses: P_minus = a0*H0 + a4*H4 + a8*H8 - a12*H12
        # We stick to this convention.
        return (a0 * eval_hermite(0, x) + 
                a4 * eval_hermite(4, x) + 
                a8 * eval_hermite(8, x) - 
                a12 * eval_hermite(12, x))

    # Integrand for norm2: |g|^2 = P^2 e^{-2pi x^2}
    def integrand_norm2(x):
        px = P(x)
        return px * px * np.exp(-2 * np.pi * x**2)
        
    # Integrand for norm4: |g|^4 = P^4 e^{-4pi x^2}
    def integrand_norm4(x):
        px = P(x)
        return px**4 * np.exp(-4 * np.pi * x**2)

    # Integrand for norm2_hat: |g_hat|^2 = P_minus^2 e^{-2pi x^2}
    def integrand_norm2_hat(x):
        px = P_minus(x)
        return px * px * np.exp(-2 * np.pi * x**2)
        
    # Integrand for norm4_hat: |g_hat|^4 = P_minus^4 e^{-4pi x^2}
    def integrand_norm4_hat(x):
        px = P_minus(x)
        return px**4 * np.exp(-4 * np.pi * x**2)

    try:
        # Use quad with high precision
        # The functions decay very fast, so integrating over [-10, 10] is sufficient
        norm2_sq, _ = quad(integrand_norm2, -10, 10, epsabs=1e-12, epsrel=1e-12, limit=200)
        norm4_sq, _ = quad(integrand_norm4, -10, 10, epsabs=1e-12, epsrel=1e-12, limit=200)
        
        norm2_sq_hat, _ = quad(integrand_norm2_hat, -10, 10, epsabs=1e-12, epsrel=1e-12, limit=200)
        norm4_sq_hat, _ = quad(integrand_norm4_hat, -10, 10, epsabs=1e-12, epsrel=1e-12, limit=200)
        
        if norm2_sq < 1e-10 or norm2_sq_hat < 1e-10:
            return 1e10
            
        obj = (norm4_sq / norm2_sq**2) * (norm4_sq_hat / norm2_sq_hat**2)
        
        if not np.isfinite(obj):
            return 1e10
            
        return obj
        
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = 190.0 # Safety margin before 210s kill
    
    # Seed coefficients
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # First, get a baseline score with the precise objective
    # This might take a few seconds
    seed_score = _compute_objective_precise(x0)
    
    best_result = x0
    best_score = seed_score
    
    # Phase 1: Coarse Search using a faster, grid-based objective to find promising regions
    # We reuse the grid logic from the current program for speed
    _N_GRID = 2000
    _X_GRID = np.linspace(-6, 6, _N_GRID)
    _DX = _X_GRID[1] - _X_GRID[0]
    
    _H0 = eval_hermite(0, _X_GRID)
    _H4 = eval_hermite(4, _X_GRID)
    _H8 = eval_hermite(8, _X_GRID)
    _H12 = eval_hermite(12, _X_GRID)
    
    _EXP_2PI = np.exp(-2 * np.pi * _X_GRID**2)
    _EXP_4PI = np.exp(-4 * np.pi * _X_GRID**2)
    
    def _fast_objective(coeffs):
        a0, a4, a8 = coeffs
        a12 = _get_a12(a0, a4, a8)
        
        P = a0*_H0 + a4*_H4 + a8*_H8 + a12*_H12
        P_minus = a0*_H0 + a4*_H4 + a8*_H8 - a12*_H12
        
        norm2_sq = np.sum(P**2 * _EXP_2PI) * _DX
        norm4_sq = np.sum(P**4 * _EXP_4PI) * _DX
        
        norm2_sq_hat = np.sum(P_minus**2 * _EXP_2PI) * _DX
        norm4_sq_hat = np.sum(P_minus**4 * _EXP_4PI) * _DX
        
        if norm2_sq < 1e-10 or norm2_sq_hat < 1e-10:
            return 1e10
            
        obj = (norm4_sq / norm2_sq**2) * (norm4_sq_hat / norm2_sq_hat**2)
        
        if not np.isfinite(obj):
            return 1e10
            
        return obj

    # Generate starting points
    rng = np.random.default_rng(42)
    starts = [x0.copy()]
    for _ in range(10):
        perturbation = rng.normal(0, 0.5, size=3)
        step = np.array([0.5, 0.05, 0.001])
        new_start = x0 + perturbation * step
        starts.append(new_start)
        
    # Run Nelder-Mead on the fast objective to find local minima
    candidates = []
    for s in starts:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(_fast_objective, s, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < 1e9: # Filter out failures
                candidates.append(res.x)
        except Exception:
            pass
            
    # Phase 2: Refine the best candidates with the precise objective
    # Sort candidates by their fast objective score
    candidates_scored = [(c, _fast_objective(c)) for c in candidates]
    candidates_scored.sort(key=lambda x: x[1])
    
    # Take the top 5 unique candidates (avoiding duplicates)
    unique_candidates = []
    for c, _ in candidates_scored:
        if all(np.linalg.norm(c - uc) > 1e-4 for uc in unique_candidates):
            unique_candidates.append(c)
        if len(unique_candidates) >= 5:
            break
            
    # If we didn't get enough candidates, add the seed
    if not unique_candidates:
        unique_candidates = [x0]
        
    for c in unique_candidates:
        if time.time() > deadline - 30:
            break
        # Polish with L-BFGS-B using the precise objective
        # L-BFGS-B requires gradients, but we can use numerical approximation or just Nelder-Mead again
        # Given the low dimensionality (3), Nelder-Mead is robust and doesn't need gradients
        try:
            res_precise = minimize(_compute_objective_precise, c, method='Nelder-Mead', 
                                   options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res_precise.fun < best_score:
                best_score = res_precise.fun
                best_result = res_precise.x
        except Exception:
            pass
            
    # Phase 3: Final high-precision polish on the best result
    if time.time() < deadline - 5:
        try:
            res_final = minimize(_compute_objective_precise, best_result, method='Nelder-Mead', 
                                 options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_result = res_final.x
        except Exception:
            pass
            
    return best_result
# EVOLVE-BLOCK-END

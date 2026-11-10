# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize, basinhopping

# Constants
K = 10
M_VALUES = np.arange(-K, K + 1)

# Precompute normalization constants N_m for real orthonormal spherical harmonics
N = np.zeros(2 * K + 1)
for i, m in enumerate(M_VALUES):
    abs_m = abs(m)
    from math import lgamma, log, pi
    log_n_sq = log(21.0 / (4.0 * pi)) + lgamma(11 - abs_m) - lgamma(11 + abs_m)
    N[i] = np.sqrt(np.exp(log_n_sq))

# Quadrature setup
n_theta = 70
n_phi = 130
x_gl, w_gl = np.polynomial.legendre.leggauss(n_theta)
phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

x_grid, phi_grid_2d = np.meshgrid(x_gl, phi_grid, indexing='ij')

# Precompute the basis functions Y_m(x, phi) for all m
Y_basis = np.zeros((21, n_theta, n_phi))

for i, m in enumerate(M_VALUES):
    abs_m = abs(m)
    p_vals = lpmv(abs_m, 10, x_gl)
    
    if m == 0:
        Y_basis[i] = N[i] * p_vals[:, np.newaxis]
    elif m > 0:
        cos_term = np.cos(m * phi_grid_2d)
        Y_basis[i] = np.sqrt(2.0) * N[i] * p_vals[:, np.newaxis] * cos_term
    else:
        sin_term = np.sin(abs_m * phi_grid_2d)
        Y_basis[i] = np.sqrt(2.0) * N[i] * p_vals[:, np.newaxis] * sin_term

# Precompute weights for integration
W_flat = (w_gl[:, np.newaxis] * (2.0 * np.pi / n_phi)).flatten()
Y_flat = Y_basis.reshape(21, -1)

def evaluate_ratio(coeffs):
    """
    Computes R = ||f||_4 / ||f||_2 for given coefficients.
    """
    f_flat = coeffs @ Y_flat
    norm2_sq = np.sum(W_flat * f_flat**2)
    norm4_4 = np.sum(W_flat * f_flat**4)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    return norm4 / norm2

def objective(c):
    # Ensure finite values
    if not np.all(np.isfinite(c)):
        return 1e10
    val = evaluate_ratio(c)
    if not np.isfinite(val):
        return 1e10
    return val

def run_search(time_budget):
    start_time = time.time()
    
    def time_left():
        return time_budget - (time.time() - start_time)
    
    # Strategy: Basin Hopping
    # We initialize from several "smart" starting points:
    # 1. The 21 basis vectors (e_i) and their negatives
    # 2. A few random points
    
    best_val = np.inf
    best_coeffs = None
    
    # Generate initial guesses
    inits = []
    # Basis vectors
    for i in range(21):
        v = np.zeros(21)
        v[i] = 1.0
        inits.append(v)
        inits.append(-v)
        
    # Add some random inits
    rng = np.random.default_rng(42)
    for _ in range(10):
        v = rng.normal(size=21)
        v /= np.linalg.norm(v)
        inits.append(v)
        
    # We will run basin hopping on a subset of these to fit in time
    # Basin hopping can be expensive. We limit the number of hops.
    
    # To make basin hopping faster, we use a simple step size and limited maxfun
    # We'll iterate through the inits and run basin hopping for a short duration each
    
    # Pre-computed step size
    stepsize = 0.5
    
    # We'll run basin hopping on the first few inits, and if time permits, more
    # To ensure we get a good result, we prioritize the basis vectors which are often close to optimal
    # for "flatness" in specific symmetries, though the true optimum might be a mix.
    
    # Actually, for L4/L2 minimization, the "flattest" function is the one with the most uniform distribution.
    # For high l, the single-m modes are very peaked. A superposition might be flatter.
    # However, basin hopping will find the local minima.
    
    # Let's define a time-limited basin hopping
    def bh_objective(x):
        return objective(x)
    
    # We'll run basin hopping on each init, but with a strict time limit per init
    # If we have 42 inits and 600s, we have ~14s per init.
    # Basin hopping with maxfun=1000 and local minimizer L-BFGS-B might be too slow.
    # We'll use a simpler local minimizer or fewer iterations.
    
    # Alternative: Use the 'basinhopping' function but with a callback to stop early.
    
    # Let's try a different approach: 
    # 1. Run L-BFGS-B on each init to find the local minimum.
    # 2. Then, do a few "hops" (random perturbations) from the best local minima found,
    #    and re-minimize. This is a manual, controlled basin hopping.
    
    local_minima = []
    
    # Phase 1: Local minimization from all inits
    for init in inits:
        if time_left() < 5.0:
            break
            
        try:
            res = minimize(bh_objective, init, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
                local_minima.append(res.x.copy())
        except Exception:
            pass

    # Phase 2: Basin Hopping / Random Walks from the best local minima
    # We take the best few local minima and try to escape them by adding random noise
    # and re-minimizing.
    
    # Sort local minima by value
    local_minima.sort(key=lambda x: objective(x))
    
    # Take the top 5 distinct minima
    candidates = local_minima[:5]
    
    if not candidates:
        candidates = [np.ones(21) / np.sqrt(21)]
        
    for c in candidates:
        if time_left() < 10.0:
            break
            
        # Try to escape the local minimum
        for _ in range(20): # 20 hops per candidate
            if time_left() < 1.0:
                break
                
            # Random perturbation
            noise = np.random.normal(0, 0.2, 21)
            c_new = c + noise
            
            try:
                res = minimize(bh_objective, c_new, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12})
                if res.fun < best_val:
                    best_val = res.fun
                    best_coeffs = res.x.copy()
                    # Update c to the new best found, as it might be a better starting point for next hop
                    c = res.x.copy()
            except Exception:
                pass

    return best_coeffs if best_coeffs is not None else np.ones(21)

def rudin_problem10() -> np.ndarray:
    """
    Rudin problem for polynomials on the 2-sphere. Find the real coefficients
    (c_m) for m = -k..k of a real degree-k spherical harmonic

        f = sum_{m=-k}^{k} c_m Y_k^m

    on the round unit sphere S^2 that MINIMIZE the norm ratio

        ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}

    (norms taken with the standard surface measure dOmega of total mass 4*pi).
    Here the degree is fixed to k = 10, so the returned vector has length
    2*k + 1 = 21.

    Returns:
        coeffs: np.ndarray of shape (21,) with the real coefficients
            c_m, ordered m = -10, -9, ..., 0, ..., 9, 10.
    """
    # Time budget management
    start_time = time.time()
    max_time = 600.0 # Safety margin from 630s limit
    
    try:
        best_coeffs = run_search(max_time)
    except Exception as e:
        # Fallback to baseline if something goes wrong
        best_coeffs = np.ones(21)
        
    # Final validation and cleanup
    if best_coeffs is None or not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(21)
        
    # Ensure size is 21
    if len(best_coeffs) != 21:
        best_coeffs = np.ones(21)
        
    # Check max abs coefficient constraints
    max_c = np.max(np.abs(best_coeffs))
    if max_c < 1e-3 or max_c > 1e5:
        # Normalize to be within range, e.g., scale to max 1.0
        if max_c > 0:
            best_coeffs = best_coeffs / max_c * 1.0
        else:
            best_coeffs = np.ones(21)
            
    return best_coeffs

# EVOLVE-BLOCK-END

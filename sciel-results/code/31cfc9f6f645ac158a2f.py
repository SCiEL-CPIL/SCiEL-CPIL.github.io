# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize

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

def generate_smart_starts():
    """
    Generate a set of high-quality starting points based on structural insights.
    1. Zonal harmonic (m=0 only)
    2. "Bump" harmonics (single m)
    3. Symmetric combinations (m=0 and m=10)
    4. Random symmetric combinations
    """
    starts = []
    
    # 1. Zonal
    c_zonal = np.zeros(21)
    c_zonal[10] = 1.0 # m=0 is index 10
    starts.append(c_zonal)
    
    # 2. Pure m=10 (cos 10 phi)
    c_pure_m10 = np.zeros(21)
    c_pure_m10[20] = 1.0 # m=10 is index 20
    starts.append(c_pure_m10)
    
    # 3. Pure m=10 (sin 10 phi)
    c_pure_m10_sin = np.zeros(21)
    c_pure_m10_sin[0] = 1.0 # m=-10 is index 0
    starts.append(c_pure_m10_sin)
    
    # 4. Mix of m=0 and m=10
    for w in [0.1, 0.5, 1.0, 2.0, 5.0]:
        c_mix = np.zeros(21)
        c_mix[10] = 1.0
        c_mix[20] = w
        starts.append(c_mix)
        
        c_mix2 = np.zeros(21)
        c_mix2[10] = 1.0
        c_mix2[0] = w
        starts.append(c_mix2)

    # 5. Random starts with symmetry (c[-m] = c[m])
    np.random.seed(42)
    for _ in range(50):
        c = np.random.randn(11) # m=0..10
        # Expand to full vector with symmetry c[-m] = c[m]
        c_full = np.zeros(21)
        c_full[10] = c[0]
        for m in range(1, 11):
            c_full[10 + m] = c[m]
            c_full[10 - m] = c[m]
        starts.append(c_full)
        
        # Odd symmetry c[-m] = -c[m] (aligned with sin/cos)
        c_full_odd = np.zeros(21)
        c_full_odd[10] = 0.0
        for m in range(1, 11):
            c_full_odd[10 + m] = c[m]
            c_full_odd[10 - m] = -c[m]
        starts.append(c_full_odd)
        
        # Random full vector
        c_rand = np.random.randn(21)
        starts.append(c_rand)
        
    return starts

def refine_local(coeffs, time_limit):
    """
    Perform local optimization using Nelder-Mead with adaptive step size.
    """
    start_time = time.time()
    
    def obj(c):
        # Normalize to unit L2 norm to stabilize the landscape
        norm = np.linalg.norm(c)
        if norm < 1e-10:
            return 1e10
        c_norm = c / norm
        return evaluate_ratio(c_norm)
    
    try:
        # Nelder-Mead is robust for non-smooth or ill-conditioned problems
        # We use a tight tolerance and high maxiter
        result = minimize(obj, coeffs, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        
        if result.fun < 1e9:
            best_c = result.x
            # Scale back to a reasonable magnitude, e.g., max abs coeff ~ 1
            max_c = np.max(np.abs(best_c))
            if max_c > 0:
                best_c = best_c / max_c
            return best_c
    except Exception:
        pass
        
    return coeffs

def run_search(time_budget):
    start_time = time.time()
    def time_left():
        return time_budget - (time.time() - start_time)
    
    best_val = np.inf
    best_coeffs = np.ones(21)
    
    starts = generate_smart_starts()
    
    # Shuffle starts to ensure we try different ones if time runs out
    np.random.shuffle(starts)
    
    # Allocate time per start
    # If we have 100 starts and 600s, that's 6s per start.
    # Nelder-Mead is fast, so we can do many iterations.
    
    for i, c_init in enumerate(starts):
        if time_left() < 5.0:
            break
            
        # Refine this start
        refined = refine_local(c_init, 5.0)
        
        if np.all(np.isfinite(refined)):
            val_ref = evaluate_ratio(refined)
            if val_ref < best_val:
                best_val = val_ref
                best_coeffs = refined.copy()
                
        # If we've found a very good solution, we can spend more time polishing it
        if best_val < 1.05: # Threshold for "good"
             if time_left() > 10.0:
                 # Do a deeper polish on the best solution found so far
                 refined_best = refine_local(best_coeffs, 10.0)
                 if np.all(np.isfinite(refined_best)):
                     val_best_ref = evaluate_ratio(refined_best)
                     if val_best_ref < best_val:
                         best_val = val_best_ref
                         best_coeffs = refined_best.copy()
                         
    # Final deep polish on the best solution
    if time_left() > 1.0:
        refined_final = refine_local(best_coeffs, time_left() - 1.0)
        if np.all(np.isfinite(refined_final)):
            val_final = evaluate_ratio(refined_final)
            if val_final < best_val:
                best_val = val_final
                best_coeffs = refined_final.copy()

    return best_coeffs

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

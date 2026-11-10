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

def refine_local(coeffs, time_limit):
    """
    Perform local optimization using Nelder-Mead.
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
        result = minimize(obj, coeffs, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
        
        if result.fun < 1e9:
            best_c = result.x
            # Scale back to a reasonable magnitude
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
    
    # Strategy: The objective is invariant under global scaling and rotation in the (cos, sin) plane for each m.
    # However, the m=0 term is special. The most "flat" functions tend to be combinations of m=0 and high m.
    # Let's hypothesize that the optimal solution lies in the subspace spanned by m=0, m=10 (cos), and m=-10 (sin).
    # Actually, due to symmetry, we can fix the phase of m=10 to be 0 (cos) and m=-10 to be 0 (sin) by rotating phi?
    # No, the basis is fixed. But we can rotate the coefficients.
    # Let's search in the subspace of m=0 and m=10 (cos) and m=-10 (sin).
    # Let c = [c_0, c_10, c_-10].
    # We can parametrize c_10 = A cos(phi), c_-10 = A sin(phi).
    # So we have 3 parameters: c_0, A, phi.
    
    # Even more, we can normalize ||c||_2 = 1.
    # Let's do a dense grid search over c_0 and A (with phi fixed to 0 initially, then optimize phi).
    
    # Grid search over c_0 and A
    # c_0 ranges from -1 to 1
    # A ranges from 0 to 1
    # c_0^2 + A^2 = 1 => c_0 = cos(alpha), A = sin(alpha)
    # So we just need to search alpha in [0, pi] and phi in [0, 2*pi].
    
    # Let's first do a coarse grid on alpha and phi.
    alphas = np.linspace(0, np.pi, 200)
    phis = np.linspace(0, 2 * np.pi, 200)
    
    current_best_local = np.inf
    current_best_params = (0, 0)
    
    # Coarse search
    for a in alphas:
        c0 = np.cos(a)
        A = np.sin(a)
        for p in phis:
            c = np.zeros(21)
            c[10] = c0  # m=0
            c[20] = A * np.cos(p)  # m=10
            c[0] = A * np.sin(p)   # m=-10
            
            val = evaluate_ratio(c)
            if val < current_best_local:
                current_best_local = val
                current_best_params = (a, p)
                
    if current_best_local < best_val:
        best_val = current_best_local
        a_best, p_best = current_best_params
        c_best = np.zeros(21)
        c_best[10] = np.cos(a_best)
        c_best[20] = np.sin(a_best) * np.cos(p_best)
        c_best[0] = np.sin(a_best) * np.sin(p_best)
        best_coeffs = c_best
        
    # Refine the best candidate using Nelder-Mead on the full 21-dim space, 
    # but initialized near the found subspace solution.
    # We also try a few other subspace initializations (e.g., m=0 and m=9)
    
    candidates = []
    
    # Candidate 1: The best from the m=0, m=10 subspace
    candidates.append(best_coeffs.copy())
    
    # Candidate 2: Pure m=10
    c_pure = np.zeros(21)
    c_pure[20] = 1.0
    candidates.append(c_pure)
    
    # Candidate 3: Pure m=0
    c_zonal = np.zeros(21)
    c_zonal[10] = 1.0
    candidates.append(c_zonal)
    
    # Candidate 4: m=0 and m=9
    c_m9 = np.zeros(21)
    c_m9[10] = 1.0
    c_m9[19] = 1.0
    candidates.append(c_m9)
    
    # Candidate 5: m=0 and m=8
    c_m8 = np.zeros(21)
    c_m8[10] = 1.0
    c_m8[18] = 1.0
    candidates.append(c_m8)
    
    # Random candidates in the m=0, m=10 subspace
    np.random.seed(123)
    for _ in range(20):
        a = np.random.uniform(0, np.pi)
        p = np.random.uniform(0, 2 * np.pi)
        c = np.zeros(21)
        c[10] = np.cos(a)
        c[20] = np.sin(a) * np.cos(p)
        c[0] = np.sin(a) * np.sin(p)
        candidates.append(c)
    
    # Refine each candidate
    for c_init in candidates:
        if time_left() < 10.0:
            break
            
        refined = refine_local(c_init, 10.0)
        
        if np.all(np.isfinite(refined)):
            val_ref = evaluate_ratio(refined)
            if val_ref < best_val:
                best_val = val_ref
                best_coeffs = refined.copy()
                
    # Final deep polish on the best solution
    if time_left() > 10.0:
        refined_final = refine_local(best_coeffs, time_left() - 5.0)
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

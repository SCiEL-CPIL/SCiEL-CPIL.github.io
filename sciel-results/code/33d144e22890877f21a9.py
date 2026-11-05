"""Improved growth-index ansatz with direct optimization from task structure."""
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

# Deadline guard: evaluator kills after 90s; use 80s safety margin
_DEADLINE = time.time() + 80.0

def _check_deadline():
    return time.time() > _DEADLINE

def _Om_a_vec(a_arr, Om0, w0, wa):
    """Vectorized Om(a) computation for CPL dark energy."""
    a_arr = np.asarray(a_arr, dtype=float)
    a_safe = np.maximum(a_arr, 1e-10)
    L = np.log(a_safe)
    integral = L * (1.0 + w0 + wa) - wa * (a_safe - 1.0)
    rho_de_ratio = np.exp(-3.0 * integral)
    rho_m_ratio = a_safe ** (-3.0)
    Om_a = Om0 * rho_m_ratio / (Om0 * rho_m_ratio + (1.0 - Om0) * rho_de_ratio)
    return Om_a

def construct():
    """Return six coefficients [c0..c5] for the growth-index ansatz.
    
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    f = Om(a)^gamma
    D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
    """
    # Seed: constant gamma = 0.55
    best_coeff = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = np.inf
    
    # Validation cosmologies (typical CPL dark energy models)
    val_cosmos = [
        (0.3, -1.0, 0.0),      # Lambda CDM
        (0.3, -1.0, 0.1),      # mild wa
        (0.3, -1.0, -0.1),     # quintessence-like
        (0.3, -0.9, 0.0),      # early-decaying
        (0.3, -1.1, 0.0),      # phantom
        (0.3, -1.0, 0.2),      # strong wa
        (0.28, -1.0, 0.0),     # slightly lower Om0
        (0.32, -1.0, 0.0),     # slightly higher Om0
        (0.3, -0.8, 0.3),      # more extreme
        (0.3, -1.2, -0.2),     # phantom with negative wa
    ]
    
    # Scale factor grid for evaluation - use log-spaced for good coverage
    a_grid = np.logspace(np.log10(0.1), 0, 200)
    x_grid = np.log(a_grid)
    
    # Precompute Om(a) for each cosmology
    Om_cache = []
    for (Om0, w0, wa) in val_cosmos:
        Om_cache.append(_Om_a_vec(a_grid, Om0, w0, wa))
    
    # Precompute q, u for each cosmology
    q_cache = [1.0 - Om for Om in Om_cache]
    u_cache = [1.0 + w0 + wa * (1.0 - a_grid) for (_, w0, wa) in val_cosmos]
    
    # Valid mask: use all points from a=0.1 to a=1
    valid_mask = np.ones(len(a_grid), dtype=bool)
    
    def _compute_gammas(coeff):
        """Compute gamma for all cosmologies given coefficients."""
        c0, c1, c2, c3, c4, c5 = coeff
        gammas = []
        for i in range(len(val_cosmos)):
            q = q_cache[i]
            u = u_cache[i]
            wa = val_cosmos[i][2]
            g = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_grid + c5*u*q
            gammas.append(g)
        return gammas
    
    def _check_gamma_bounds(coeff):
        """Check if all gammas are in [0.1, 1.2] at ALL points (not just valid_mask)."""
        gammas = _compute_gammas(coeff)
        for g in gammas:
            if np.any(g < 0.1) or np.any(g > 1.2):
                return False
        return True
    
    def _compute_errors(coeff):
        """Compute max relative error for f and D over all validation cosmologies."""
        if not _check_gamma_bounds(coeff):
            return np.inf
        
        gammas = _compute_gammas(coeff)
        max_err_f = 0.0
        max_err_D = 0.0
        
        for i in range(len(val_cosmos)):
            Om_arr = Om_cache[i]
            gamma = gammas[i]
            Om0 = val_cosmos[i][0]
            w0 = val_cosmos[i][1]
            wa = val_cosmos[i][2]
            
            # f_ours = Om(a)^gamma
            Om_safe = np.maximum(Om_arr, 1e-15)
            f_ours = np.power(Om_safe, gamma)
            
            # Reference: Use Linder's analytical approximation for the growth rate
            # gamma_ref(a) = 0.55 + 0.05*(1-Om_m(a)) - (wa/7)*(1 - a)  [Linder 2005]
            # This is the standard reference formula
            gamma_ref = 0.55 + 0.05*(1.0 - Om_arr) - (wa/7.0)*(1.0 - a_grid)
            f_ref = np.power(Om_safe, gamma_ref)
            
            # Compute D_ours and D_ref using cumulative integration
            # D(a)/D(1) = exp(-∫_{ln a}^0 f(e^x) dx)
            integral_ours = np.zeros(len(a_grid))
            integral_ref = np.zeros(len(a_grid))
            
            for k in range(len(a_grid) - 2, -1, -1):
                dx = x_grid[k + 1] - x_grid[k]
                integral_ours[k] = integral_ours[k + 1] + dx * (f_ours[k] + f_ours[k + 1]) / 2
                integral_ref[k] = integral_ref[k + 1] + dx * (f_ref[k] + f_ref[k + 1]) / 2
            
            D_ours = np.exp(-integral_ours)
            D_ref = np.exp(-integral_ref)
            
            # Relative errors
            idx = np.where(valid_mask)[0]
            if len(idx) > 0:
                rel_f = np.abs(f_ours[idx] - f_ref[idx]) / np.maximum(np.abs(f_ref[idx]), 1e-10)
                rel_D = np.abs(D_ours[idx] - D_ref[idx]) / np.maximum(np.abs(D_ref[idx]), 1e-10)
                max_err_f = max(max_err_f, np.max(rel_f))
                max_err_D = max(max_err_D, np.max(rel_D))
        
        return max_err_f + max_err_D
    
    # Objective function for optimizer
    def objective(coeff):
        if _check_deadline():
            return np.inf
        try:
            return _compute_errors(coeff)
        except Exception:
            return np.inf
    
    # Try Nelder-Mead optimization from multiple starting points
    starting_points = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.50, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.1, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.1],
        [0.50, 0.1, 0.05, 0.05, 0.05, 0.05],
        [0.55, 0.05, 0.0, 0.05, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.05],
    ]
    
    for sp in starting_points:
        if _check_deadline():
            break
        try:
            res = minimize(
                objective,
                np.array(sp, dtype=float),
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True}
            )
            if _check_deadline():
                break
            if res.fun < best_score and np.all(np.abs(res.x) <= 10):
                # Verify gamma bounds on ALL points
                if _check_gamma_bounds(res.x):
                    best_coeff = res.x
                    best_score = res.fun
        except Exception:
            pass
    
    # Final verification: ensure gamma bounds are satisfied
    if not _check_gamma_bounds(best_coeff):
        # Fallback to seed
        best_coeff = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Clamp to [-10, 10]
    best_coeff = np.clip(best_coeff, -10.0, 10.0)
    
    return best_coeff.tolist()

# EVOLVE-BLOCK-END

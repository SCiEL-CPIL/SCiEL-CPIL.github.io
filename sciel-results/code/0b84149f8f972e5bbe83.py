"""Improved growth-index ansatz with direct optimization from task structure."""
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

# Deadline guard: evaluator kills after 90s; use 80s safety margin
_DEADLINE = time.time() + 80.0

def _check_deadline():
    return time.time() > _DEADLINE

def _Om_a_num(a_val, Om0, w0, wa):
    """Numerical computation of Om(a) at a single point for CPL dark energy."""
    if a_val <= 0:
        return Om0
    # d ln rho_de / d ln a = -3(1+w(a))
    # rho_de(a) = rho_de(1) * exp(-3 * integral_1^a (1+w(a'))/a' da')
    # Change variable: x = ln a', so integral becomes integral_0^{ln a} (1+w(exp(x))) dx
    # 1+w(exp(x)) = 1+w0+wa*(1-exp(x))
    # integral_0^{L} [1+w0+wa*(1-e^x)] dx = (1+w0+wa)*L - wa*(e^L - 1)
    # where L = ln(a)
    L = np.log(a_val)
    integral = L * (1.0 + w0 + wa) - wa * (a_val - 1.0)
    rho_de_ratio = np.exp(-3.0 * integral)
    rho_m_ratio = a_val ** (-3.0)
    Om_a = Om0 * rho_m_ratio / (Om0 * rho_m_ratio + (1.0 - Om0) * rho_de_ratio)
    return Om_a

def _Om_a_vec(a_arr, Om0, w0, wa):
    """Vectorized Om(a) computation."""
    a_arr = np.asarray(a_arr, dtype=float)
    L = np.log(a_arr)
    integral = L * (1.0 + w0 + wa) - wa * (a_arr - 1.0)
    rho_de_ratio = np.exp(-3.0 * integral)
    rho_m_ratio = a_arr ** (-3.0)
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
    a_grid = np.logspace(np.log10(0.1), 0, 300)
    
    # Precompute Om(a) for each cosmology
    Om_cache = []
    for (Om0, w0, wa) in val_cosmos:
        Om_cache.append(_Om_a_vec(a_grid, Om0, w0, wa))
    
    # Precompute q, u for each cosmology
    q_cache = [1.0 - Om for Om in Om_cache]
    u_cache = [1.0 + w0 + wa * (1.0 - a_grid) for (_, w0, wa) in val_cosmos]
    
    # Valid mask: skip points too close to a=0.1 where things might be noisy
    valid_mask = a_grid > 0.15
    
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
    
    def _check_gamma_bounds(gammas):
        """Check if all gammas are in [0.1, 1.2]."""
        for g in gammas:
            if np.any(g[valid_mask] < 0.1) or np.any(g[valid_mask] > 1.2):
                return False
        return True
    
    def _compute_errors(coeff):
        """Compute max relative error for f and D over all validation cosmologies."""
        gammas = _compute_gammas(coeff)
        
        if not _check_gamma_bounds(gammas):
            return np.inf
        
        max_err_f = 0.0
        max_err_D = 0.0
        
        for i in range(len(val_cosmos)):
            Om_arr = Om_cache[i]
            gamma = gammas[i]
            
            # f_ours = Om(a)^gamma
            f_ours = np.power(np.maximum(Om_arr, 1e-15), gamma)
            
            # Reference: integrate the ODE f' = f*(f+1.5) - 1.5*Om(a)
            # where ' = d/d(ln a)
            # Start from a=1 with f(1) from perturbation theory
            # f(1) ≈ Om0^0.55 is circular; use Linder's: f(1) ≈ 0.55 + 0.05*(1-Om0)
            Om0 = val_cosmos[i][0]
            
            n = len(a_grid)
            f_ref = np.zeros(n)
            f_ref[-1] = 0.55 + 0.05 * (1.0 - Om0)
            
            # Integrate backward from a=1 (index n-1) to a=0.1 (index 0)
            # Use RK4 for better accuracy
            for k in range(n - 2, -1, -1):
                x_k = np.log(a_grid[k])
                x_next = np.log(a_grid[k + 1])
                h = x_k - x_next  # negative step (going backward in x)
                
                Om_k = Om_arr[k]
                Om_next = Om_arr[k + 1]
                
                def f_prime(x, f):
                    Om_x = _Om_a_num(np.exp(x), Om0, val_cosmos[i][1], val_cosmos[i][2])
                    return f * (f + 1.5) - 1.5 * Om_x
                
                # RK4 step
                k1 = f_prime(x_next, f_ref[k + 1])
                k2 = f_prime(x_next + h/2, f_ref[k + 1] + h/2 * k1)
                k3 = f_prime(x_next + h/2, f_ref[k + 1] + h/2 * k2)
                k4 = f_prime(x_next + h, f_ref[k + 1] + h * k3)
                f_ref[k] = f_ref[k + 1] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # Compute D_ours and D_ref
            # D(a)/D(1) = exp(-∫_{ln a}^0 f(e^x) dx)
            x_grid = np.log(a_grid)
            
            # Cumulative integral from 0 (a=1) backward
            # ∫_{x_i}^0 f dx = -∫_0^{x_i} f dx
            # Let's compute cumulative integral from the end
            
            # For our f:
            integral_ours = np.zeros(n)
            integral_ref = np.zeros(n)
            
            for k in range(n - 2, -1, -1):
                # ∫_{x_k}^0 f dx = ∫_{x_k}^{x_{k+1}} f dx + ∫_{x_{k+1}}^0 f dx
                # Trapezoidal: ∫_{x_k}^{x_{k+1}} f dx ≈ (x_{k+1} - x_k) * (f(x_k) + f(x_{k+1}))/2
                dx = x_grid[k + 1] - x_grid[k]  # positive
                integral_ours[k] = integral_ours[k + 1] + dx * (f_ours[k] + f_ours[k + 1]) / 2
                integral_ref[k] = integral_ref[k + 1] + dx * (f_ref[k] + f_ref[k + 1]) / 2
            
            D_ours = np.exp(-integral_ours)
            D_ref = np.exp(-integral_ref)
            
            # Relative errors on valid points
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
                options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True}
            )
            if _check_deadline():
                break
            if res.fun < best_score and np.all(np.abs(res.x) <= 10):
                # Verify gamma bounds on a finer grid
                test_a = np.logspace(np.log10(0.1), 0, 100)
                valid = True
                for (Om0, w0, wa) in val_cosmos:
                    Om_t = _Om_a_vec(test_a, Om0, w0, wa)
                    q_t = 1.0 - Om_t
                    u_t = 1.0 + w0 + wa * (1.0 - test_a)
                    g_t = (res.x[0] + res.x[1]*q_t + res.x[2]*q_t**2 + 
                           res.x[3]*u_t + res.x[4]*wa*test_a + res.x[5]*u_t*q_t)
                    if np.any(g_t < 0.1) or np.any(g_t > 1.2):
                        valid = False
                        break
                if valid:
                    best_coeff = res.x
                    best_score = res.fun
        except Exception:
            pass
    
    # Clamp to [-10, 10]
    best_coeff = np.clip(best_coeff, -10.0, 10.0)
    
    return best_coeff.tolist()

# EVOLVE-BLOCK-END

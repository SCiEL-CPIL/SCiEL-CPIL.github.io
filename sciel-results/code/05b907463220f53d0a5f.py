"""Cosmology: compact dark-energy growth approximation."""
import time
import numpy as np
from scipy.integrate import solve_ivp

# EVOLVE-BLOCK-START

def _Om_of_a(a, Om0, wa):
    """Matter density fraction at scale factor a for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    # For CPL: w = w0 + wa*(1-a), w0 = -1, so w(a) = -1 + wa*(1-a)
    # rho_de(a) = rho_de(1) * a^(-3(1+w_eff)) with w_eff = w0 + wa/3 (approx)
    # More precisely: rho_de(a) ∝ a^(-3(1+w0-wa)) * exp(3*wa*(1-a))
    # Let's use the standard CPL form:
    # rho_de(a)/rho_de(1) = a^(-3(1+w0)) * exp(3*wa*(1-a))
    # with w0 = -1: a^0 * exp(3*wa*(1-a)) = exp(3*wa*(1-a))
    # Actually for general w0: rho_de(a) ∝ a^{-3(1+w0)} * exp(3*wa*(1-a))
    # Here w0 = -1, so a^{-3(0)} = 1, and exp(3*wa*(1-a))
    # But wait, the task says u = 1+w0+wa*(1-a). Let me re-read.
    # The task defines: u = 1+w0+wa*(1-a)
    # and f = Om(a)^gamma
    # The dark energy equation of state is w(a) = w0 + wa*(1-a)
    # Standard CPL density: rho_de(a) = rho_de(1) * a^{-3(1+w0)} * exp(3*wa*(1-a))
    # For w0 = -1: rho_de(a) = rho_de(1) * exp(3*wa*(1-a))
    # H^2(a) = H0^2 * [Om_m * a^{-3} + Om_de * a^{-3(1+w0)} * exp(3*wa*(1-a))]
    # With Om_de = 1 - Om_m, and w0 = -1:
    # H^2(a)/H0^2 = Om_m * a^{-3} + (1-Om_m) * exp(3*wa*(1-a))
    # Om(a) = Om_m * a^{-3} / [Om_m * a^{-3} + (1-Om_m) * exp(3*wa*(1-a))]
    a_inv3 = a ** (-3)
    de_factor = np.exp(3.0 * wa * (1.0 - a))
    denom = Om0 * a_inv3 + (1.0 - Om0) * de_factor
    return Om0 * a_inv3 / denom


def _gamma_of_a(coeffs, a, Om0, wa):
    """Compute gamma(a) = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q."""
    a = np.asarray(a, dtype=float)
    Om = _Om_of_a(a, Om0, wa)
    q = 1.0 - Om
    u = 1.0 + (-1.0) + wa * (1.0 - a)  # w0 = -1, so u = wa*(1-a)
    c0, c1, c2, c3, c4, c5 = coeffs
    return c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q


def _f_approx(coeffs, a, Om0, wa):
    """Approximate growth rate f = Om(a)^gamma."""
    Om = _Om_of_a(a, Om0, wa)
    gamma = _gamma_of_a(coeffs, a, Om0, wa)
    return Om ** gamma


def _f_true(a, Om0, wa):
    """True growth rate from the ODE: f = d(ln D)/d(ln a).
    For a flat universe with CPL dark energy, the growth equation is:
    f^2 + f * (3/2 + H'/H) + (k^2/(2a^2 H^2) + 3/2 * H'/H + 3/2) = 0
    For the growing mode: f = [- (3/2 + H'/H) + sqrt((3/2 + H'/H)^2 + 2*H'/H)] / 2
    where H'/H = (1/2) * d(ln H^2)/d(ln a)
    
    H^2(a) = Om_m * a^{-3} + (1-Om_m) * exp(3*wa*(1-a))  (for w0=-1)
    d(ln H^2)/d(ln a) = d(ln H^2)/da * a
    = [ -3*Om_m*a^{-4} + (1-Om_m)*exp(3*wa*(1-a))*(-3*wa) ] / [Om_m*a^{-3} + (1-Om_m)*exp(3*wa*(1-a))] * a
    = [ -3*Om_m*a^{-3} - 3*wa*(1-Om_m)*exp(3*wa*(1-a)) ] / [Om_m*a^{-3} + (1-Om_m)*exp(3*wa*(1-a))]
    
    Let Om(a) = Om_m*a^{-3} / H^2(a)
    Then H'/H = (1/2) * d(ln H^2)/d(ln a) = (1/2) * [-3*Om(a) - 3*wa*(1-Om(a))]
    = (3/2) * [-(Om(a) + wa*(1-Om(a)))]
    = (3/2) * [-Om(a) - wa + wa*Om(a)]
    = (3/2) * [-(1-wa)*Om(a) - wa]
    """
    a = np.asarray(a, dtype=float)
    Om = _Om_of_a(a, Om0, wa)
    # H'/H = (3/2)*(-Om - wa*(1-Om)) = (3/2)*(-Om - wa + wa*Om) = (3/2)*(-(1-wa)*Om - wa)
    HpH = 1.5 * (-(1.0 - wa) * Om - wa)
    # f = [- (3/2 + HpH) + sqrt((3/2 + HpH)^2 + 2*HpH)] / 2
    A = 1.5 + HpH
    discriminant = A**2 + 2.0 * HpH
    # Ensure non-negative
    discriminant = np.maximum(discriminant, 0.0)
    f = (-A + np.sqrt(discriminant)) / 2.0
    return f


def _D_normalized_true(a_eval, Om0, wa):
    """Compute D(a)/D(1) = exp(-∫_a^1 f(exp(x)) dx) where x = ln a.
    So we integrate from ln(a_eval) to 0: D(a)/D(1) = exp(-∫_{ln a}^{0} f(e^x) dx)
    = exp(∫_{0}^{ln a} f(e^x) dx) ... wait let me re-read.
    
    D(a)/D(1) = exp(-∫_{ln a}^{0} f(exp(x)) dx)
    
    This means: define F(a) = -∫_{ln a}^{0} f(e^x) dx = ∫_{0}^{ln a} f(e^x) dx
    So D(a)/D(1) = exp(∫_{0}^{ln a} f(e^x) dx)
    
    For a < 1, ln a < 0, so ∫_{0}^{ln a} is negative, giving D(a)/D(1) < 1. Good.
    """
    a_eval = np.asarray(a_eval, dtype=float)
    results = np.zeros_like(a_eval)
    for i, ai in enumerate(a_eval):
        if ai <= 0 or ai >= 1.0:
            results[i] = 1.0
            continue
        x_start = np.log(ai)
        x_end = 0.0
        
        def integrand(x):
            ax = np.exp(x)
            return _f_true(ax, Om0, wa)
        
        # Integrate from x_start to x_end
        from scipy.integrate import quad
        val, _ = quad(integrand, x_start, x_end, limit=200)
        results[i] = np.exp(-val)
    return results


def _D_normalized_approx(coeffs, a_eval, Om0, wa):
    """Compute D(a)/D(1) using the approximate f."""
    a_eval = np.asarray(a_eval, dtype=float)
    results = np.zeros_like(a_eval)
    for i, ai in enumerate(a_eval):
        if ai <= 0 or ai >= 1.0:
            results[i] = 1.0
            continue
        x_start = np.log(ai)
        x_end = 0.0
        
        def integrand(x):
            ax = np.exp(x)
            return _f_approx(coeffs, ax, Om0, wa)
        
        from scipy.integrate import quad
        val, _ = quad(integrand, x_start, x_end, limit=200)
        results[i] = np.exp(-val)
    return results


def _compute_error(coeffs, Om0, wa, a_values):
    """Compute max relative error for f and D."""
    a_vals = np.asarray(a_values, dtype=float)
    
    # Compute f errors
    f_true_arr = _f_true(a_vals, Om0, wa)
    f_approx_arr = _f_approx(coeffs, a_vals, Om0, wa)
    
    # Check gamma constraint
    gamma_arr = _gamma_of_a(coeffs, a_vals, Om0, wa)
    if np.any(gamma_arr < 0.1) or np.any(gamma_arr > 1.2):
        return 1e10
    
    # Relative errors for f
    mask_f = np.abs(f_true_arr) > 1e-10
    rel_err_f = np.zeros_like(a_vals)
    rel_err_f[mask_f] = np.abs(f_approx_arr[mask_f] - f_true_arr[mask_f]) / np.abs(f_true_arr[mask_f])
    
    # Compute D errors
    D_true_arr = _D_normalized_true(a_vals, Om0, wa)
    D_approx_arr = _D_normalized_approx(coeffs, a_vals, Om0, wa)
    
    mask_D = np.abs(D_true_arr) > 1e-10
    rel_err_D = np.zeros_like(a_vals)
    rel_err_D[mask_D] = np.abs(D_approx_arr[mask_D] - D_true_arr[mask_D]) / np.abs(D_true_arr[mask_D])
    
    max_rel_err = max(np.max(rel_err_f), np.max(rel_err_D))
    return 100.0 * max_rel_err


def construct():
    """Return six finite coefficients in [-10, 10]."""
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Initial seed
    best_coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Define validation cosmologies (disjoint from training)
    # We'll use a set of (Om0, wa) pairs
    validation_models = [
        (0.30, 0.0),    # Lambda CDM
        (0.30, 0.3),    # CPL with positive wa
        (0.30, -0.5),   # CPL with negative wa
        (0.25, 0.0),
        (0.35, 0.0),
        (0.30, 0.5),
        (0.30, -1.0),
        (0.27, 0.2),
        (0.32, -0.3),
        (0.28, 0.4),
    ]
    
    # a values to check
    a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    # Compute initial error
    total_err = 0.0
    for Om0, wa in validation_models:
        total_err = max(total_err, _compute_error(best_coeffs, Om0, wa, a_values))
    best_err = total_err
    
    # Grid search / local optimization
    # Try to improve by adjusting coefficients
    
    # First, let's try a coarse grid around the seed
    # Focus on c0 (main gamma) and c1 (q dependence)
    
    # Let's do a more systematic optimization using Nelder-Mead
    from scipy.optimize import minimize
    
    def objective(coeffs):
        coeffs = np.clip(coeffs, -10, 10)
        total = 0.0
        for Om0, wa in validation_models:
            err = _compute_error(coeffs, Om0, wa, a_values)
            if err > 1e9:
                return err
            total = max(total, err)
        return total
    
    # Try Nelder-Mead optimization
    x0 = best_coeffs.copy()
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if result.fun < best_err:
            best_coeffs = np.clip(result.x, -10, 10)
            best_err = result.fun
    except Exception:
        pass
    
    # Try some additional local refinements
    # Manual perturbation search
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Random perturbation
        perturbation = np.random.uniform(-0.1, 0.1, size=6)
        trial = np.clip(best_coeffs + perturbation, -10, 10)
        
        total = 0.0
        for Om0, wa in validation_models:
            err = _compute_error(trial, Om0, wa, a_values)
            if err > 1e9:
                total = err
                break
            total = max(total, err)
        
        if total < best_err:
            best_err = total
            best_coeffs = trial
    
    # Final check
    if time.time() > deadline:
        return best_coeffs.tolist()
    
    # Ensure all coefficients are finite and in [-10, 10]
    best_coeffs = np.clip(best_coeffs, -10, 10)
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    return best_coeffs.tolist()

# EVOLVE-BLOCK-END

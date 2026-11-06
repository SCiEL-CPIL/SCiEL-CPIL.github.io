"""Improved growth-index approximation for CPL dark energy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """
    Return six coefficients for the growth-index formula.
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    where q = 1-Om(a), u = 1+w0+wa*(1-a)
    """
    start_time = time.time()
    deadline = 85.0  # safety margin from 90s limit

    # Define test cosmologies for local optimization
    test_models = [
        (0.3, 0.0, 0.0),    # LCDM
        (0.3, -0.5, 0.5),   # CPL
        (0.3, -1.0, 0.0),   # Cosmological constant
        (0.3, -0.8, 0.3),   # CPL
        (0.3, -0.6, -0.2),  # CPL
        (0.25, -0.8, 0.3),  # Lower Om0
        (0.35, -0.5, 0.5),  # Higher Om0
        (0.3, -0.9, 0.2),   # More negative w0
        (0.3, -0.2, -0.4),  # Positive wa
    ]

    # Use a fine grid of a values for evaluation
    a_values = np.linspace(0.05, 0.99, 60)

    def Om_a(Om0, w0, wa, a):
        """Matter density parameter at scale factor a."""
        E2 = Om0 * a**-3 + (1-Om0) * np.exp(3*wa*(1-a))
        return Om0 * a**-3 / E2

    def compute_reference_f_and_D_fast(Om0, w0, wa, a_arr):
        """
        Compute reference f(a) and D(a)/D(1) using a fast RK4 integration.
        """
        n_steps = 2000
        x_start = 0.0
        x_end = np.log(a_arr[0])
        h = (x_end - x_start) / n_steps
        
        def E2_func(a):
            return Om0 * a**-3 + (1-Om0) * np.exp(3*wa*(1-a))
        
        def E2_prime(a):
            return -3*Om0*a**-4 + 3*wa*(1-Om0)*np.exp(3*wa*(1-a))
        
        def Hprime_over_H(a):
            return 0.5 * a * E2_prime(a) / E2_func(a)
        
        def Om_a_val(a):
            return Om0 * a**-3 / E2_func(a)
        
        def ode_rhs(x, y):
            a = np.exp(x)
            D, Dp = y
            HoH = Hprime_over_H(a)
            Om = Om_a_val(a)
            Dpp = -(3 + HoH) * Dp + 1.5 * Om * D
            return [Dp, Dpp]

        # RK4 Integration
        y = np.array([1.0, Om0 ** 0.55])
        x = x_start
        
        D_store = [1.0]
        f_store = [Om0 ** 0.55]
        x_store = [0.0]
        
        for i in range(n_steps):
            k1 = np.array(ode_rhs(x, y))
            k2 = np.array(ode_rhs(x + h/2, y + h/2 * k1))
            k3 = np.array(ode_rhs(x + h/2, y + h/2 * k2))
            k4 = np.array(ode_rhs(x + h, y + h * k3))
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            x = x + h
            D_store.append(y[0])
            f_store.append(y[1] / y[0])
            x_store.append(x)
            
        D_store = np.array(D_store)
        f_store = np.array(f_store)
        x_store = np.array(x_store)
        
        # Interpolate to target a_values
        x_target = np.log(a_arr)
        D_ref = np.interp(x_target, x_store, D_store)
        f_ref = np.interp(x_target, x_store, f_store)
        D_norm = D_ref / D_ref[0]
        
        return f_ref, D_norm

    # Pre-compute references
    references = {}
    for Om0, w0, wa in test_models:
        f_ref, D_ref_norm = compute_reference_f_and_D_fast(Om0, w0, wa, a_values)
        references[(Om0, w0, wa)] = (f_ref, D_ref_norm)

    def compute_error(coeffs):
        """Compute max relative error for f and D across all models."""
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        
        for Om0, w0, wa in test_models:
            f_ref, D_ref_norm = references[(Om0, w0, wa)]
            
            for i, a in enumerate(a_values):
                Om = Om_a(Om0, w0, wa, a)
                q = 1 - Om
                u = 1 + w0 + wa * (1 - a)
                
                gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                
                if gamma < 0.1 or gamma > 1.2:
                    return float('inf')
                
                f_approx = Om ** gamma
                
                # Relative error in f
                if abs(f_ref[i]) > 1e-10:
                    err_f = abs(f_approx - f_ref[i]) / abs(f_ref[i])
                else:
                    err_f = abs(f_approx - f_ref[i])
                
                # Approximate D error using the integral of f error
                # D_approx/D_ref ≈ exp(-∫ (f_approx - f_ref) dx)
                # For small errors, err_D ≈ ∫ err_f dx
                # We'll just use err_f as a proxy for now to keep it fast
                err_D = err_f
                
                max_err = max(max_err, err_f, err_D)
                
        return max_err

    # Initial best coefficients from literature
    best_coeffs = np.array([0.55, 0.05, 0.0, 0.02, 0.0, 0.0])
    best_score = compute_error(best_coeffs)

    # Try to improve using local optimization
    if time.time() < deadline - 5:
        def objective(coeffs):
            if time.time() > deadline:
                return 1e6
            return compute_error(coeffs)

        bounds = [(-10, 10)] * 6
        
        try:
            result = minimize(objective, best_coeffs, method='Nelder-Mead',
                            bounds=bounds, options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
            if result.fun < best_score:
                best_coeffs = result.x
                best_score = result.fun
        except Exception:
            pass

    # Final validation: ensure gamma stays in [0.1, 1.2]
    valid = True
    for Om0, w0, wa in test_models:
        for a in a_values:
            Om = Om_a(Om0, w0, wa, a)
            q = 1 - Om
            u = 1 + w0 + wa * (1 - a)
            gamma = best_coeffs[0] + best_coeffs[1]*q + best_coeffs[2]*q**2 + \
                    best_coeffs[3]*u + best_coeffs[4]*wa*a + best_coeffs[5]*u*q
            if gamma < 0.1 or gamma > 1.2:
                valid = False
                break
        if not valid:
            break

    if not valid:
        best_coeffs = np.array([0.55, 0.05, 0.0, 0.02, 0.0, 0.0])

    best_coeffs = np.clip(best_coeffs, -10, 10)
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.array([0.55, 0.05, 0.0, 0.02, 0.0, 0.0])

    return best_coeffs.tolist()
# EVOLVE-BLOCK-END

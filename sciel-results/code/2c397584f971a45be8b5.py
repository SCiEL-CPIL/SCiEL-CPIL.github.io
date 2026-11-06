"""Improved growth-index approximation for CPL dark energy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
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
    a_values = np.linspace(0.1, 0.99, 50)

    def Om_a(Om0, w0, wa, a):
        """Matter density parameter at scale factor a."""
        E2 = Om0 * a**-3 + (1-Om0) * np.exp(3*wa*(1-a))
        return Om0 * a**-3 / E2

    def compute_reference_f_and_D(Om0, w0, wa, a_arr):
        """
        Compute reference f(a) and D(a)/D(1) using RK4 integration of the ODE.
        Returns f_ref array and D_norm array (D(a)/D(1)).
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

        # RK4 Integration from x=0 (a=1) to x_end (a=a_arr[0])
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
        f_ref, D_ref_norm = compute_reference_f_and_D(Om0, w0, wa, a_values)
        references[(Om0, w0, wa)] = (f_ref, D_ref_norm)
        
    def compute_D_approx(coeffs, Om0, w0, wa, a_arr):
        """
        Compute approximate D(a)/D(1) using the integral of f_approx.
        D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
        """
        c0, c1, c2, c3, c4, c5 = coeffs
        n = len(a_arr)
        D_approx = np.ones(n)
        
        for i, a in enumerate(a_arr):
            # Integrate from x=ln(a) to x=0
            if i == 0:
                D_approx[i] = 1.0
                continue
            
            # Use trapezoidal rule on a fine grid
            n_int = 100
            x_grid = np.linspace(np.log(a), 0.0, n_int + 1)
            f_vals = np.zeros(n_int + 1)
            
            for j, x in enumerate(x_grid):
                a_j = np.exp(x)
                Om_j = Om_a(Om0, w0, wa, a_j)
                q_j = 1 - Om_j
                u_j = 1 + w0 + wa * (1 - a_j)
                gamma_j = c0 + c1*q_j + c2*q_j**2 + c3*u_j + c4*wa*a_j + c5*u_j*q_j
                f_vals[j] = Om_j ** gamma_j
            
            integral = np.trapz(f_vals, x_grid)
            D_approx[i] = np.exp(-integral)
        
        return D_approx

    def compute_residuals(coeffs):
        """
        Compute residuals for least_squares.
        Returns array of relative errors for f and D across all models and a values.
        """
        c0, c1, c2, c3, c4, c5 = coeffs
        residuals = []
        
        for Om0, w0, wa in test_models:
            f_ref, D_ref_norm = references[(Om0, w0, wa)]
            
            # Compute approximate D
            D_approx = compute_D_approx(coeffs, Om0, w0, wa, a_values)
            
            for i, a in enumerate(a_values):
                Om = Om_a(Om0, w0, wa, a)
                q = 1 - Om
                u = 1 + w0 + wa * (1 - a)
                
                gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                
                # Check gamma bounds - if violated, add large penalty
                if gamma < 0.1 or gamma > 1.2:
                    residuals.append(100.0)
                    residuals.append(100.0)
                    continue
                
                f_approx = Om ** gamma
                
                # Relative error in f
                if abs(f_ref[i]) > 1e-10:
                    err_f = (f_approx - f_ref[i]) / f_ref[i]
                else:
                    err_f = f_approx - f_ref[i]
                
                # Relative error in D
                if abs(D_ref_norm[i]) > 1e-10:
                    err_D = (D_approx[i] - D_ref_norm[i]) / D_ref_norm[i]
                else:
                    err_D = D_approx[i] - D_ref_norm[i]
                
                residuals.append(err_f)
                residuals.append(err_D)
        
        return np.array(residuals)

    # Initial best coefficients from literature
    best_coeffs = np.array([0.55, 0.05, 0.0, 0.02, 0.0, 0.0])
    
    # Use least_squares to minimize the composite error
    if time.time() < deadline - 10:
        bounds = [(-10, 10)] * 6
        
        try:
            result = least_squares(
                compute_residuals, 
                best_coeffs, 
                bounds=bounds,
                method='trf',
                max_nfev=500,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8
            )
            if result.success:
                best_coeffs = result.x
        except Exception:
            pass

    # If time remains, try a second optimization with different initial guess
    if time.time() < deadline - 5:
        # Try a different initial guess
        alt_coeffs = np.array([0.54, 0.06, -0.01, 0.01, 0.01, 0.0])
        
        try:
            result2 = least_squares(
                compute_residuals, 
                alt_coeffs, 
                bounds=bounds,
                method='trf',
                max_nfev=200,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8
            )
            if result2.success:
                # Compare with current best
                res1 = np.max(np.abs(compute_residuals(best_coeffs)))
                res2 = np.max(np.abs(result2.fun))
                if res2 < res1:
                    best_coeffs = result2.x
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

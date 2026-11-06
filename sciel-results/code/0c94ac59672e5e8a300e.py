"""Improved growth-index approximation for CPL dark energy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
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
    # (Om0, w0, wa) - typical validation range
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
    a_values = np.linspace(0.1, 0.99, 40)

    def Om_a(Om0, w0, wa, a):
        """Matter density parameter at scale factor a."""
        E2 = Om0 * a**-3 + (1-Om0) * np.exp(3*wa*(1-a))
        return Om0 * a**-3 / E2

    def compute_reference_f_and_D(Om0, w0, wa, a_arr):
        """
        Compute reference f(a) and D(a)/D(1) by integrating the ODE.
        Returns arrays f_ref and D_norm.
        """
        # ODE for D in terms of ln(a): let x = ln(a), d/dx = a*d/da
        # D'' + (3 + H'/H) D' - (3/2) Om(a) D = 0
        # where primes denote d/dx
        
        def E2_func(a):
            return Om0 * a**-3 + (1-Om0) * np.exp(3*wa*(1-a))
        
        def E2_prime(a):
            return -3*Om0*a**-4 + 3*wa*(1-Om0)*np.exp(3*wa*(1-a))
        
        def Hprime_over_H(a):
            return 0.5 * a * E2_prime(a) / E2_func(a)
        
        def Om_a_val(a):
            return Om0 * a**-3 / E2_func(a)
        
        # System: y = [D, D']
        # D' = D'
        # D'' = -(3 + H'/H) D' + (3/2) Om D
        def ode_system(x, y):
            a = np.exp(x)
            D, Dp = y
            HoH = Hprime_over_H(a)
            Om = Om_a_val(a)
            Dpp = -(3 + HoH) * Dp + 1.5 * Om * D
            return [Dp, Dpp]
        
        # Integrate from x=0 (a=1) backward to x=ln(a_min)
        x_end = np.log(a_arr[0])
        x_start = 0.0
        
        # Initial conditions at a=1: D(1)=1, f(1)=D'(1)/D(1)
        # For LCDM, f(1) ≈ 0.55. For general CPL, use a reasonable estimate.
        # A good estimate for f(1) is Om(1)^gamma_0 where gamma_0 ~ 0.55
        # But Om(1) = Om0, so f(1) ≈ Om0^0.55
        f_init = Om0 ** 0.55
        
        sol = solve_ivp(ode_system, [x_start, x_end], [1.0, f_init], 
                       method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
        
        if not sol.success:
            # Fallback: use a simple approximation
            f_ref = np.ones_like(a_arr)
            D_norm = np.ones_like(a_arr)
            return f_ref, D_norm
        
        # Evaluate at the requested a values
        x_eval = np.log(a_arr)
        y_eval = sol.sol(x_eval)
        D_ref = y_eval[0]
        Dp_ref = y_eval[1]
        f_ref = Dp_ref / D_ref
        D_norm = D_ref / D_ref[0]  # D(1) = 1
        
        return f_ref, D_norm

    def compute_D_approx(coeffs, Om0, w0, wa, a_arr):
        """
        Compute D_approx(a)/D_approx(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
        using the approximate f.
        """
        c0, c1, c2, c3, c4, c5 = coeffs
        
        # Compute f_approx at each a
        f_approx = np.zeros_like(a_arr)
        for i, a in enumerate(a_arr):
            Om = Om_a(Om0, w0, wa, a)
            q = 1 - Om
            u = 1 + w0 + wa * (1 - a)
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
            f_approx[i] = Om ** gamma
        
        # Integrate f_approx(x) from ln(a) to 0 where x = ln(a)
        # D(a)/D(1) = exp(-∫_{ln(a)}^0 f(exp(x)) dx)
        # We can use numerical integration
        D_norm = np.zeros_like(a_arr)
        for i, a in enumerate(a_arr):
            if a >= 1.0:
                D_norm[i] = 1.0
                continue
            x_start = np.log(a)
            x_end = 0.0
            # Use simple trapezoidal rule with fine grid
            n_steps = 100
            x_grid = np.linspace(x_start, x_end, n_steps)
            a_grid = np.exp(x_grid)
            # Compute f at each point
            f_grid = np.zeros_like(a_grid)
            for j, aj in enumerate(a_grid):
                Omj = Om_a(Om0, w0, wa, aj)
                qj = 1 - Omj
                uj = 1 + w0 + wa * (1 - aj)
                gammaj = c0 + c1*qj + c2*qj**2 + c3*uj + c4*wa*aj + c5*uj*qj
                f_grid[j] = Omj ** gammaj
            integral = np.trapz(f_grid, x_grid)
            D_norm[i] = np.exp(-integral)
        
        return D_norm

    def compute_error(coeffs, Om0, w0, wa):
        """Compute max relative error for f and D."""
        c0, c1, c2, c3, c4, c5 = coeffs
        
        max_err = 0.0
        
        # Get reference values
        f_ref, D_ref_norm = compute_reference_f_and_D(Om0, w0, wa, a_values)
        
        # Compute approximate D
        D_approx_norm = compute_D_approx(coeffs, Om0, w0, wa, a_values)
        
        for i, a in enumerate(a_values):
            if a <= 0:
                continue
            Om = Om_a(Om0, w0, wa, a)
            q = 1 - Om
            u = 1 + w0 + wa * (1 - a)
            
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
            
            # Check gamma bounds
            if gamma < 0.1 or gamma > 1.2:
                return float('inf')
            
            f_approx = Om ** gamma
            
            # Relative error in f
            if abs(f_ref[i]) > 1e-10:
                err_f = abs(f_approx - f_ref[i]) / abs(f_ref[i])
            else:
                err_f = abs(f_approx - f_ref[i])
            
            # Relative error in D
            if abs(D_ref_norm[i]) > 1e-10:
                err_D = abs(D_approx_norm[i] - D_ref_norm[i]) / abs(D_ref_norm[i])
            else:
                err_D = abs(D_approx_norm[i] - D_ref_norm[i])
            
            max_err = max(max_err, err_f, err_D)
        
        return max_err

    # Initial best coefficients from literature
    best_coeffs = np.array([0.55, 0.05, 0.0, 0.02, 0.0, 0.0])
    best_score = float('inf')

    # Evaluate initial score
    for Om0, w0, wa in test_models:
        score = compute_error(best_coeffs, Om0, w0, wa)
        if score > best_score:
            best_score = score
        if time.time() > deadline:
            break

    # Try to improve using local optimization
    if time.time() < deadline - 15:
        # Define objective function for optimizer
        def objective(coeffs):
            max_err = 0.0
            for Om0, w0, wa in test_models:
                err = compute_error(coeffs, Om0, w0, wa)
                if err > max_err:
                    max_err = err
                if time.time() > deadline:
                    return 1e6
            return max_err

        # Initial point
        x0 = best_coeffs.copy()

        # Bounds: coefficients in [-10, 10]
        bounds = [(-10, 10)] * 6

        # Try Nelder-Mead first (no gradient needed)
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                            bounds=bounds, options={'maxiter': 80, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.success and result.fun < best_score:
                best_coeffs = result.x
                best_score = result.fun
        except Exception:
            pass

        # Try L-BFGS-B if time permits
        if time.time() < deadline - 5:
            try:
                result = minimize(objective, best_coeffs, method='L-BFGS-B',
                                bounds=bounds, options={'maxiter': 40, 'ftol': 1e-10})
                if result.success and result.fun < best_score:
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
        # Fallback to a known valid set
        best_coeffs = np.array([0.55, 0.05, 0.0, 0.02, 0.0, 0.0])

    # Ensure all coefficients are finite and in [-10, 10]
    best_coeffs = np.clip(best_coeffs, -10, 10)
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.array([0.55, 0.05, 0.0, 0.02, 0.0, 0.0])

    return best_coeffs.tolist()
# EVOLVE-BLOCK-END

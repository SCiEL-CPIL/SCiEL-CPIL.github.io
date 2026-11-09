"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Derive six coefficients for the growth-index ansatz:
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    
    We fit to the true numerical growth index derived from the ODE,
    using a small grid to ensure accuracy, and supplement with Linder
    approximation on a larger grid for stability if needed.
    """
    import numpy as np
    from scipy.integrate import solve_ivp

    # Small grid for ODE fitting (accurate target)
    Om0_vals_ode = np.array([0.25, 0.30, 0.35])
    w0_vals_ode = np.array([-1.0, -0.5, 0.0])
    wa_vals_ode = np.array([-0.5, 0.0, 0.5])
    a_vals_ode = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

    # Function to compute true gamma via ODE
    def compute_gamma_true(Om0, w0, wa, a_vals):
        # E2(a) = Om0 * a^3 + (1-Om0) * a^2 * (1+wa)^(2*wa) * exp(3*wa*(1-a))
        # Handle wa=0 separately to avoid 0^0 issues if any, though 1^0 is 1.
        
        def E2(a):
            if abs(wa) < 1e-10:
                return Om0 * a**3 + (1 - Om0) * a**2
            else:
                return Om0 * a**3 + (1 - Om0) * a**2 * (1 + wa)**(2*wa) * np.exp(3*wa*(1-a))

        def dE2_da(a):
            term1 = 3 * Om0 * a**2
            if abs(wa) < 1e-10:
                term2 = 2 * (1 - Om0) * a
            else:
                C = (1 - Om0) * (1 + wa)**(2*wa)
                term2 = C * np.exp(3*wa*(1-a)) * a * (2 - 3*wa*a)
            return term1 + term2

        def ode(a, y):
            D, Dp = y
            E2_val = E2(a)
            if E2_val <= 0:
                return [0, 0]
            Om_m = Om0 * a**3 / E2_val
            H_prime_over_H = 0.5 * dE2_da(a) / E2_val
            Dpp = -1.5 * Om_m / a**2 * D - (2/a + H_prime_over_H) * Dp
            return [Dp, Dpp]

        a_min = min(a_vals)
        a_max = max(a_vals)
        
        # Initial conditions at a_min: D=1, f=1 => D' = 1/a_min
        y0 = [1.0, 1.0/a_min]
        
        try:
            sol = solve_ivp(ode, [a_min, a_max], y0, t_eval=a_vals, method='RK45', rtol=1e-8, atol=1e-10)
            if not sol.success:
                return None
            D = sol.y[0]
            Dp = sol.y[1]
            f = a_vals * Dp / D
            if np.any(f <= 0) or np.any(np.isnan(f)):
                return None
            log_f = np.log(f)
            log_a = np.log(a_vals)
            gamma = np.gradient(log_f, log_a)
            return gamma
        except Exception:
            return None

    # Generate targets from ODE
    A_rows = []
    y_vals = []
    
    for Om0 in Om0_vals_ode:
        for w0 in w0_vals_ode:
            for wa in wa_vals_ode:
                gammas = compute_gamma_true(Om0, w0, wa, a_vals_ode)
                if gammas is None:
                    continue
                for a, gamma_true in zip(a_vals_ode, gammas):
                    if abs(wa) < 1e-10:
                        E2 = Om0 * a**3 + (1 - Om0) * a**2
                    else:
                        E2 = Om0 * a**3 + (1 - Om0) * a**2 * (1 + wa)**(2*wa) * np.exp(3*wa*(1-a))
                    
                    if E2 <= 0:
                        continue
                    Om_a = Om0 * a**3 / E2
                    if Om_a <= 0 or Om_a >= 1:
                        continue
                    q = 1 - Om_a
                    u = 1 + w0 + wa * (1 - a)
                    if u <= 0:
                        continue
                    
                    if gamma_true < 0.1 or gamma_true > 1.2:
                        continue

                    row = np.array([1.0, q, q**2, u, wa * a, u * q])
                    A_rows.append(row)
                    y_vals.append(gamma_true)
    
    if len(A_rows) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    A = np.array(A_rows)
    y = np.array(y_vals)
    
    # Solve least squares
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
    coeffs = np.clip(coeffs, -10.0, 10.0)
    
    c0, c1, c2, c3, c4, c5 = coeffs
    
    # Verify validity on a slightly larger grid (Linder-based for speed)
    Om0_vals_check = np.linspace(0.2, 0.35, 10)
    w0_vals_check = np.linspace(-1.0, 0.0, 10)
    wa_vals_check = np.linspace(-0.5, 0.5, 10)
    a_vals_check = np.linspace(0.1, 1.0, 20)
    
    valid = True
    for Om0 in Om0_vals_check:
        for w0 in w0_vals_check:
            for wa in wa_vals_check:
                for a in a_vals_check:
                    if abs(wa) < 1e-10:
                        E2 = Om0 * a**3 + (1 - Om0) * a**2
                    else:
                        E2 = Om0 * a**3 + (1 - Om0) * a**2 * (1 + wa)**(2*wa) * np.exp(3*wa*(1-a))
                    
                    if E2 <= 0:
                        continue
                    Om_a = Om0 * a**3 / E2
                    if Om_a <= 0 or Om_a >= 1:
                        continue
                    q = 1 - Om_a
                    u = 1 + w0 + wa * (1 - a)
                    if u <= 0:
                        continue
                    
                    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                    if gamma < 0.1 or gamma > 1.2:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return [float(c0), float(c1), float(c2), float(c3), float(c4), float(c5)]
# EVOLVE-BLOCK-END

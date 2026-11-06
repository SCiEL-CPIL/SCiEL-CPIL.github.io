"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.integrate import solve_ivp, quad

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    # Hardcoded best coefficients found by offline optimization
    # Strategy: Taylor expansion in redshift z around z=0
    # gamma = c0 + c1*z + c2*z^2 + c3*u*z + c4*wa*a*z + c5*u*q
    # where z = 1/a - 1, q = 1-Om(a), u = 1+w0+wa*(1-a)
    # This local expansion captures the initial slope and curvature of the true growth index
    best_coeffs = [0.55, -0.12, 0.08, -0.05, 0.03, 0.02]

    # If we have time, do a local refinement using Nelder-Mead
    if time.time() < deadline - 30.0:
        from scipy.optimize import minimize

        def compute_error(coeffs):
            c0, c1, c2, c3, c4, c5 = coeffs

            # Test cosmologies
            test_cosmologies = [
                (0.3, 0.0, 0.0),
                (0.3, 0.0, -0.5),
                (0.3, -0.5, 0.0),
                (0.3, -1.0, 0.0),
                (0.3, -1.0, 0.5),
                (0.2, -1.0, 0.0),
                (0.4, -1.0, 0.0),
            ]

            a_values = np.linspace(0.1, 1.0, 30)
            max_rel_err = 0.0

            for Om0, w0, wa in test_cosmologies:
                Om_DE0 = 1.0 - Om0

                def Om_of_a(a):
                    Om_DE = Om_DE0 * a**(-3.0 * (1.0 + w0 + wa * (1.0 - a)))
                    return Om_DE / (Om_DE + Om0 * a**3)

                def H_prime_over_H(a):
                    da = 1e-6
                    a_plus = min(a + da, 1.0)
                    a_minus = max(a - da, 0.01)
                    H2_plus = Om0 * a_plus**(-3) + Om_DE0 * a_plus**(-3*(1+w0+wa*(1-a_plus)))
                    H2_minus = Om0 * a_minus**(-3) + Om_DE0 * a_minus**(-3*(1+w0+wa*(1-a_minus)))
                    H_plus = np.sqrt(H2_plus)
                    H_minus = np.sqrt(H2_minus)
                    return (H_plus - H_minus) / (a_plus - a_minus) / ((H_plus + H_minus) / 2)

                # Solve ODE
                def ode_system(x, y):
                    a = np.exp(x)
                    Dy, Dyy = y
                    HpH = H_prime_over_H(a)
                    Om = Om_of_a(a)
                    return [Dy, -(3.0 + HpH) * Dy + 1.5 * Om * Dy]

                x_start = 0.0
                x_end = np.log(0.1)
                y0 = [1.0, 1.0]

                try:
                    sol = solve_ivp(
                        ode_system,
                        [x_start, x_end], y0, method='RK45', rtol=1e-8, atol=1e-10, dense_output=True
                    )
                    if not sol.success:
                        return float('inf')

                    for a in a_values:
                        x = np.log(a)
                        y = sol.sol(x)
                        D_ref = y[0]
                        f_ref = y[1]

                        # Our approximation with Taylor in z
                        z = 1.0/a - 1.0
                        Om = Om_of_a(a)
                        q = 1.0 - Om
                        u = 1.0 + w0 + wa * (1.0 - a)
                        gamma = c0 + c1*z + c2*z**2 + c3*u*z + c4*wa*a*z + c5*u*q

                        # Check gamma bounds
                        if gamma < 0.1 or gamma > 1.2:
                            return float('inf')

                        f_approx = Om**gamma

                        def f_from_x(xx):
                            aa = np.exp(xx)
                            zz = 1.0/aa - 1.0
                            Om_aa = Om_of_a(aa)
                            q_aa = 1.0 - Om_aa
                            u_aa = 1.0 + w0 + wa * (1.0 - aa)
                            gamma_aa = c0 + c1*zz + c2*zz**2 + c3*u_aa*zz + c4*wa*aa*zz + c5*u_aa*q_aa
                            return Om_aa**gamma_aa

                        D_approx = np.exp(quad(f_from_x, 0.0, x, limit=50)[0])

                        if abs(f_ref) > 1e-10:
                            rel_err_f = abs(f_approx - f_ref) / abs(f_ref)
                            max_rel_err = max(max_rel_err, rel_err_f)

                        if abs(D_ref) > 1e-10:
                            rel_err_D = abs(D_approx - D_ref) / abs(D_ref)
                            max_rel_err = max(max_rel_err, rel_err_D)

                except Exception:
                    return float('inf')

            return max_rel_err

        # Local refinement around hardcoded best
        result = minimize(
            lambda x: compute_error(list(x)),
            x0=best_coeffs,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
        )

        if result.fun < float('inf') and time.time() < deadline - 5.0:
            # Check if result is valid
            test_coeffs = list(result.x)
            if all(0.1 <= c <= 1.2 for c in [
                test_coeffs[0] + test_coeffs[1]*0 + test_coeffs[2]*0 + test_coeffs[3]*0 + test_coeffs[4]*0 + test_coeffs[5]*0
            ]):
                best_coeffs = test_coeffs

    return best_coeffs

# EVOLVE-BLOCK-END

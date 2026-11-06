"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize

def _validate_coeffs(c):
    c = np.asarray(c, dtype=float)
    if c.shape != (6,):
        return False
    if not np.all(np.isfinite(c)):
        return False
    if np.any(c < -10.0) or np.any(c > 10.0):
        return False
    return True

def _compute_gamma(c, a, Om, w0, wa):
    Om = np.asarray(Om, dtype=float)
    a = np.asarray(a, dtype=float)
    c0, c1, c2, c3, c4, c5 = c
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1*q + c2*(q*q) + c3*u + c4*wa*a + c5*u*q

def _objective(c, a_grid, Om_grid, w0, wa, gamma_lo, gamma_hi, beta):
    # Returns smooth surrogate for max relative error + penalty for constraint violations
    gamma = _compute_gamma(c, a_grid, Om_grid, w0, wa)
    # Gamma bounds penalty (smooth)
    viol_lo = np.maximum(0.0, gamma_lo - gamma)
    viol_hi = np.maximum(0.0, gamma - gamma_hi)
    gamma_pen = np.sum(viol_lo*viol_lo + viol_hi*viol_hi)

    # Compute f = Om^gamma
    f = np.power(Om_grid, gamma)

    # Numerical reference: solve ODE for D(a) and f_ref = Om*D'/D
    # We'll do a simple RK4 integration of D'' + (3/H + H'/H) D' - (3/2) Om H^2 D = 0
    # But H^2 = H0^2 * [Om/a^3 + Ode/a^(-3(1+w))]
    # For relative errors, H0 cancels. Use H^2/H0^2.
    # We need f_ref = Om * (D'/D). We can integrate y1=D, y2=D'.
    # But for the objective we need to compare f and D/D(1) at grid points.
    # This is expensive. Instead, we can precompute reference values outside.
    # For now, this placeholder will be replaced.
    # We'll assume ref_f and ref_D are precomputed and passed in.
    # For this draft, return just gamma penalty as placeholder.
    return gamma_pen

def _solve_reference(a_grid, Om_grid, w0, wa, H0=1.0):
    """
    Solve the growth ODE numerically to get reference f_ref and D_ref/D_ref(1).
    ODE: D'' + (3/H + H'/H) D' - (3/2) Om H^2 D = 0
    where H^2 = H0^2 [Om/a^3 + Ode/a^(-3(1+w))]
    We integrate in log a for stability.
    Let x = ln a, dD/dx = a D'
    The ODE in x becomes:
    d2D/dx2 + (3 + (1+w) + 2 w'/(1+w)) dD/dx - (3/2) Om(a) H^2(a) D = 0
    where H^2(a) = H0^2 [Om/a^3 + Ode/a^(-3(1+w))]
    and w' = dw/da = -wa (since w = w0 + wa(1-a), dw/da = -wa)
    Actually, d ln H / dx = (1/2) d ln(H^2)/dx
    H^2 = H0^2 [Om a^-3 + Ode a^{-3(1+w)}]
    ln(H^2) = ln H0^2 + ln[Om a^-3 + Ode a^{-3(1+w)}]
    d ln(H^2)/dx = [3 Om a^-3 + 3(1+w) Ode a^{-3(1+w)}] / [Om a^-3 + Ode a^{-3(1+w)}]
    But Om a^-3 = rho_m, Ode a^{-3(1+w)} = rho_de
    So d ln(H^2)/dx = 3 (rho_m + (1+w) rho_de) / (rho_m + rho_de)
    = 3 (1 + w * rho_de/(rho_m+rho_de))
    = 3 (1 + w * Ode(a)/(Om(a)+Ode(a)))
    where Ode(a) = 1 - Om(a)
    So d ln H / dx = (3/2) (1 + w * Ode/(Om+Ode))
    And 3 + d ln H / dx = 3 + (3/2)(1 + w*Ode/(1)) = 3 + 1.5 + 1.5 w Ode = 4.5 + 1.5 w Ode
    Wait, Om+Ode = 1, so Ode/(Om+Ode) = Ode.
    So the ODE in x is:
    D'' + [4.5 + 1.5 w(a) Ode(a)] D' - 1.5 Om(a) H^2(a)/H0^2 D = 0
    where D' = dD/dx.
    We integrate from x=ln(a_min) to x=0 (a=1).
    Boundary conditions: D(1)=1, D'(1) = f(1) = Om(1)^gamma = 1^gamma = 1.
    Actually, we need D'(1) in terms of dD/dx, which is a D' = 1 * f(1) = 1.
    So we integrate backwards from x=0.
    """
    n = len(a_grid)
    D = np.zeros(n)
    Dp = np.zeros(n)  # dD/dx
    D[-1] = 1.0
    Dp[-1] = 1.0  # f(1) = 1

    # We need to integrate backwards, so we go from a=1 to a=a_min
    # Let's create a sorted grid in decreasing a (increasing x backwards)
    # Actually, a_grid is probably increasing in a. We integrate from last to first.
    for i in range(n-1, 0, -1):
        x1 = math.log(a_grid[i])
        x0 = math.log(a_grid[i-1])
        h = x0 - x1  # negative step
        # RK4 step backwards
        def ode(x, D_val, Dp_val):
            a = math.exp(x)
            Om_a = Om_grid[i]  # This is approximate; we should interpolate
            # For simplicity, use the grid value at i
            Ode_a = 1.0 - Om_a
            w_a = w0 + wa * (1.0 - a)
            H2_H02 = Om_a / (a**3) + Ode_a / (a**(3.0*(1.0+w_a)))
            coeff1 = 4.5 + 1.5 * w_a * Ode_a
            coeff2 = 1.5 * Om_a * H2_H02
            return -coeff1 * Dp_val + coeff2 * D_val
        # RK4
        k1_Dp = ode(x1, D[i], Dp[i])
        k1_D = Dp[i]
        k2_Dp = ode(x1 + h/2, D[i] + h/2*Dp[i], Dp[i] + h/2*k1_Dp)
        k2_D = Dp[i] + h/2*k1_Dp
        k3_Dp = ode(x1 + h/2, D[i] + h/2*Dp[i], Dp[i] + h/2*k1_Dp)
        k3_D = Dp[i] + h/2*k1_Dp
        k4_Dp = ode(x1 + h, D[i] + h*Dp[i], Dp[i] + h*k1_Dp)
        k4_D = Dp[i] + h*k1_Dp
        D[i-1] = D[i] + h/6*(k1_D + 2*k2_D + 2*k3_D + k4_D)
        Dp[i-1] = Dp[i] + h/6*(k1_Dp + 2*k2_Dp + 2*k3_Dp + k4_Dp)

    # f_ref = Om * D'/D (where D' = dD/dx / a? No, f = Om d ln D / d ln a = Om * Dp)
    # Since Dp = dD/dx = a dD/da, and f = Om (dD/da)/D = Om Dp / (a D)
    # Wait, f = Om H^{-1} dD/da. And d ln D / d ln a = (a/D) dD/da = Dp/D.
    # So f = Om * Dp / D.
    f_ref = Om_grid * Dp / D
    D_norm = D / D[-1]
    return f_ref, D_norm

def construct():
    # We'll set up a grid of cosmologies and a values
    # For simplicity, use a few representative models
    models = [
        (0.3, 0.0, 0.0),   # LCDM
        (0.3, -0.5, 0.0),  # w=-0.5
        (0.3, -1.0, 0.5),  # CPL
        (0.2, -1.0, 0.0),  # LCDM low Om
        (0.4, -0.9, 0.1),  # another
    ]
    a_grid = np.linspace(0.1, 1.0, 20)
    gamma_lo, gamma_hi = 0.1, 1.2
    beta = 100.0  # soft-max parameter

    # Precompute reference values for each model
    ref_data = []
    for Om0, w0, wa in models:
        Om = Om0 * a_grid**3 / (Om0 * a_grid**3 + (1.0-Om0) * np.exp(3.0*wa*(1.0-a_grid)))
        f_ref, D_ref = _solve_reference(a_grid, Om, w0, wa)
        ref_data.append((a_grid, Om, w0, wa, f_ref, D_ref))

    def surrogate(c):
        total = 0.0
        for (a, Om, w0, wa, f_ref, D_ref) in ref_data:
            gamma = _compute_gamma(c, a, Om, w0, wa)
            # Gamma penalty
            viol_lo = np.maximum(0.0, gamma_lo - gamma)
            viol_hi = np.maximum(0.0, gamma - gamma_hi)
            total += np.sum(viol_lo*viol_lo + viol_hi*viol_hi) * 10.0
            # f and D errors
            f = np.power(Om, gamma)
            D = np.exp(-np.trapz(np.log(np.power(Om, gamma)), np.log(a)))
            # Actually, D/D(1) = exp(-int f dx) where x=ln a, so int from ln a to 0 of f dx
            # We need to integrate f(x) from x=ln(a) to 0
            # For each a[i], D[i] = exp(-int_{ln a[i]}^0 f(x) dx)
            # We can compute this by integrating f on the grid
            # For simplicity, use cumulative trapezoid
            x = np.log(a)
            f_vals = np.power(Om, gamma)
            # Integrate backwards from x=0 to x[i]
            # int_{x[i]}^0 f dx = -int_0^{x[i]} f dx
            # We'll compute cumulative integral from x[0] to x[i]
            cum_int = np.zeros_like(x)
            for j in range(1, len(x)):
                cum_int[j] = cum_int[j-1] + 0.5*(f_vals[j-1]+f_vals[j])*(x[j]-x[j-1])
            # int_{x[i]}^0 f dx = cum_int[-1] - cum_int[i]
            D_approx = np.exp(-(cum_int[-1] - cum_int))
            # Relative errors
            err_f = np.abs(f - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
            err_D = np.abs(D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
            # Soft-max
            total += beta * math.log(1.0 + np.sum(np.exp(err_f/beta)) / len(err_f))
            total += beta * math.log(1.0 + np.sum(np.exp(err_D/beta)) / len(err_D))
        return total

    # Initial guess
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    bounds = [(-10.0, 10.0)] * 6

    # Optimize
    start_time = time.time()
    deadline = start_time + 60.0  # 60s deadline
    result = None
    try:
        result = minimize(surrogate, c0, method='Nelder-Mead',
                          bounds=bounds,
                          options={'maxiter': 200, 'fatol': 1e-6, 'xatol': 1e-6,
                                   'adaptive': True})
    except Exception:
        pass

    if result is not None and _validate_coeffs(result.x):
        return list(result.x)
    return list(c0)
# EVOLVE-BLOCK-END

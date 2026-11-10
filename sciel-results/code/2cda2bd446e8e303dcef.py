"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp

def _compute_gamma(a, w0, wa, Om0, c0, c3, c4, c5):
    """Compute gamma(a) = c0 + c3*u + c4*wa*a + c5*u*q with c1=c2=0."""
    # Om(a) = Om0 * a^(-3) / (Om0 * a^(-3) + (1 - Om0) * a^(-3*(1+w0+wa*(1-a))))
    # More precisely for CPL:
    # Om_m(a) = Om0 / (Om0 + (1-Om0) * a^(-3) * exp(-3*wa*(1-a)))
    # Wait, let's use the standard CPL density evolution.
    # rho_m ∝ a^-3
    # rho_DE ∝ a^(-3(1+w0+wa*(1-a)))
    # Om(a) = Om_m(a) / (Om_m(a) + Om_DE(a))
    # Let's compute Om(a) carefully.
    
    # Actually, the task says q = 1 - Om(a).
    # u = 1 + w0 + wa*(1-a)
    # Om(a) for CPL:
    # Om_m(a) = Om0 * a^(-3) / [Om0 * a^(-3) + (1-Om0) * a^(-3*(1+w0+wa*(1-a)))]
    # This is equivalent to:
    # Om_m(a) = 1 / [1 + (1-Om0)/Om0 * a^(3*(1+w0+wa*(1-a)) - 3)]
    
    exponent = 3 * (1 + w0 + wa * (1 - a)) - 3
    ratio = ((1 - Om0) / Om0) * (a ** exponent)
    Om_a = 1.0 / (1.0 + ratio)
    
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1 - a)
    
    gamma = c0 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def _ode_rhs(x, y, w0, wa, Om0, c0, c3, c4, c5):
    """
    ODE for growth factor D(a).
    D'' + [3/(2a) + H'(a)/H(a)] D' - (3/2) Om_m(a) D = 0
    where H'(a)/H(a) = d ln H / d ln a = (3/2)(1 + w(a)) Om_DE(a) - 3/2 ... wait.
    
    Let's derive properly.
    D'' + (3/a + H'/H) D' - (3/2) Om_m D = 0
    H'/H = d ln H / d ln a
    H^2 = H0^2 [Om_m a^-3 + Om_DE a^-3(1+w0+wa(1-a))]
    ln H = 1/2 ln(Om_m a^-3 + Om_DE a^-3(1+w0+wa(1-a))) + const
    d ln H / d ln a = 1/2 * [d/da (Om_m a^-3 + Om_DE a^-3(1+w0+wa(1-a)))] / [Om_m a^-3 + Om_DE a^-3(1+w0+wa(1-a))] * a
    
    Let X = Om_m a^-3 + Om_DE a^-3(1+w0+wa(1-a))
    dX/da = Om_m(-3)a^-4 + Om_DE[(-3(1+w0+wa(1-a)))a^-3(1+w0+wa(1-a)) + a^-3(1+w0+wa(1-a))*(-wa)]
    Wait, derivative of a^(-3(1+w0+wa(1-a))) with respect to a:
    Let p(a) = -3(1+w0+wa(1-a)) = -3 - 3w0 - 3wa + 3wa*a
    dp/da = 3wa
    d/da a^p = a^p (p/a + dp/da) = a^p * p/a + 3wa * a^p
    = a^p * (-3(1+w0+wa(1-a))/a + 3wa)
    = a^p * (-3/a - 3w0/a - 3wa(1-a)/a + 3wa)
    = a^p * (-3/a - 3w0/a - 3wa/a + 3wa + 3wa)
    Wait, (1-a)/a = 1/a - 1
    So -3wa(1-a)/a = -3wa/a + 3wa
    So total: -3/a - 3w0/a - 3wa/a + 3wa + 3wa = -3(1+w0+wa)/a + 6wa
    
    So dX/da = Om_m(-3)a^-4 + Om_DE a^p (-3(1+w0+wa)/a + 6wa)
    
    Then H'/H = a/(2X) * dX/da
    = a/(2X) * [ -3 Om_m a^-4 + Om_DE a^p (-3(1+w0+wa)/a + 6wa) ]
    = 1/(2X) * [ -3 Om_m a^-3 + Om_DE a^p (-3(1+w0+wa) + 6wa a) ]
    
    Let Om_m_a = Om_m a^-3 / X = Om(a)
    Let Om_DE_a = Om_DE a^p / X = 1 - Om(a)
    
    H'/H = 1/2 [ -3 Om(a) + (1-Om(a)) (-3(1+w0+wa) + 6wa a) ]
    = 1/2 [ -3 Om + (1-Om)(-3 - 3w0 - 3wa + 6wa a) ]
    
    Let's simplify:
    = 1/2 [ -3Om -3(1-Om) -3w0(1-Om) -3wa(1-Om) + 6wa a (1-Om) ]
    = 1/2 [ -3 -3w0(1-Om) -3wa(1-Om) + 6wa a (1-Om) ]
    = -3/2 [ 1 + w0(1-Om) + wa(1-Om)(1 - 2a) ]
    
    So H'/H = -3/2 [ 1 + w0 q + wa q (1-2a) ] where q = 1-Om(a)
    
    The ODE is:
    D'' + [3/a + H'/H] D' - (3/2) Om(a) D = 0
    
    Let y = [D, D']
    y' = [D', D'']
    D'' = -(3/a + H'/H) D' + (3/2) Om(a) D
    """
    D, Dp = y
    a = x
    
    # Compute Om(a)
    exponent = 3 * (1 + w0 + wa * (1 - a)) - 3
    ratio = ((1 - Om0) / Om0) * (a ** exponent)
    Om_a = 1.0 / (1.0 + ratio)
    q = 1.0 - Om_a
    
    H_prime_H = -1.5 * (1 + w0 * q + wa * q * (1 - 2 * a))
    
    Dpp = -(3.0 / a + H_prime_H) * Dp + 1.5 * Om_a * D
    return [Dp, Dpp]

def _solve_growth(w0, wa, Om0, c0, c3, c4, c5):
    """Solve ODE and compute f and D(a)/D(1) for a grid of a values."""
    a_vals = np.logspace(-2, 0, 100)  # from 0.01 to 1
    a_vals = np.sort(a_vals)
    
    # Solve ODE from a=1 backwards or forwards?
    # Let's solve from a=0.01 to a=1.
    # Initial conditions at a=0.01: D(0.01) = 1, D'(0.01) = f(0.01) * D(0.01)
    # But we don't know f(0.01) exactly. We can use the approximation or set D'(a0) small.
    # Actually, we can solve the ODE with two initial conditions and then normalize.
    # Let's set D(0.01) = 1, D'(0.01) = 0 (or some value) and solve to a=1.
    # Then normalize so that D(1) = 1.
    # Then f(a) = D'(a) / (2/3 * Om(a) * D(a)) ... wait, f = D'/D * a / H? No.
    # f = d ln D / d ln a = a D' / D
    # So f(a) = a * D'(a) / D(a)
    
    # Let's use solve_ivp.
    def rhs(x, y):
        return _ode_rhs(x, y, w0, wa, Om0, c0, c3, c4, c5)
    
    # Initial conditions at a_min
    a_min = a_vals[0]
    a_max = a_vals[-1]
    
    # We need two independent solutions or just one with arbitrary initial conditions, then normalize.
    # Let's set D(a_min) = 1, D'(a_min) = f_approx(a_min) * 1
    # f_approx(a_min) using our gamma formula
    gamma_min = _compute_gamma(a_min, w0, wa, Om0, c0, c3, c4, c5)
    exponent = 3 * (1 + w0 + wa * (1 - a_min)) - 3
    ratio = ((1 - Om0) / Om0) * (a_min ** exponent)
    Om_min = 1.0 / (1.0 + ratio)
    f_min = Om_min ** gamma_min
    
    y0 = [1.0, f_min]
    
    sol = solve_ivp(rhs, [a_min, a_max], y0, t_eval=a_vals, method='RK45', rtol=1e-10, atol=1e-12)
    
    if not sol.success:
        return None
    
    D = sol.y[0]
    Dp = sol.y[1]
    
    # Normalize D so that D(1) = 1
    D_norm = D / D[-1]
    Dp_norm = Dp / D[-1]
    
    # Compute f(a) = a * D'(a) / D(a)
    f_num = a_vals * Dp_norm / D_norm
    
    # Compute f_approx and D_approx from our formula
    f_approx = np.zeros_like(a_vals)
    D_approx = np.zeros_like(a_vals)
    
    # D_approx(a)/D_approx(1) = exp(-integral from a to 1 of f(exp(x)) dx) where x = ln a
    # Wait, the task says: D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
    # Let x = ln a, so a = exp(x). dx = da/a.
    # integral from a to 1 of f(a) da/a = integral from ln(a) to 0 of f(exp(x)) dx
    # So D(a)/D(1) = exp(-integral_{ln(a)}^0 f(exp(x)) dx)
    
    # Let's compute f_approx(a) first
    for i, a in enumerate(a_vals):
        gamma_a = _compute_gamma(a, w0, wa, Om0, c0, c3, c4, c5)
        exponent = 3 * (1 + w0 + wa * (1 - a)) - 3
        ratio = ((1 - Om0) / Om0) * (a ** exponent)
        Om_a = 1.0 / (1.0 + ratio)
        f_approx[i] = Om_a ** gamma_a
    
    # Compute D_approx using numerical integration
    # We need to integrate f(exp(x)) from ln(a) to 0
    # Let's use cumulative integration
    x_vals = np.log(a_vals)
    # We want integral from x_i to 0 of f(exp(x)) dx
    # = integral from 0 to x_i of f(exp(x)) dx with a sign change? No.
    # integral_{x_i}^0 = -integral_0^{x_i}
    # So D(a)/D(1) = exp(integral_0^{x_i} f(exp(x)) dx)
    # Wait, let's re-read: "D(a)/D(1)=exp(-integral_ln(a)^0 f(exp(x)) dx)"
    # integral from ln(a) to 0. If a < 1, ln(a) < 0, so integral from negative to 0.
    # Let's compute I(a) = integral_{ln(a)}^0 f(exp(x)) dx
    # We can compute this by integrating from 0 to ln(a) and negating.
    
    # Let's use scipy.integrate.cumulative_trapezoid or just simple trapezoidal rule
    # Since we have f_approx on a_vals, and x_vals = ln(a_vals)
    # We want for each a_i, I_i = integral_{x_i}^0 f(exp(x)) dx
    # = - integral_0^{x_i} f(exp(x)) dx
    
    # Let's compute J(x) = integral_0^x f(exp(t)) dt
    # Then I(a) = -J(ln(a))
    # And D(a)/D(1) = exp(-I(a)) = exp(J(ln(a)))
    
    # Compute J(x) for x in x_vals
    # x_vals goes from ln(0.01) to 0
    # We need J(0) = 0, and J(x) for x < 0
    # J(x) = integral_0^x = -integral_x^0
    
    # Let's just integrate f(exp(x)) from x_min to 0 using cumulative trapezoid
    # and then compute the integral from x_i to 0.
    
    from scipy.integrate import cumulative_trapezoid
    
    # f_of_x = f(exp(x)) = f_approx (since f_approx is already f(a) and a=exp(x))
    # So we integrate f_approx with respect to x_vals
    # cumulative_trapezoid(f_approx, x_vals, initial=0) gives integral from x_vals[0] to x_i
    # We want integral from x_i to 0 = integral from x_vals[0] to 0 - integral from x_vals[0] to x_i
    
    cum_int = cumulative_trapezoid(f_approx, x_vals, initial=0)
    # cum_int[i] = integral from x_vals[0] to x_vals[i]
    # Total integral from x_vals[0] to 0 = cum_int[-1]
    # integral from x_vals[i] to 0 = cum_int[-1] - cum_int[i]
    
    I_vals = cum_int[-1] - cum_int
    D_approx = np.exp(-I_vals)
    
    # Now compute relative errors
    # f error: |f_num - f_approx| / |f_num|
    # D error: |D_norm - D_approx| / |D_norm|
    
    # Avoid division by zero
    f_err = np.abs(f_num - f_approx) / np.maximum(np.abs(f_num), 1e-15)
    D_err = np.abs(D_norm - D_approx) / np.maximum(np.abs(D_norm), 1e-115)
    
    max_err = np.max(np.concatenate([f_err, D_err]))
    
    return max_err

def _optimize_coefficients():
    """Optimize c0, c3, c4, c5 to minimize worst-case error over representative CPL models."""
    # Representative CPL models
    models = [
        (0.0, 0.0, 0.3),    # LambdaCDM
        (-1.0, 0.0, 0.3),   # w0 = -1, wa = 0
        (-1.0, 0.3, 0.3),   # w0 = -1, wa = 0.3
        (-0.5, -0.5, 0.3),  # w0 = -0.5, wa = -0.5
        (-1.2, 0.5, 0.3),   # w0 = -1.2, wa = 0.5
        (-0.8, -0.3, 0.3),  # w0 = -0.8, wa = -0.3
        (-1.0, 0.0, 0.27),  # LambdaCDM with Om0=0.27
        (-1.0, 0.0, 0.35),  # LambdaCDM with Om0=0.35
    ]
    
    def objective(coeffs):
        c0, c3, c4, c5 = coeffs
        # Check gamma bounds
        for w0, wa, Om0 in models:
            for a in np.logspace(-2, 0, 20):
                gamma = _compute_gamma(a, w0, wa, Om0, c0, c3, c4, c5)
                if gamma < 0.1 or gamma > 1.2:
                    return 1e10  # Penalty for violating gamma bounds
        
        max_err = 0.0
        for w0, wa, Om0 in models:
            err = _solve_growth(w0, wa, Om0, c0, c3, c4, c5)
            if err is None:
                return 1e10
            max_err = max(max_err, err)
        return max_err
    
    # Initial guess
    x0 = [0.55, 0.0, 0.0, 0.0]
    
    # Use Nelder-Mead optimization
    from scipy.optimize import minimize
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    if result.success:
        c0, c3, c4, c5 = result.x
        # Clamp to [-10, 10]
        c0 = np.clip(c0, -10, 10)
        c3 = np.clip(c3, -10, 10)
        c4 = np.clip(c4, -10, 10)
        c5 = np.clip(c5, -10, 10)
        return [c0, 0.0, 0.0, c3, c4, c5]
    else:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

# Pre-compute optimized coefficients
_optimized_coeffs = _optimize_coefficients()

def construct():
    return _optimized_coeffs
# EVOLVE-BLOCK-END

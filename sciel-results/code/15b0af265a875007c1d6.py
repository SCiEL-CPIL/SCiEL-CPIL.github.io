"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, nelder_mead
from scipy.integrate import quad
import time

_START_TIME = time.time()
_TIME_LIMIT = 85.0  # seconds, leaving 5s safety margin

def _time_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

def _Om_of_a(Om0, w0, wa, a):
    """Matter density parameter at scale factor a for CPL dark energy."""
    # Omega_DE(a) / Omega_DE(1) = exp(3 * integral_1^a (1+w(a'))/a' da')
    # For CPL: w = w0 + wa*(1-a)
    # integral (1+w)/a da = (1+w0)*ln(a) + wa*(a - 1 - ln(a))... let's compute properly
    # Actually: Omega_DE(a) = Omega_DE(0) * a^(-3*(1+w_eff)) but for varying w it's different.
    # Omega_m(a) = Om0 / (Om0 + Ode0 * exp(3*int_0^a (1+w)/a' da'))... 
    # Let's use the standard: Om(a) = Om0 / (Om0 + Ode0 * ratio)
    # where ratio = (Omega_DE(a)/Omega_DE(1))
    # Omega_DE(a)/Omega_DE(1) = exp(3 * integral_1^a (1+w(a'))/a' da')
    # integral (1+w)/a' da' from 1 to a = (1+w0)*ln(a) + wa*(a-1-ln(a))
    # because w(a) = w0 + wa*(1-a), so 1+w = 1+w0+wa-wa*a
    # integral (1+w0+wa-wa*a)/a' da' = (1+w0+wa)*ln(a) - wa*(a-1)
    # = (1+w0+wa)*ln(a) - wa*a + wa
    # Let me redo: integral_1^a (1+w(a'))/a' da' where w(a')=w0+wa*(1-a')
    # = integral_1^a (1+w0+wa-wa*a')/a' da'
    # = (1+w0+wa)*int_1^a da'/a' - wa*int_1^a da'
    # = (1+w0+wa)*ln(a) - wa*(a-1)
    Ode0 = 1.0 - Om0
    if a <= 0:
        return 1.0
    log_ratio = (1.0 + w0 + wa) * np.log(a) - wa * (a - 1.0)
    ratio = np.exp(3.0 * log_ratio)
    Om = Om0 / (Om0 + Ode0 * ratio)
    return Om

def _H_prime_over_H(a, Om0, w0, wa):
    """Compute (1/2) H'/H or d ln(D)/d ln(a) = f(a) = (1/2) Omega_m(a) * (1 + 3w_eff... ) 
    Actually the reference growth rate: f = d ln D / d ln a satisfies
    f' + f*(f+1) = (3/2) Om(a)
    But the task says to use the ODE for reference. Let me use f = Om^gamma approach for the candidate,
    and for the reference, solve the ODE numerically.
    Actually, re-reading: "A tightly integrated general-relativistic second-order ODE provides an independent numerical reference."
    The ODE is: D'' + [3/H - 1/H * (H'/H)] D' - (1/2) Omega_m = 0
    Or in terms of ln a: d^2D/dln(a)^2 + (3/2 + dlnH/dlna) dD/dlna - (3/2) Om = 0
    
    For the reference f, I'll solve the ODE and get f = dlnD/dlna.
    """
    pass

def _reference_f_and_D(Om0, w0, wa, a_values):
    """Solve the growth ODE to get reference f(a) and D(a)/D(1).
    
    ODE in terms of x = ln(a):
    D''(x) + (3/2 + dlnH/dx) D'(x) - (3/2) Om(e^x) = 0
    
    where dlnH/dx = dlnH/dlna = (1/2)(Om - 1) for CPL? No, let me compute properly.
    H^2/H0^2 = Om a^-3 + Ode a^-3(1+3w0+3wa) + Ode*(wa)*(terms)... 
    Actually: H^2/H0^2 = Om0*a^-3 + Ode0 * exp(-3*(1+w0+wa)*ln(a) + 3*wa*(a-1))
    = Om0/a^3 + Ode0 * a^(-3-3w0-3wa) * exp(3*wa*(a-1))
    
    dlnH/dlna = (1/2) dln(H^2)/dlna
    """
    Ode0 = 1.0 - Om0
    # H2(a) = Om0 * a^-3 + Ode0 * a^(-3*(1+w0+wa)) * exp(3*wa*(a-1))
    # dH2/dlna = -3*Om0*a^-3 + Ode0 * a^(-3*(1+w0+wa)) * exp(3*wa*(a-1)) * (-3*(1+w0+wa) + 3*wa)
    #         = -3*Om0*a^-3 + Ode0*a^(-3-3w0-3wa)*exp(3wa*(a-1)) * (-3-3w0)
    #          since -3*(1+w0+wa)+3*wa = -3-3w0
    
    def H2(a):
        return Om0 / a**3 + Ode0 * a**(-3.0 - 3.0*w0 - 3.0*wa) * np.exp(3.0*wa*(a - 1.0))
    
    def dlnH_dlna(a):
        H2v = H2(a)
        dH2 = -3.0*Om0/a**3 + Ode0 * a**(-3.0-3.0*w0-3.0*wa) * np.exp(3.0*wa*(a-1.0)) * (-3.0 - 3.0*w0)
        return 0.5 * dH2 / H2v
    
    def Om_at_a(a):
        return Om0 / a**3 / H2(a)
    
    # Solve ODE: y1 = D, y2 = f = dD/dlna
    # y2' = -y2*(1.5 + dlnH_dlna) + 1.5*Om(a)
    # y1' = y2
    
    def ode(x, y):
        a = np.exp(x)
        dlnH = dlnH_dlna(a)
        Om_a = Om_at_a(a)
        f = y[1]
        df = -f * (1.5 + dlnH) + 1.5 * Om_a
        return [f, df]
    
    # Integrate from x=0 (a=1) backwards to x=ln(min_a)
    # At a=1, D(1)=1, and f(1) ~ 0.55 roughly. Use a shooting method or integrate from early times.
    # Better: integrate from a small a to a=1, normalize D(1)=1.
    
    a_min = a_values.min()
    x_min = np.log(a_min)
    x_max = 0.0
    
    # Use a grid for integration
    n_pts = 200
    x_grid = np.linspace(x_min, x_max, n_pts)
    
    # Integrate forward from x_min
    # Need initial conditions at x_min. Use the approximation f ~ Om^0.55 at small a.
    a_init = np.exp(x_min)
    f_init = _Om_of_a(Om0, w0, wa, a_init)**0.55
    y_init = [1.0, f_init]  # D=1 at a_init, we'll renormalize
    
    # Use solve_ivp
    from scipy.integrate import solve_ivp
    sol = solve_ivp(ode, [x_min, x_max], y_init, rtol=1e-10, atol=1e-12, method='RK45')
    
    if not sol.success:
        # Fallback: use constant gamma
        f_vals = np.ones_like(a_values) * 0.55
        D_vals = np.exp(-np.array([quad(lambda x_: (1.0 + Ode0 * np.exp(3*((1+w0+wa)*x_ - wa*(np.exp(x_)-1)))) ** 0.55 / (1.0 + Ode0 * np.exp(3*((1+w0+wa)*x_ - wa*(np.exp(x_)-1)))) ** 0.55, x_, 0)[0] for a in a_values]))
        return f_vals, D_vals
    
    # Interpolate to a_values
    x_eval = np.log(a_values)
    D_raw = np.interp(x_eval, sol.t, sol.y[0])
    f_raw = np.interp(x_eval, sol.t, sol.y[1])
    
    # Normalize D so D(1) = 1
    D_norm_factor = D_raw[-1] if a_values[-1] == 1.0 else np.interp(0.0, sol.t, sol.y[0])
    D_normed = D_raw / D_norm_factor
    
    # The normalized growth factor: D(a)/D(1)
    return f_raw, D_normed

def _gamma_of_a(c, a, Om0, w0, wa):
    """Compute gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q"""
    Om_a = _Om_of_a(Om0, w0, wa, a)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
    return gamma, Om_a, q, u

def _candidate_f_and_D(c, Om0, w0, wa, a_values):
    """Compute candidate f and D/D(1) using the parameterization."""
    f_vals = np.zeros_like(a_values)
    for i, a in enumerate(a_values):
        gamma, Om_a, _, _ = _gamma_of_a(c, a, Om0, w0, wa)
        # Clamp gamma to [0.1, 1.2]
        gamma = np.clip(gamma, 0.1, 1.2)
        f_vals[i] = Om_a ** gamma
    
    # Compute D(a)/D(1) = exp(-integral_{ln(a)}^0 f(exp(x)) dx)
    D_vals = np.zeros_like(a_values)
    D_vals[-1] = 1.0  # at a=1
    # We need to integrate f(exp(x)) from ln(a) to 0
    # Use cumulative trapezoidal or just quad for each point
    for i in range(len(a_values)):
        a = a_values[i]
        if a >= 1.0 - 1e-10:
            D_vals[i] = 1.0
            continue
        x_low = np.log(a)
        # Integrate f(exp(x)) dx from x_low to 0
        # Use adaptive quadrature with our f function
        def integrand(x):
            ai = np.exp(x)
            gamma, Om_ai, _, _ = _gamma_of_a(c, ai, Om0, w0, wa)
            gamma = np.clip(gamma, 0.1, 1.2)
            return Om_ai ** gamma
        try:
            integral, _ = quad(integrand, x_low, 0.0, limit=200)
        except:
            integral = 0.55 * (0.0 - x_low)
        D_vals[i] = np.exp(-integral)
    
    return f_vals, D_vals

def _objective(c):
    """Compute the max relative error over a grid of validation cosmologies."""
    if not _time_ok():
        return 1e6
    
    c = np.array(c, dtype=float)
    # Check gamma bounds
    Om0_grid = [0.2, 0.27, 0.3, 0.35]
    w0_grid = [-1.2, -1.0, -0.9, -0.8, -0.5]
    wa_grid = [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.3]
    a_grid = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    
    max_rel_err = 0.0
    
    for Om0 in Om0_grid:
        if not _time_ok():
            return 1e6
        for w0 in w0_grid:
            if not _time_ok():
                return 1e6
            for wa in wa_grid:
                if not _time_ok():
                    return 1e6
                # Ensure physical: Om(a) > 0 for a in [0.1, 1]
                Ode0 = 1.0 - Om0
                # Check that dark energy doesn't dominate too badly
                if Ode0 <= 0:
                    continue
                
                # Reference
                f_ref, D_ref = _reference_f_and_D(Om0, w0, wa, a_grid)
                # Candidate
                f_cand, D_cand = _candidate_f_and_D(c, Om0, w0, wa, a_grid)
                
                # Relative errors
                for j in range(len(a_grid)):
                    # f relative error
                    if abs(f_ref[j]) > 1e-10:
                        rel_f = abs(f_cand[j] - f_ref[j]) / abs(f_ref[j])
                    else:
                        rel_f = abs(f_cand[j] - f_ref[j])
                    # D relative error
                    if abs(D_ref[j]) > 1e-10:
                        rel_D = abs(D_cand[j] - D_ref[j]) / abs(D_ref[j])
                    else:
                        rel_D = abs(D_cand[j] - D_cand[j])
                    max_rel_err = max(max_rel_err, rel_f, rel_D)
    
    return 100.0 * max_rel_err

def construct():
    """Find optimal 6 coefficients for the growth-index formula."""
    if not _time_ok():
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Initial guess: constant gamma ~ 0.55, small perturbations
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Bounds
    bounds = [(-10, 10)] * 6
    
    best_c = x0[:]
    best_score = _objective(x0)
    
    if not _time_ok():
        return best_c
    
    # Nelder-Mead optimization
    try:
        result = nelder_mead(
            _objective,
            x0,
            bounds=bounds,
            maxiter=300,
            tol=1e-6,
            xatol=1e-4,
            fatol=1e-4,
            options={'disp': False, 'initial_simplex': 0.01 * np.ones(6)}
        )
        if result.fun < best_score:
            best_c = result.x.tolist()
            best_score = result.fun
    except:
        pass
    
    if not _time_ok():
        return best_c
    
    # Local refinement with smaller steps
    try:
        result2 = nelder_mead(
            _objective,
            best_c,
            bounds=bounds,
            maxiter=200,
            tol=1e-8,
            xatol=1e-5,
            fatol=1e-5,
            options={'disp': False, 'initial_simplex': 0.001 * np.ones(6)}
        )
        if result2.fun < best_score:
            best_c = result2.x.tolist()
            best_score = result2.fun
    except:
        pass
    
    if not _time_ok():
        return best_c
    
    # Final polish
    try:
        result3 = nelder_mead(
            _objective,
            best_c,
            bounds=bounds,
            maxiter=100,
            tol=1e-10,
            xatol=1e-6,
            fatol=1e-6,
            options={'disp': False, 'initial_simplex': 0.0001 * np.ones(6)}
        )
        if result3.fun < best_score:
            best_c = result3.x.tolist()
            best_score = result3.fun
    except:
        pass
    
    # Ensure all values are finite and in range
    best_c = np.array(best_c, dtype=float)
    best_c = np.clip(best_c, -10, 10)
    best_c[~np.isfinite(best_c)] = 0.0
    
    return best_c.tolist()
# EVOLVE-BLOCK-END


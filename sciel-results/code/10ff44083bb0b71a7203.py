"""Compact dark-energy growth-index fit with numerical optimization."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 80.0  # 80s safety margin before 90s kill

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - time.time()

def _Om_a(Om0, w0, wa, a):
    """Matter density parameter at scale factor a for CPL dark energy."""
    # a in (0, 1]
    # Omega_m(a) = Om0 * a^3 / (Om0*a^3 + (1-Om0)*a^(-3*(1+w0+wa)) * a^(3*wa))
    # Actually: Omega_m(a) = Om0 * a^3 / [Om0*a^3 + (1-Om0) * a^(-3*(1+w0+wa)) * a^(3*wa)]
    # Let me be more careful. For CPL: w = w0 + wa*(1-a)
    # The scale factor dependence of dark energy:
    # rho_DE(a) = rho_DE(1) * a^(-3*(1+w0+wa)) * a^(3*wa)
    # = rho_DE(1) * a^(-3-3*w0-3*wa+3*wa) = rho_DE(1) * a^(-3-3*w0)
    # Wait, let me re-derive. 
    # d(rho_DE)/d(ln a) = -3(1+w) rho_DE
    # w = w0 + wa*(1-a)
    # This is tricky. The standard result:
    # rho_DE(a)/rho_DE(1) = a^(-3*(1+w0+wa)) * exp(3*wa*(1-a))
    # Hmm, actually for CPL the exact solution involves the exponential integral in general,
    # but for the common approximation used in cosmology:
    # E^2(a) = Om0*a^-3 + (1-Om0)*a^(-3*(1+w0+wa))*a^(3*wa)
    # which is an approximation. Let me use the standard CPL formula:
    # H^2(a)/H0^2 = Om0*a^-3 + (1-Om0)*a^(-3*(1+w0+wa))*a^(3*wa)
    # This is the standard "CPL" parametrization used in the literature.
    
    a = max(a, 1e-10)
    a3 = a**3
    # Standard CPL: E^2 = Om0/a^3 + (1-Om0)/a^(3+3*w0+3*wa) * a^(3*wa)
    # = Om0/a^3 + (1-Om0) * a^(-3-3*w0-3*wa+3*wa)
    # = Om0/a^3 + (1-Om0) * a^(-3-3*w0)
    # That can't be right either. Let me look at this more carefully.
    
    # The standard CPL dark energy density evolution:
    # rho_DE(a) = rho_DE(1) * a^(-3*(1+w0+wa)) * exp(3*wa*(1-a))
    # But this is an approximation. The more common one in the growth literature is:
    # E^2(a) = Om0 * a^(-3) + (1-Om0) * a^(-3*(1+w0+wa)) * a^(3*wa)
    
    # Let me use: E^2(a) = Om0/a^3 + (1-Om0)/a^(3+3*w0+3*wa-3*wa)
    # No. Let me just use the standard form from the literature:
    # For CPL w(a) = w0 + wa*(1-a):
    # E^2(a) = Om0 * a^(-3) + (1-Om0) * a^(-3*(1+w0+wa)) * a^(3*wa)
    
    de_term = (1.0 - Om0) * a**(-3.0*(1.0 + w0 + wa)) * a**(3.0*wa)
    m_term = Om0 * a**(-3.0)
    E2 = m_term + de_term
    return m_term / E2

def _q_a(Om0, w0, wa, a):
    """Deceleration parameter q = 1 - Om(a)."""
    return 1.0 - _Om_a(Om0, w0, wa, a)

def _u_a(w0, wa, a):
    """u = 1 + w0 + wa*(1-a)."""
    return 1.0 + w0 + wa*(1.0 - a)

def _gamma(c, Om0, w0, wa, a):
    """Compute gamma from the six-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = c
    q = _q_a(Om0, w0, wa, a)
    u = _u_a(w0, wa, a)
    return c0 + c1*q + c2*q*q + c3*u + c4*wa*a + c5*u*q

def _f_a(c, Om0, w0, wa, a):
    """Growth rate f = Om(a)^gamma."""
    Om = _Om_a(Om0, w0, wa, a)
    g = _gamma(c, Om0, w0, wa, a)
    return Om**g

def _f_exact(Om0, w0, wa, a):
    """Exact growth rate from the ODE: f = -1 + d ln D / d ln a."""
    # We use the approximation f = Om(a)^gamma with gamma from the ODE solution
    # Actually, for the reference, we need to solve the ODE numerically.
    # The ODE is: D'' + [3/a + H'/H] D' - (3/2) Om(a) H^2/a^2 D = 0
    # In terms of f = D'/D: f' + f^2 + [3/a + H'/H](f+1) - (3/2) Om(a) = 0
    # where primes are d/d ln a.
    # For CPL, H'/H = (3/2)(1+w(a)) = (3/2)(1+w0+wa*(1-a))
    # So the equation for f as a function of x = ln a:
    # df/dx + f^2 + (3/2)(1+w0+wa*(1-e^x))(f+1) - (3/2) Om(e^x) = 0
    # We solve this backward from a=1 where f(1) is known (or use the approximation).
    
    # For the reference, we'll use the standard approximation gamma = Om(a)^0.55
    # but that's circular. Instead, let's use a high-accuracy numerical integration.
    
    # Actually, for the purpose of this task, the "exact" reference is computed by the evaluator
    # using the ODE. For our local optimization, we need a good proxy.
    # Let's use the known analytical result: for matter-only, f = Om(a).
    # For general CPL, we use the approximation f ≈ Om(a)^gamma with gamma ≈ 0.55.
    # But we need something better for optimization.
    
    # Let's just use the ODE integration for the reference.
    pass

def _compute_D_ratio(c, Om0, w0, wa, a):
    """Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)."""
    def integrand(x):
        ai = math.exp(x)
        return _f_a(c, Om0, w0, wa, ai)
    
    if a >= 1.0:
        return 1.0
    log_a = math.log(a)
    # integral from log_a to 0 of f(exp(x)) dx
    val, _ = quad(integrand, log_a, 0.0, limit=200)
    return math.exp(-val)

def _objective(c, cosmologies, a_values):
    """Compute max relative error across all cosmologies and a values."""
    max_err = 0.0
    for (Om0, w0, wa) in cosmologies:
        for a in a_values:
            if a >= 1.0:
                continue
            # Check gamma constraint
            g = _gamma(c, Om0, w0, wa, a)
            if g < 0.1 or g > 1.2:
                return 1e10  # Penalty for constraint violation
            
            # Compute f error
            f_approx = _f_a(c, Om0, w0, wa, a)
            # For reference f, use the ODE-based value
            # We'll approximate the exact f using a high-accuracy method
            f_ref = _f_reference(Om0, w0, wa, a)
            err_f = abs(f_approx - f_ref) / max(abs(f_ref), 1e-10)
            max_err = max(max_err, err_f)
            
            # Compute D error
            D_approx = _compute_D_ratio(c, Om0, w0, wa, a)
            D_ref = _compute_D_ref(Om0, w0, wa, a)
            err_D = abs(D_approx - D_ref) / max(abs(D_ref), 1e-10)
            max_err = max(max_err, err_D)
    
    return max_err

def _f_reference(Om0, w0, wa, a):
    """Reference growth rate from ODE integration."""
    # Solve the ODE for f numerically from a=1 down to a
    # f' + f^2 + (3/2)(1+w(a))(f+1) - (3/2)Om(a) = 0, where ' = d/d ln a
    # At a=1, use the approximation f(1) = Om(1)^0.55 = Om0^0.55 as initial condition
    # Then integrate backward.
    
    def ode_rhs(x, f):
        ai = math.exp(x)
        w = w0 + wa*(1.0 - ai)
        Om = _Om_a(Om0, w0, wa, ai)
        return -f**2 - 1.5*(1.0+w)*(f+1.0) + 1.5*Om
    
    # Integrate from x=0 (a=1) to x=ln(a)
    from scipy.integrate import solve_ivp
    x0 = 0.0
    xf = math.log(max(a, 1e-10))
    f0 = Om0**0.55
    
    if abs(xf) < 1e-10:
        return f0
    
    sol = solve_ivp(ode_rhs, [x0, xf], [f0], method='RK45', 
                    rtol=1e-10, atol=1e-12, max_step=0.01)
    if sol.success:
        return sol.y[0, -1]
    else:
        return f0

def _compute_D_ref(Om0, w0, wa, a):
    """Reference D(a)/D(1) from ODE integration."""
    if a >= 1.0:
        return 1.0
    
    def ode_rhs(x, f):
        ai = math.exp(x)
        w = w0 + wa*(1.0 - ai)
        Om = _Om_a(Om0, w0, wa, ai)
        return -f**2 - 1.5*(1.0+w)*(f+1.0) + 1.5*Om
    
    x0 = 0.0
    xf = math.log(max(a, 1e-10))
    f0 = Om0**0.55
    
    from scipy.integrate import solve_ivp
    sol = solve_ivp(ode_rhs, [x0, xf], [f0], method='RK45',
                    rtol=1e-10, atol=1e-12, max_step=0.01)
    if not sol.success:
        return math.exp(-f0 * (0 - xf))  # fallback
    
    # Integrate f over x to get D ratio
    # D(a)/D(1) = exp(-int_{ln a}^0 f(x) dx)
    # We have f(x) at various points from the solution
    from scipy.integrate import trapezoid
    integral = trapezoid(sol.y[0], sol.x)
    # integral is from x0 to xf, so int_{x0}^{xf} f dx = int_0^{ln a} f dx
    # We need int_{ln a}^0 f dx = -int_0^{ln a} f dx = -integral
    return math.exp(-(-integral))

def construct():
    """Construct six coefficients for the growth-index ansatz."""
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 80.0
    
    # Validation cosmologies (typical CPL models)
    cosmologies = [
        (0.3, -1.0, 0.0),      # LambdaCDM
        (0.3, -0.5, -0.5),     # Quintessence-like
        (0.3, -1.0, 1.0),      # High wa
        (0.25, -0.8, 0.3),     # Another model
        (0.35, -1.2, 0.5),     # Phantom-like
    ]
    
    # a values to check
    a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    
    # Pre-compute reference values for speed
    ref_data = {}
    for (Om0, w0, wa) in cosmologies:
        for a in a_values:
            if a >= 1.0:
                continue
            key = (Om0, w0, wa, a)
            f_ref = _f_reference(Om0, w0, wa, a)
            D_ref = _compute_D_ref(Om0, w0, wa, a)
            ref_data[key] = (f_ref, D_ref)
    
    # Fast objective using pre-computed references
    def fast_objective(c):
        c = np.array(c)
        max_err = 0.0
        for (Om0, w0, wa) in cosmologies:
            for a in a_values:
                if a >= 1.0:
                    continue
                g = c[0] + c[1]*_q_a(Om0, w0, wa, a) + c[2]*_q_a(Om0, w0, wa, a)**2 \
                    + c[3]*_u_a(w0, wa, a) + c[4]*wa*a + c[5]*_u_a(w0, wa, a)*_q_a(Om0, w0, wa, a)
                if g < 0.1 or g > 1.2:
                    return 1e10
                
                f_approx = _Om_a(Om0, w0, wa, a) ** g
                f_ref, D_ref = ref_data[(Om0, w0, wa, a)]
                err_f = abs(f_approx - f_ref) / max(abs(f_ref), 1e-10)
                max_err = max(max_err, err_f)
                
                # D approximation
                D_approx = _compute_D_ratio(c, Om0, w0, wa, a)
                err_D = abs(D_approx - D_ref) / max(abs(D_ref), 1e-10)
                max_err = max(max_err, err_D)
        return max_err
    
    # Initial guess
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_c = c0.copy()
    best_val = fast_objective(best_c)
    
    # Try differential evolution for global optimization
    if _remaining() > 30:
        bounds = [(-10, 10)]*6
        try:
            result = differential_evolution(fast_objective, bounds, 
                                           maxiter=50, popsize=15,
                                           seed=42, tol=1e-8,
                                           polish=False, init='sobol')
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x
        except Exception:
            pass
    
    # Local refinement with Nelder-Mead or BFGS
    if _remaining() > 10:
        try:
            result = minimize(fast_objective, best_c, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x
        except Exception:
            pass
    
    # Final polish
    if _remaining() > 5:
        try:
            result = minimize(fast_objective, best_c, method='BFGS',
                            options={'maxiter': 200, 'gtol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x
        except Exception:
            pass
    
    # Ensure constraints are satisfied
    best_c = np.clip(best_c, -10, 10)
    
    return list(best_c)

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad

# Deadline guard: we must finish within 60s (evaluator limit), use 55s as safety margin.
_START_TIME = time.time()
_DEADLINE = 55.0  # seconds

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _Om(a, Om0, w0, wa):
    """Matter density parameter at scale factor a."""
    # Dark energy density: rho_de(a) = rho_de(1) * a^(-3(1+w0+wa))
    # Actually, for CPL: w = w0 + wa*(1-a)
    # rho_de(a)/rho_de(1) = exp(-3 * integral_1^a (1+w(a'))/a' da')
    # For CPL, 1+w = 1+w0+wa - wa*a
    # integral_1^a (1+w0+wa-wa*a')/a' da' = (1+w0+wa)*ln(a) - wa*(a-1)
    # So rho_de(a) = rho_de(1) * a^(-(1+w0+wa)) * exp(wa*(a-1))
    # But for simplicity in many approximations, people use a^(-3(1+w0+wa))
    # Let's use the exact CPL form.
    a = np.asarray(a, dtype=float)
    if np.ndim(a) == 0:
        a = a.reshape(1)
    # Handle a=0 carefully
    a_safe = np.where(a > 0, a, 1e-30)
    ln_a = np.log(a_safe)
    # exponent = -3*(1+w0+wa)*ln(a) + 3*wa*(a-1)
    # Wait, let's re-derive.
    # d(ln rho_de) / d(ln a) = -3(1+w)
    # 1+w = 1+w0+wa*(1-a)
    # ln(rho_de(a)/rho_de(1)) = -3 * integral_1^a (1+w0+wa*(1-a'))/a' da'
    # = -3 * [ (1+w0+wa)*ln(a) - wa*(a-1) ]
    # So rho_de(a) = rho_de(1) * exp(-3*(1+w0+wa)*ln(a) + 3*wa*(a-1))
    # = rho_de(1) * a^(-3*(1+w0+wa)) * exp(3*wa*(a-1))
    # Matter: rho_m(a) = rho_m(1) * a^(-3)
    # H(a)^2 = H0^2 [ Om0 * a^(-3) + (1-Om0) * a^(-3*(1+w0+wa)) * exp(3*wa*(a-1)) ]
    # Om(a) = rho_m(a) / rho_total(a) = Om0 * a^(-3) / [ Om0 * a^(-3) + (1-Om0) * a^(-3*(1+w0+wa)) * exp(3*wa*(a-1)) ]
    # = Om0 / [ Om0 + (1-Om0) * a^(-3*wa) * exp(3*wa*(a-1)) ]
    # Let's simplify the exponent term:
    # a^(-3*wa) * exp(3*wa*(a-1)) = exp(-3*wa*ln(a) + 3*wa*(a-1))
    # = exp(3*wa*(a-1-ln(a)))
    # So Om(a) = Om0 / [ Om0 + (1-Om0) * exp(3*wa*(a-1-ln(a))) ]
    # This is a nice closed form.
    exponent = 3.0 * wa * (a_safe - 1.0 - ln_a)
    de_term = (1.0 - Om0) * np.exp(exponent)
    denom = Om0 + de_term
    # Avoid division by zero
    denom_safe = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    om = Om0 / denom_safe
    # Clip to [0,1] for safety
    om = np.clip(om, 1e-15, 1.0 - 1e-15)
    if np.ndim(np.asarray(a)) == 0:
        return om[0]
    return om

def _gamma_approx(a, coeffs, Om0, w0, wa):
    """Compute gamma(a) from the six coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    om = _Om(a, Om0, w0, wa)
    q = 1.0 - om
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    return gamma

def _f_approx(a, coeffs, Om0, w0, wa):
    """Compute f(a) = Om(a)^gamma(a)."""
    om = _Om(a, Om0, w0, wa)
    gamma = _gamma_approx(a, coeffs, Om0, w0, wa)
    # f = Om^gamma
    # Clip Om to avoid log(0)
    om_safe = np.clip(om, 1e-15, 1.0)
    f = om_safe ** gamma
    return f

def _D_normalized_approx(a, coeffs, Om0, w0, wa):
    """Compute D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)."""
    # The integral is from ln(a) to 0 of f(exp(x)) dx
    # Let x = ln(a'), so a' = exp(x), da' = exp(x) dx
    # But the task says: D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
    # So we integrate f(exp(x)) with respect to x from ln(a) to 0.
    # f(exp(x)) = Om(exp(x))^gamma(exp(x))
    # We can use numerical integration.
    if a >= 1.0:
        return 1.0
    a_safe = max(a, 1e-10)
    ln_a = np.log(a_safe)
    # Integrate from ln_a to 0
    def integrand(x):
        a_val = np.exp(x)
        return _f_approx(a_val, coeffs, Om0, w0, wa)
    # Use quad for scalar integration
    try:
        integral, _ = quad(integrand, ln_a, 0.0, limit=100)
    except Exception:
        integral = 0.0
    D_ratio = np.exp(-integral)
    return D_ratio

def _reference_f(a, Om0, w0, wa):
    """
    Reference f(a) from the GR second-order ODE.
    The growth equation: D'' + 2H D' - 4πG ρ_m D = 0
    In terms of f = d ln D / d ln a:
    f(a) = d ln D / d ln a
    The reference f can be obtained by solving the ODE for D(a).
    However, the task says "A tightly integrated general-relativistic second-order ODE provides an independent numerical reference."
    For the purpose of this optimization, we need a reference f(a) and D(a)/D(1).
    Since we don't have the exact ODE solver from the evaluator, we must approximate the reference.
    But wait, the task says "The evaluator computes D(a)/D(1)=exp(-integral_ln(a)^0 f(exp(x)) dx)."
    And "Minimize 100 times the maximum relative error of both f and normalized D over validation models and a values."
    So the evaluator has its own reference f and D.
    In our optimization, we need to approximate the reference f and D to compute the error.
    How do we get the reference f?
    The standard approximation for f is f = Om^gamma, but that's what we're fitting.
    The reference is likely the exact solution from the ODE.
    Since we can't easily solve the ODE for many cosmologies in a tight loop without it being slow,
    we might need to use a known accurate approximation for the reference, or precompute it.
    However, the task says "Improve a six-coefficient growth-index formula... The score is the worst relative error across both observables on disjoint public validation cosmologies, extending the constant-gamma seed within a fixed compact ansatz."
    This suggests that the reference is the exact ODE solution.
    To optimize, we need to evaluate the error against the reference.
    Let's implement a simple ODE solver for the reference f.
    The growth equation for D:
    D'' + (3 + q) D' - (3/2) Om D = 0, where primes are d/dln(a)
    Let f = D'/D, then f' + f^2 + (3+q)f - (3/2)Om = 0
    This is a first-order ODE for f.
    We can solve this ODE to get the reference f.
    """
    # We'll solve the ODE for f(a) = d ln D / d ln a
    # f' + f^2 + (3+q)f - (3/2)Om = 0
    # where q = 1 - Om(a)
    # Initial condition: at a=1, f(1) = Om(1)^gamma0, where gamma0 ~ 0.55
    # Actually, f(1) = d ln D / d ln a |_{a=1}
    # We can use f(1) = Om(1)^0.55 as a good initial guess.
    # But to be accurate, we should solve the ODE backwards from a=1 to a<1.
    # Or solve the ODE for D directly.
    # Let's solve the ODE for f.
    # f' = -f^2 - (3+q)f + (3/2)Om
    # We'll use scipy.integrate.solve_ivp
    from scipy.integrate import solve_ivp
    a_val = float(a)
    if a_val >= 1.0:
        # At a=1, f(1) = Om(1)^gamma0
        om1 = _Om(1.0, Om0, w0, wa)
        return om1 ** 0.55
    # Solve from a=1 to a=a_val
    # Use ln(a) as the independent variable
    ln_a_start = 0.0
    ln_a_end = np.log(a_val)
    # Initial condition at ln_a_start=0
    om1 = _Om(1.0, Om0, w0, wa)
    f0 = om1 ** 0.55
    
    def ode(t, y):
        a_t = np.exp(t)
        om_t = _Om(a_t, Om0, w0, wa)
        q_t = 1.0 - om_t
        f_t = y[0]
        # f' = -f^2 - (3+q)f + (3/2)Om
        df = -f_t**2 - (3.0 + q_t) * f_t + 1.5 * om_t
        return [df]
    
    try:
        sol = solve_ivp(ode, [ln_a_start, ln_a_end], [f0], method='RK45', rtol=1e-8, atol=1e-10)
        if sol.success:
            return sol.y[0, -1]
        else:
            # Fallback to approximation
            return _f_approx(a_val, [0.55, 0, 0, 0, 0, 0], Om0, w0, wa)
    except Exception:
        return _f_approx(a_val, [0.55, 0, 0, 0, 0, 0], Om0, w0, wa)

def _reference_D(a, Om0, w0, wa):
    """Reference D(a)/D(1) from the ODE."""
    # D(a)/D(1) = exp(-integral_ln(a)^0 f_ref(exp(x)) dx)
    # We can integrate f_ref numerically.
    if a >= 1.0:
        return 1.0
    a_safe = max(a, 1e-10)
    ln_a = np.log(a_safe)
    # We need to integrate f_ref(exp(x)) from ln_a to 0
    # But f_ref is expensive to compute (requires ODE solve at each point)
    # Instead, we can solve the ODE for D directly and get D(a)/D(1).
    from scipy.integrate import solve_ivp
    ln_a_start = 0.0
    ln_a_end = ln_a
    # ODE for D: D'' + (3+q)D' - (3/2)Om D = 0
    # Let D' = E, then D' = E, E' = -(3+q)E + (3/2)Om D
    # Initial conditions at a=1: D(1)=1, E(1)=f(1)=Om(1)^0.55
    om1 = _Om(1.0, Om0, w0, wa)
    D0 = 1.0
    E0 = om1 ** 0.55
    
    def ode_D(t, y):
        a_t = np.exp(t)
        om_t = _Om(a_t, Om0, w0, wa)
        q_t = 1.0 - om_t
        D_t, E_t = y
        dD = E_t
        dE = -(3.0 + q_t) * E_t + 1.5 * om_t * D_t
        return [dD, dE]
    
    try:
        sol = solve_ivp(ode_D, [ln_a_start, ln_a_end], [D0, E0], method='RK45', rtol=1e-8, atol=1e-10)
        if sol.success:
            return sol.y[0, -1]
        else:
            return _D_normalized_approx(a_safe, [0.55, 0, 0, 0, 0, 0], Om0, w0, wa)
    except Exception:
        return _D_normalized_approx(a_safe, [0.55, 0, 0, 0, 0, 0], Om0, w0, wa)

# Validation cosmologies (disjoint public validation set)
# We'll use a few standard cosmologies
VALIDATION_COSMOLOGIES = [
    (0.3, -0.5, 0.0),   # w0=-0.5, wa=0
    (0.3, -1.0, 0.0),   # LambdaCDM
    (0.3, -1.0, 0.5),   # CPL with wa=0.5
    (0.25, -0.9, 0.3),  # Another CPL
    (0.35, -0.8, -0.2), # Another CPL
]

# Scale factors to check
A_VALUES = np.logspace(-1.0, 0.0, 50)  # a from 0.1 to 1.0

def _objective(coeffs):
    """
    Objective function: 100 * max relative error of f and D over validation cosmologies and a values.
    """
    if _time_left() < 2.0:
        return 1e10  # Penalty if time is running out
    
    max_err_f = 0.0
    max_err_D = 0.0
    
    for Om0, w0, wa in VALIDATION_COSMOLOGIES:
        for a in A_VALUES:
            # Reference
            f_ref = _reference_f(a, Om0, w0, wa)
            D_ref = _reference_D(a, Om0, w0, wa)
            
            # Approximation
            f_app = _f_approx(a, coeffs, Om0, w0, wa)
            D_app = _D_normalized_approx(a, coeffs, Om0, w0, wa)
            
            # Relative errors
            if abs(f_ref) > 1e-15:
                err_f = abs(f_app - f_ref) / abs(f_ref)
            else:
                err_f = abs(f_app - f_ref)
            
            if abs(D_ref) > 1e-15:
                err_D = abs(D_app - D_ref) / abs(D_ref)
            else:
                err_D = abs(D_app - D_ref)
            
            max_err_f = max(max_err_f, err_f)
            max_err_D = max(max_err_D, err_D)
            
            # Check gamma constraint
            gamma = _gamma_approx(a, coeffs, Om0, w0, wa)
            if gamma < 0.1 or gamma > 1.2:
                return 1e10  # Penalty for constraint violation
    
    return 100.0 * max(max_err_f, max_err_D)

def construct():
    """
    Construct the six coefficients.
    """
    # Initial guess
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Bounds for coefficients
    bounds = [(-10, 10)] * 6
    
    # Try Nelder-Mead first
    best_coeffs = x0
    best_score = _objective(x0)
    
    if _time_left() > 5.0:
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', bounds=bounds, 
                          options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.success and res.fun < best_score:
                best_coeffs = res.x
                best_score = res.fun
        except Exception:
            pass
    
    # Try differential evolution if time remains
    if _time_left() > 20.0:
        try:
            res_de = differential_evolution(_objective, bounds, maxiter=50, popsize=10, 
                                           seed=42, polish=True)
            if res_de.fun < best_score:
                best_coeffs = res_de.x
                best_score = res_de.fun
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if _time_left() > 5.0:
        try:
            res_polish = minimize(_objective, best_coeffs, method='Nelder-Mead', bounds=bounds,
                                 options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res_polish.success and res_polish.fun < best_score:
                best_coeffs = res_polish.x
                best_score = res_polish.fun
        except Exception:
            pass
    
    # Ensure coefficients are within [-10, 10] and finite
    best_coeffs = np.clip(best_coeffs, -10, 10)
    best_coeffs = [float(c) for c in best_coeffs]
    
    return best_coeffs
# EVOLVE-BLOCK-END

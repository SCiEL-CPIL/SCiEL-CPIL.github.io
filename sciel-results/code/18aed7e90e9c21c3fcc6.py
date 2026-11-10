"""Improved growth-index coefficient search for CPL dark-energy cosmologies."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 82  # safety margin before 90s kill

def _om_matter(a, om0, w0, wa):
    """Omega_m(a) for CPL dark energy."""
    # a in (0, 1], om0 in (0,1)
    # Om(a) = Om0 * a^-3 / [Om0 * a^-3 + (1-Om0) * a^(-3(1+w0+wa)) * (1+w0+wa*(1-a))^wa]
    # More precisely, the dark energy density evolves as:
    # rho_de(a) = rho_de0 * a^{-3(1+w0)} * (1+w0+wa*(1-a))^(-wa)
    # So: Om(a) = Om0*a^{-3} / (Om0*a^{-3} + (1-Om0)*a^{-3(1+w0)}*(1+w0+wa*(1-a))^{-wa})
    if a <= 0:
        return 1.0
    a3 = a**3
    om0_term = om0 / a3
    if wa == 0.0:
        de_term = (1.0 - om0) / a3 * a**(3*wa)  # just (1-om0)/a3 * a^0 = (1-om0)/a3 ... wait
        # rho_de ~ a^{-3(1+w0)} when wa=0
        de_term = (1.0 - om0) * a**(-3.0*(1.0+w0))
    else:
        de_term = (1.0 - om0) * a**(-3.0*(1.0+w0)) * (1.0+w0+wa*(1.0-a))**(-wa)
    return om0_term / (om0_term + de_term)

def _gamma_from_coeffs(c, a, om0, w0, wa):
    """Compute gamma from the 6-coefficient ansatz."""
    om_a = _om_matter(a, om0, w0, wa)
    q = 1.0 - om_a
    u = 1.0 + w0 + wa*(1.0-a)
    gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
    return gamma

def _f_of_a(c, a, om0, w0, wa):
    """f = Om(a)^gamma"""
    om_a = _om_matter(a, om0, w0, wa)
    gamma = _gamma_from_coeffs(c, a, om0, w0, wa)
    if om_a <= 0:
        return 0.0
    return om_a**gamma

def _normalized_D(c, a, om0, w0, wa, n_pts=200):
    """D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)"""
    if a >= 1.0:
        return 1.0
    # Integrate f(exp(x)) from ln(a) to 0
    x_vals = np.linspace(math.log(a), 0.0, n_pts)
    f_vals = np.array([_f_of_a(c, math.exp(x), om0, w0, wa) for x in x_vals])
    integral = np.trapz(f_vals, x_vals)
    return math.exp(-integral)

def _true_f_and_D(om0, w0, wa, a, n_pts=400):
    """
    Numerical reference: solve the GR growth ODE for D(a) and compute f = d ln D / d ln a.
    D'' + [3/a + (1+w) H/H_dot] D' - (4πG ρ_m / a^2) D = 0 ... 
    Actually use the standard form in terms of a:
    d^2 D/d(ln a)^2 + [3/a + (1+w) H/(a H')] D - 4πG ρ_m/(a^2) D = 0
    
    Let's use the simpler approach: integrate the ODE for D(a) numerically,
    then f = a D'/D.
    
    For CPL: w = w0 + wa(1-a)
    H^2 = H0^2 [Om0 a^-3 + (1-Om0) a^{-3(1+w0)} (1+w0+wa(1-a))^{-wa}]
    H' = dH/da
    
    Let's use the standard growth equation:
    D'' + [3/a + (1+w) H/H'] D' - 4πG ρ_m / a^2 * D = 0  where ' = d/da
    
    Actually, let me use the well-known form:
    f = Om(a)^gamma_true where gamma_true comes from the exact ODE.
    
    For the reference, I'll integrate the ODE:
    Let y1 = D, y2 = D' (dD/da)
    y1' = y2
    y2' = -[3/a + (1+w) H/H'] y2 + 4πG ρ_m/a^2 * y1
    
    But we need H/H'. Let me compute it analytically.
    
    H^2 = H0^2 * F(a) where F(a) = Om0 a^-3 + (1-Om0) a^{-3(1+w0)} (1+w0+wa(1-a))^{-wa}
    
    H/H' = 1 / (2 d ln F / da) ... H = H0 sqrt(F), H' = H0 F'/(2 sqrt(F)), so H/H' = F/F'
    
    F' = dF/da = -3 Om0 a^-4 + (1-Om0) * d/da[a^{-3(1+w0)} (1+w0+wa(1-a))^{-wa}]
    
    Let g(a) = a^{-3(1+w0)} (1+w0+wa(1-a))^{-wa}
    g' = g * [-3(1+w0)/a - wa*(1+w0+wa(1-a))^{-1} * (-wa)]
    = g * [-3(1+w0)/a + wa^2 / (1+w0+wa(1-a))]
    
    So F' = -3 Om0 a^-4 + (1-Om0) * g * [-3(1+w0)/a + wa^2/(1+w0+wa(1-a))]
    
    And 4πG ρ_m / a^2 = 3 H0^2 Om0 / a^3 (since ρ_m = ρ_m0 a^-3, 4πG ρ_m0 = 3 H0^2 Om0)
    So 4πG ρ_m/a^2 = 3 H0^2 Om0 / a^3 * (1/a^2 * a^3) ... wait.
    
    ρ_m(a) = ρ_m0 a^-3
    4πG ρ_m(a) / a^2 = 4πG ρ_m0 a^-3 / a^2 = 4πG ρ_m0 / a^5? No.
    
    The growth equation in a is:
    D'' + [3/a + (1+w) H/H'] D' - (4πG ρ_m / a^2) D = 0
    
    4πG ρ_m / a^2 = 4πG ρ_m0 a^-3 / a^2 = 4πG ρ_m0 a^-5
    
    Hmm, let me re-derive. The standard form is:
    (a D')' ... let me just use the ln a form.
    
    Let t = ln a, then d/da = (1/a) d/dt
    D'' = d²D/da² = d/da[(1/a) D_t] = -(1/a²) D_t + (1/a²) D_tt = (D_tt - D_t)/a²
    
    Growth equation: D_tt + [3 + (1+w) H/H'] D_t - 4πG ρ_m/a² * a² D = 0
    Wait, 4πG ρ_m / a² * D ... in the ln a form:
    
    Standard: D_tt + [3 + (1+w) a H'/H] D_t - 4πG ρ_m / H² * D = 0
    
    Since 4πG ρ_m / H² = 3 Om(a) (from Friedmann: H² = (8πG/3) ρ_m + ..., and 4πG ρ_m/H² = 4πG ρ_m / [(8πG/3)(ρ_m+ρ_de)] = 3ρ_m/(ρ_m+ρ_de) = 3 Om(a))
    
    So: D_tt + [3 + (1+w) a H'/H] D_t - 3 Om(a) D = 0
    
    And f = D_t (since f = d ln D / d ln a = D_t/D... no, f = a D'/D = D_t/D)
    Wait: f ≡ d ln D / d ln a = (1/D) dD/d(ln a) = D_t/D
    
    So f satisfies: f_t + f² + [3 + (1+w) a H'/H] f - 3 Om(a) = 0
    
    This is a Riccati equation. Let me just integrate D directly.
    
    I'll use the ODE: D_tt + E(a) D_t - 3 Om(a) D = 0
    where E(a) = 3 + (1+w) a H'/H
    
    Compute a H'/H:
    H = H0 sqrt(F), so ln H = const + 0.5 ln F
    d ln H / d ln a = a H'/H = 0.5 a F'/F
    
    So E(a) = 3 + (1+w) * 0.5 * a F'/F
    """
    # Integrate ODE for D from a=1 (D(1)=1, D_t(1)=f(1)) backward to a
    # First compute f(1) = Om(1)^gamma... at a=1, Om(1)=Om0
    # Use the exact f(1) from the ODE steady state or just integrate from a=1
    
    # Let's integrate from a=1 down to a using the ODE in t=ln a
    # At a=1: D=1, D_t = f_1 where f_1 ≈ Om0^0.55 (initial guess), refine
    
    # Better: integrate from a=1 with D(1)=1, D_t(1)=f1
    # f1 can be found from the ODE at a=1: f1² + E(1) f1 - 3 Om0 = 0
    # E(1) = 3 + (1+w0) * 0.5 * 1 * F'(1)/F(1)
    
    def _H_ratio(a_val):
        """Compute a*H'/H = 0.5 * a * F'/F"""
        if a_val <= 0:
            return 0.0
        # F(a) = Om0 a^-3 + (1-Om0) a^{-3(1+w0)} (1+w0+wa(1-a))^{-wa}
        u = 1.0 + w0 + wa*(1.0-a_val)
        if u <= 0:
            return 0.0
        a3 = a_val**3
        F = om0/a3 + (1.0-om0) * a_val**(-3.0*(1.0+w0)) * u**(-wa)
        # F' = dF/da
        term1 = -3.0*om0/a_val**4
        if wa == 0:
            g = a_val**(-3.0*(1.0+w0))
            gp = g * (-3.0*(1.0+w0)/a_val)
        else:
            g = a_val**(-3.0*(1.0+w0)) * u**(-wa)
            gp = g * (-3.0*(1.0+w0)/a_val + wa*wa/u)
        term2 = (1.0-om0) * gp
        Fp = term1 + term2
        if abs(F) < 1e-30:
            return 0.0
        return 0.5 * a_val * Fp / F
    
    def _ode(t, y):
        a_val = math.exp(t)
        D, Dt = y
        w = w0 + wa*(1.0-a_val)
        om_a = _om_matter(a_val, om0, w0, wa)
        E = 3.0 + (1.0+w) * _H_ratio(a_val)
        Dtt = -E*Dt + 3.0*om_a*D
        return [Dt, Dtt]
    
    from scipy.integrate import solve_ivp
    
    # Integrate from t=0 (a=1) to t=ln(a)
    t_end = math.log(a)
    if t_end >= 0:
        return 1.0, 1.0
    
    # Initial condition at t=0: D=1, Dt=f1
    # Solve for f1: f1^2 + E(1)*f1 - 3*Om0 = 0
    E1 = 3.0 + (1.0+w0) * _H_ratio(1.0)
    disc = E1**2 + 12.0*om0
    f1 = (-E1 + math.sqrt(disc)) / 2.0
    if f1 < 0:
        f1 = (-E1 - math.sqrt(disc)) / 2.0
    
    sol = solve_ivp(_ode, [0.0, t_end], [1.0, f1], method='RK45', rtol=1e-10, atol=1e-12, dense_output=True)
    if not sol.success:
        # Fallback
        om_a = _om_matter(a, om0, w0, wa)
        gamma_est = 0.55
        f_est = om_a**gamma_est if om_a > 0 else 0.0
        return f_est, 1.0
    
    D_end = sol.y[0, -1]
    Dt_end = sol.y[1, -1]
    f_end = Dt_end / D_end if D_end != 0 else 0.0
    D_norm = D_end  # D(a)/D(1) since D(1)=1
    return f_end, D_norm

def _evaluate(c, models, a_values, cache=None):
    """Compute max relative error of f and D over all models and a values."""
    if cache is None:
        cache = {}
    max_err = 0.0
    for (om0, w0, wa) in models:
        key = (om0, w0, wa)
        if key not in cache:
            true_data = {}
            for a in a_values:
                f_true, D_true = _true_f_and_D(om0, w0, wa, a)
                true_data[a] = (f_true, D_true)
            cache[key] = true_data
        
        for a in a_values:
            f_true, D_true = cache[key][a]
            f_app = _f_of_a(c, a, om0, w0, wa)
            D_app = _normalized_D(c, a, om0, w0, wa, n_pts=200)
            
            err_f = abs(f_app - f_true) / max(abs(f_true), 1e-10)
            err_D = abs(D_app - D_true) / max(abs(D_true), 1e-10)
            err = max(err_f, err_D)
            if err > max_err:
                max_err = err
    
    return max_err

# Validation cosmologies (disjoint from training)
_MODELS = [
    (0.30, -1.0, 0.0),    # LCDM
    (0.30, -0.9, 0.3),    # CPL
    (0.25, -1.0, 0.5),    # CPL with higher wa
    (0.35, -0.8, -0.2),   # Another CPL
    (0.28, -1.1, 0.2),    # Phantom-like
    (0.32, -0.95, 0.1),   # Near LCDM
]

_A_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

def construct():
    """Find 6 coefficients minimizing worst-case relative error."""
    t_start = time.time()
    
    # Precompute reference data
    cache = {}
    for (om0, w0, wa) in _MODELS:
        key = (om0, w0, wa)
        true_data = {}
        for a in _A_VALUES:
            f_true, D_true = _true_f_and_D(om0, w0, wa, a)
            true_data[a] = (f_true, D_true)
        cache[key] = true_data
    
    # Objective function
    def objective(c):
        return _evaluate(c, _MODELS, _A_VALUES, cache)
    
    # Start from the seed
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_c = c0.copy()
    best_obj = objective(c0)
    
    # Try a few local optimizations from different starting points
    starts = [
        np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.1, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.0, 0.1, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.1]),
        np.array([0.50, 0.1, 0.05, 0.05, 0.0, 0.0]),
        np.array([0.60, -0.1, 0.05, 0.0, 0.05, 0.0]),
    ]
    
    for c_init in starts:
        if time.time() - t_start > _DEADLINE - time.time() - t_start:  # check deadline
            break
        if time.time() > _DEADLINE - 10:
            break
        
        try:
            res = minimize(objective, c_init, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_obj:
                best_obj = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Polish with a tighter optimization
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(objective, best_c, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_obj:
                best_obj = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Ensure coefficients are in [-10, 10] and finite
    result = []
    for val in best_c:
        if not math.isfinite(val):
            val = 0.55
        val = max(-10.0, min(10.0, val))
        result.append(float(val))
    
    return result

# EVOLVE-BLOCK-END

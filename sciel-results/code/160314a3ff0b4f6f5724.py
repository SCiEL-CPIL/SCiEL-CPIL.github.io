"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

def _Om(a, Om0, w0, wa):
    """Matter density parameter Omega_m(a)."""
    # E(a)^2 = Om0 * a^(-3) + (1 - Om0) * a^(-3(1+w0+wa))
    # Actually for CPL: w = w0 + wa*(1-a)
    # The integral for E(a) is not analytic in general, but for the growth ODE
    # we need Om(a) = Om0 * a^3 / E(a)^2
    # E(a)^2 = Om0 * a^(-3) + (1-Om0) * exp(-3*integral_1^a (1+w(a'))/a' da')
    # For CPL, this integral has a known form.
    # Let's use the standard approximation or numerical integration.
    # Actually, for the purpose of this task, we can use the standard relation.
    # E^2(a) = Om_m a^-3 + Om_r a^-4 + Om_k a^-2 + Om_Lambda(a)
    # For CPL dark energy: Om_Lambda(a) = (1-Om_m) * a^(-3(1+w0+wa))
    # This is an approximation that works well for small wa.
    # More precisely: E^2(a) = Om_m a^-3 + (1-Om_m) * a^(-3(1+w0)) * exp(-3*wa*integral_1^a (1-a')/a' da')
    # The integral is: integral_1^a (1-a')/a' da' = integral_1^a (1/a' - 1) da' = [ln(a') - a']_1^a = ln(a) - a + 1
    # So E^2(a) = Om_m a^-3 + (1-Om_m) * a^(-3(1+w0)) * exp(-3*wa*(ln(a) - a + 1))
    # Let's implement this.
    if a <= 0:
        return np.nan
    E2 = Om0 * a**(-3) + (1 - Om0) * a**(-3*(1+w0)) * np.exp(-3*wa*(np.log(a) - a + 1))
    if E2 <= 0:
        return np.nan
    return Om0 * a**3 / E2


def _f_and_D(coeffs, a, Om0, w0, wa, n_points=1000):
    """Compute f(a) and D(a)/D(1) for given coefficients and cosmology."""
    c0, c1, c2, c3, c4, c5 = coeffs
    a_arr = np.linspace(0.05, 1.0, n_points)
    
    f_vals = np.zeros(n_points)
    gamma_vals = np.zeros(n_points)
    
    for i, a_i in enumerate(a_arr):
        Om_i = _Om(a_i, Om0, w0, wa)
        if np.isnan(Om_i) or Om_i <= 0:
            f_vals[i] = np.nan
            gamma_vals[i] = np.nan
            continue
        
        q_i = 1 - Om_i
        u_i = 1 + w0 + wa*(1 - a_i)
        
        gamma_i = c0 + c1*q_i + c2*q_i**2 + c3*u_i + c4*wa*a_i + c5*u_i*q_i
        
        # Check gamma bounds
        if gamma_i < 0.1 or gamma_i > 1.2:
            f_vals[i] = np.nan
            gamma_vals[i] = np.nan
            continue
        
        gamma_vals[i] = gamma_i
        f_vals[i] = Om_i ** gamma_i
    
    # Compute D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
    # x = ln(a), so a = exp(x), dx = da/a
    # integral from ln(a) to 0 of f(exp(x)) dx
    # = integral from a to 1 of f(a')/a' da'
    
    D_ratio = np.zeros(n_points)
    for i, a_i in enumerate(a_arr):
        if np.isnan(f_vals[i]):
            D_ratio[i] = np.nan
            continue
        
        # Integrate f(a')/a' from a_i to 1
        a_int = np.linspace(a_i, 1.0, 200)
        f_int = np.zeros_like(a_int)
        for j, a_j in enumerate(a_int):
            Om_j = _Om(a_j, Om0, w0, wa)
            if np.isnan(Om_j) or Om_j <= 0:
                f_int[j] = 0
                continue
            q_j = 1 - Om_j
            u_j = 1 + w0 + wa*(1 - a_j)
            gamma_j = c0 + c1*q_j + c2*q_j**2 + c3*u_j + c4*wa*a_j + c5*u_j*q_j
            if gamma_j < 0.1 or gamma_j > 1.2:
                f_int[j] = 0
            else:
                f_int[j] = Om_j ** gamma_j
        
        integrand = f_int / a_int
        integral = np.trapz(integrand, a_int)
        D_ratio[i] = np.exp(-integral)
    
    return f_vals, D_ratio


def _true_f_and_D(a, Om0, w0, wa, n_points=1000):
    """Compute true f(a) and D(a)/D(1) using the growth ODE."""
    a_arr = np.linspace(0.05, 1.0, n_points)
    
    # Solve the growth ODE: D'' + 2H D' - 4piG rho_m D = 0
    # In terms of x = ln(a): D_xx + (3/2 + H'/H) D_x - (3/2) Om_m D = 0
    # H'/H = (1/2) d ln E^2 / d ln a = (1/2) (d ln E^2 / d a) * a
    # For CPL: d ln E^2 / d ln a = -3 Om_m - 3(1+wa) Om_Lambda... let's compute numerically.
    
    # We'll use a simple numerical ODE solver.
    # Let y = [D, D_x] where D_x = dD/dx
    # D_xx = - (3/2 + H'/H) D_x + (3/2) Om_m D
    
    # H'/H = (1/2) * d ln E^2 / d ln a
    # E^2 = Om_m a^-3 + (1-Om_m) a^(-3(1+w0)) exp(-3 wa (ln a - a + 1))
    # d ln E^2 / d ln a = (1/E^2) * dE^2/dln a
    # dE^2/dln a = -3 Om_m a^-3 + (1-Om_m) * d/da [a^(-3(1+w0)) exp(-3wa(ln a - a + 1))] * a
    # Let g(a) = a^(-3(1+w0)) exp(-3wa(ln a - a + 1))
    # d g/da = g * [-3(1+w0)/a + (-3wa)(1/a - 1)]
    # = g * [-3(1+w0)/a - 3wa/a + 3wa]
    # = g * [-3(1+w0+wa)/a + 3wa]
    # So dE^2/dln a = -3 Om_m a^-3 + (1-Om_m) * g * a * [-3(1+w0+wa)/a + 3wa]
    # = -3 Om_m a^-3 + (1-Om_m) * g * [-3(1+w0+wa) + 3wa a]
    
    # Let's implement the ODE solver.
    from scipy.integrate import solve_ivp
    
    def ode(x, y):
        a = np.exp(x)
        Om = _Om(a, Om0, w0, wa)
        if np.isnan(Om):
            return [0, 0]
        
        # Compute H'/H
        E2 = Om0 * a**(-3) + (1 - Om0) * a**(-3*(1+w0)) * np.exp(-3*wa*(np.log(a) - a + 1))
        if E2 <= 0:
            return [0, 0]
        g = (1 - Om0) * a**(-3*(1+w0)) * np.exp(-3*wa*(np.log(a) - a + 1))
        dE2_dlna = -3 * Om0 * a**(-3) + g * (-3*(1+w0+wa) + 3*wa*a)
        Hprime_H = 0.5 * dE2_dlna / E2
        
        D, D_x = y
        D_xx = -(1.5 + Hprime_H) * D_x + 1.5 * Om * D
        return [D_x, D_xx]
    
    # Initial conditions at a=1 (x=0): D(1)=1, D_x(1) = f(1) = Om0^gamma
    # We need an initial estimate for f(1). Use gamma ~ 0.55.
    f1_init = Om0 ** 0.55
    
    sol = solve_ivp(ode, [0, np.log(0.05)], [1.0, f1_init], 
                    method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    
    if not sol.success:
        return np.full(n_points, np.nan), np.full(n_points, np.nan)
    
    f_vals = np.zeros(n_points)
    D_ratio = np.zeros(n_points)
    
    for i, a_i in enumerate(a_arr):
        x_i = np.log(a_i)
        y_i = sol.sol(x_i)
        D_i = y_i[0]
        D_x_i = y_i[1]
        
        # f = d ln D / d ln a = D_x / D
        if abs(D_i) < 1e-10:
            f_vals[i] = np.nan
            D_ratio[i] = np.nan
            continue
        
        f_vals[i] = D_x_i / D_i
        D_ratio[i] = D_i  # D(1) = 1, so D(a)/D(1) = D(a)
    
    return f_vals, D_ratio


def _objective(coeffs, models):
    """Compute the maximum relative error across all models."""
    max_err = 0.0
    
    for Om0, w0, wa in models:
        f_approx, D_approx = _f_and_D(coeffs, np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9]), Om0, w0, wa, n_points=6)
        f_true, D_true = _true_f_and_D(np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9]), Om0, w0, wa, n_points=6)
        
        for i in range(len(f_approx)):
            if not (np.isnan(f_approx[i]) or np.isnan(f_true[i])) and abs(f_true[i]) > 1e-10:
                err_f = abs(f_approx[i] - f_true[i]) / abs(f_true[i])
                max_err = max(max_err, err_f)
            
            if not (np.isnan(D_approx[i]) or np.isnan(D_true[i])) and abs(D_true[i]) > 1e-10:
                err_D = abs(D_approx[i] - D_true[i]) / abs(D_true[i])
                max_err = max(max_err, err_D)
    
    return max_err


def construct():
    """Find optimal coefficients for the growth index formula."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Validation models (typical cosmological parameters)
    models = [
        (0.3, 0.0, 0.0),    # Lambda CDM
        (0.3, -0.5, 0.0),   # Quintessence
        (0.3, -1.0, 0.5),   # CPL with positive wa
        (0.3, -1.0, -0.5),  # CPL with negative wa
        (0.25, -1.0, 0.0),  # Different Om0
        (0.35, -0.9, 0.3),  # Another CPL
    ]
    
    # Initial guess: constant gamma = 0.55
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Bounds for coefficients
    bounds = [(-10, 10)] * 6
    
    best_x = x0.copy()
    best_err = _objective(x0, models)
    
    # Use Nelder-Mead optimization
    try:
        result = minimize(_objective, x0, args=(models,), method='Nelder-Mead',
                         bounds=bounds, options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if time.time() < deadline:
            if result.fun < best_err:
                best_x = result.x
                best_err = result.fun
    except Exception:
        pass
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for _ in range(5):
        if time.time() >= deadline:
            break
        
        x_rand = np.random.uniform(-2, 2, 6)
        x_rand[0] = np.clip(x_rand[0], 0.1, 1.2)  # Keep c0 in reasonable range
        
        try:
            result = minimize(_objective, x_rand, args=(models,), method='Nelder-Mead',
                             bounds=bounds, options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result.fun < best_err:
                best_x = result.x
                best_err = result.fun
        except Exception:
            pass
    
    # Ensure coefficients are within bounds and gamma constraint is satisfied
    best_x = np.clip(best_x, -10, 10)
    
    return list(best_x)

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    def om(a, Om0, w0, wa):
        """Matter density parameter at scale factor a."""
        a = np.asarray(a, dtype=float)
        de = (1 - Om0) * a**(-3*(1+w0)) * np.exp(3*wa*(a-1))
        E2 = Om0 * a**(-3) + de
        return Om0 * a**(-3) / E2
    
    def gamma_from_coeffs(c, a, Om0, w0, wa):
        a = np.asarray(a, dtype=float)
        om_a = om(a, Om0, w0, wa)
        q = 1 - om_a
        u = 1 + w0 + wa*(1-a)
        return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
    
    def f_approx(c, a, Om0, w0, wa):
        a = np.asarray(a, dtype=float)
        g = gamma_from_coeffs(c, a, Om0, w0, wa)
        om_a = om(a, Om0, w0, wa)
        return om_a ** g
    
    def D_approx(c, a, Om0, w0, wa):
        """Normalized growth factor D(a)/D(1) via numerical integration."""
        a = np.asarray(a, dtype=float)
        result = np.ones_like(a)
        for i, ai in enumerate(a):
            if ai >= 0.999:
                result[i] = 1.0
                continue
            # Use fixed-step integration for speed
            n_steps = 50
            x_vals = np.linspace(0, np.log(ai), n_steps+1)
            f_vals = np.array([f_approx(c, np.exp(x), Om0, w0, wa)[0] for x in x_vals])
            integral = np.trapz(f_vals, x_vals)
            result[i] = np.exp(-integral)
        return result
    
    def D_ode(a, Om0, w0, wa):
        """Exact normalized growth from ODE."""
        a = np.asarray(a, dtype=float)
        result = np.ones_like(a)
        for i, ai in enumerate(a):
            if ai >= 0.999:
                result[i] = 1.0
                continue
            
            def ode(x, y):
                a_x = np.exp(x)
                om_x = om(a_x, Om0, w0, wa)[0]
                f_x = y[0]
                D_x = y[1]
                dfdx = -f_x**2 - (3/2)*(1-om_x)*f_x + 3/2
                dDdx = f_x * D_x
                return [dfdx, dDdx]
            
            x0 = 0.0
            x1 = np.log(ai)
            om1 = om(1.0, Om0, w0, wa)[0]
            y0 = [om1**0.55, 1.0]
            
            sol = solve_ivp(ode, [x0, x1], y0, rtol=1e-8, atol=1e-10)
            if sol.success and len(sol.y[1]) > 0:
                result[i] = sol.y[1, -1]
            else:
                result[i] = 1.0
        return result
    
    # Validation cosmologies - keep close to LCDM to match the scoring regime
    test_models = [
        (0.3, -1.0, 0.0),   # LCDM
        (0.3, -1.0, 0.3),   # CPL with moderate wa
        (0.3, -0.8, -0.3),  # quintessence-like
        (0.25, -1.0, 0.2),  # lower Om0
        (0.35, -1.2, 0.3),  # more negative w0
        (0.3, -1.1, 0.5),   # larger wa
    ]
    a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    # Pre-compute reference values for speed
    f_refs = {}
    D_refs = {}
    for Om0, w0, wa in test_models:
        key = (Om0, w0, wa)
        f_refs[key] = {}
        D_refs[key] = {}
        for a in a_values:
            f_refs[key][a] = D_ode(a, Om0, w0, wa)[0]  # placeholder, will fix below
        # Actually need f and D separately
        D_refs[key] = D_ode(a_values, Om0, w0, wa)
        # For f, we need the ODE solution's f component
        for i, a in enumerate(a_values):
            if a >= 0.999:
                f_refs[key][a] = om(1.0, Om0, w0, wa)[0]**0.55
            else:
                def ode(x, y):
                    a_x = np.exp(x)
                    om_x = om(a_x, Om0, w0, wa)[0]
                    f_x = y[0]
                    D_x = y[1]
                    dfdx = -f_x**2 - (3/2)*(1-om_x)*f_x + 3/2
                    dDdx = f_x * D_x
                    return [dfdx, dDdx]
                x0 = 0.0
                x1 = np.log(a)
                om1 = om(1.0, Om0, w0, wa)[0]
                y0 = [om1**0.55, 1.0]
                sol = solve_ivp(ode, [x0, x1], y0, rtol=1e-8, atol=1e-10)
                if sol.success and len(sol.y[0]) > 0:
                    f_refs[key][a] = sol.y[0, -1]
                else:
                    f_refs[key][a] = om(a, Om0, w0, wa)[0]**0.55
    
    def objective(c):
        if time.time() > deadline - 3:
            return 1e6
        max_err = 0.0
        for Om0, w0, wa in test_models:
            key = (Om0, w0, wa)
            for a in a_values:
                g = gamma_from_coeffs(c, a, Om0, w0, wa)
                if g < 0.1 or g > 1.2:
                    return 1e6
                
                f_exact = f_refs[key][a]
                f_app = f_approx(c, a, Om0, w0, wa)[0]
                if abs(f_exact) > 1e-10:
                    err_f = abs(f_app - f_exact) / abs(f_exact)
                else:
                    err_f = 0.0
                
                D_exact = D_refs[key][a]
                D_app = D_approx(c, a, Om0, w0, wa)[0]
                if abs(D_exact) > 1e-10:
                    err_D = abs(D_app - D_exact) / abs(D_exact)
                else:
                    err_D = 0.0
                
                max_err = max(max_err, err_f, err_D)
        
        return 100.0 * max_err
    
    # Start from the known-good constant gamma
    c_seed = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    seed_score = objective(c_seed)
    
    best_c = c_seed
    best_score = seed_score
    
    # Try a local optimization with small steps
    try:
        # Use Nelder-Mead with tight bounds around the seed
        result = minimize(
            objective, c_seed, method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True,
                     'initial_simplex': None}
        )
        
        if result.success or hasattr(result, 'x'):
            candidate_c = np.clip(result.x, -10, 10)
            candidate_score = objective(candidate_c)
            if candidate_score < best_score:
                best_c = candidate_c
                best_score = candidate_score
        
        # Try a few more random perturbations around the best
        np.random.seed(42)
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            perturbation = np.random.normal(0, 0.05, size=6)
            candidate_c = np.clip(best_c + perturbation, -10, 10)
            candidate_score = objective(candidate_c)
            if candidate_score < best_score:
                best_c = candidate_c
                best_score = candidate_score
        
    except Exception:
        pass
    
    # If the optimized result is worse than seed, fall back to seed
    if best_score > seed_score:
        best_c = c_seed
    
    return list(best_c)

# EVOLVE-BLOCK-END

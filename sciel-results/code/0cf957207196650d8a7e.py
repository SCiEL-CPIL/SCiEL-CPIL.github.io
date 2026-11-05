"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # 80s safety margin before 90s kill
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = 1e10

    def om_a(a, Om0, w0, wa):
        """Matter density parameter at scale factor a for CPL dark energy."""
        a_safe = np.maximum(a, 1e-10)
        E2 = Om0 * a_safe**(-3) + (1 - Om0) * a_safe**(-3*(1+w0)) * np.exp(3*wa*(a_safe - 1))
        return Om0 * a_safe**(-3) / E2

    def solve_ode_fast(a_vals, Om0, w0, wa):
        """Solve growth ODE for multiple a values efficiently."""
        if len(a_vals) == 0:
            return np.array([]), np.array([])
        
        a_min = max(min(a_vals), 1e-6)
        a_max = max(a_vals)
        
        def ode(x, y):
            a_x = np.exp(x)
            om_x = om_a(a_x, Om0, w0, wa)
            f_x = y[0]
            D_x = y[1]
            dfdx = -f_x**2 - 1.5*(1 - om_x)*f_x + 1.5
            dDdx = f_x * D_x
            return [dfdx, dDdx]
        
        x_start = np.log(a_min)
        x_end = 0.0
        om_1 = om_a(1.0, Om0, w0, wa)
        y0 = [om_1**0.55, 1.0]
        
        sol = solve_ivp(ode, [x_start, x_end], y0, rtol=1e-8, atol=1e-10, 
                       t_eval=np.log(a_vals), dense_output=True)
        
        if sol.success:
            f_vals = sol.y[0]
            D_vals = sol.y[1]
            # Normalize D to D(1)=1
            D_vals = D_vals / D_vals[-1]
            return f_vals, D_vals
        else:
            return None, None

    def gamma_vec(c, a_vals, Om0, w0, wa):
        """Compute gamma for vector of a values."""
        om = om_a(a_vals, Om0, w0, wa)
        q = 1 - om
        u = 1 + w0 + wa*(1 - a_vals)
        return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a_vals + c[5]*u*q

    def f_approx_vec(c, a_vals, Om0, w0, wa):
        """Approximate f for vector of a values."""
        g = gamma_vec(c, a_vals, Om0, w0, wa)
        om = om_a(a_vals, Om0, w0, wa)
        return np.power(om, g)

    def D_approx_vec(c, a_vals, Om0, w0, wa):
        """Approximate normalized D for vector of a values."""
        if len(a_vals) == 0:
            return np.array([])
        
        # Integrate f from ln(a) to 0
        a_sorted = np.sort(a_vals)
        a_sorted = np.maximum(a_sorted, 1e-6)
        
        # Use quadrature on log scale
        x_vals = np.linspace(np.log(a_sorted[0]), 0, max(200, len(a_sorted)*10))
        a_interp = np.exp(x_vals)
        f_interp = f_approx_vec(c, a_interp, Om0, w0, wa)
        
        # Integrate -f dx from x to 0
        integral = np.zeros(len(a_sorted))
        for i, a_i in enumerate(a_sorted):
            x_i = np.log(a_i)
            mask = x_vals >= x_i
            if np.sum(mask) > 1:
                integral[i] = np.trapz(f_interp[mask], x_vals[mask])
            else:
                integral[i] = 0.0
        
        D_vals = np.exp(-integral)
        
        # Map back to original order
        sort_idx = np.argsort(a_vals)
        D_original = np.empty_like(D_vals)
        D_original[sort_idx] = D_vals
        return D_original

    def objective(c):
        """Compute 100 * max relative error with smooth soft-max."""
        if time.time() > deadline - 5:
            return 1e10
        
        max_err = 0.0
        err_terms = []
        
        for Om0, w0, wa in test_models:
            f_exact, D_exact = solve_ode_fast(a_values, Om0, w0, wa)
            if f_exact is None:
                continue
            
            f_app = f_approx_vec(c, a_values, Om0, w0, wa)
            D_app = D_approx_vec(c, a_values, Om0, w0, wa)
            
            # Check gamma bounds
            g = gamma_vec(c, a_values, Om0, w0, wa)
            if np.any(g < 0.1) or np.any(g > 1.2):
                return 1e10
            
            # Relative errors
            err_f = np.abs(f_app - f_exact) / np.maximum(np.abs(f_exact), 1e-10)
            err_D = np.abs(D_app - D_exact) / np.maximum(np.abs(D_exact), 1e-10)
            
            err_terms.extend(err_f.tolist())
            err_terms.extend(err_D.tolist())
        
        if not err_terms:
            return 1e10
        
        err_arr = np.array(err_terms)
        # Soft-max for smoothness
        beta = 20.0
        log_max = np.log(np.sum(np.exp(beta * err_arr))) / beta
        return 100.0 * log_max

    # Validation cosmologies
    test_models = [
        (0.30, -1.0, 0.0),
        (0.30, -1.0, 0.5),
        (0.30, -0.5, -0.5),
        (0.25, -1.0, 0.3),
        (0.35, -1.2, 0.4),
        (0.30, -0.8, -0.3),
    ]
    a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # Initial guess
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    bounds = [(-10, 10)] * 6

    try:
        # Use differential evolution for global search, then polish with L-BFGS-B
        # This is a structural change from the previous L-BFGS-B + Nelder-Mead approach
        
        # First, use differential evolution for global exploration
        de_result = differential_evolution(
            objective, bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-6,
            popsize=15,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=False,
            updating='immediate'
        )
        
        if time.time() < deadline - 20:
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_c = de_result.x
        
        # Polish with L-BFGS-B
        if time.time() < deadline - 10:
            result = minimize(objective, best_c, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
            
            if time.time() < deadline - 5 and result.success:
                score = result.fun
                if score < best_score:
                    best_score = score
                    best_c = result.x
        
        # Final polish with Nelder-Mead if time remains
        if time.time() < deadline - 5:
            result2 = minimize(objective, best_c, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if time.time() < deadline - 2 and result2.success:
                score2 = result2.fun
                if score2 < best_score:
                    best_score = score2
                    best_c = result2.x
        
        best_c = np.clip(best_c, -10, 10)
        return list(best_c)
        
    except Exception:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

# EVOLVE-BLOCK-END

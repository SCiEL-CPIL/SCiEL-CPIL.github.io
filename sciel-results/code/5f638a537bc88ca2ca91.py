import math
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    # ---------- Reference Cosmology Functions ----------
    
    def om_m_of_a(a, Om, w0, wa):
        """Compute Omega_m(a) for CPL dark energy."""
        h2 = Om * a ** (-3.0) + (1.0 - Om) * np.exp(3.0 * wa * (1.0 - a))
        return Om * a ** (-3.0) / h2

    def growth_ode_rhs(x, f, Om, w0, wa):
        """
        First-order ODE for f = d ln D / d ln a.
        df/dx = -f^2 - f * [3/2*(1+Om_a) + d(1+w)/dx] + 3/2 * Om_a
        where x = ln a, d(1+w)/dx = -wa * a
        """
        a = math.exp(x)
        w = w0 + wa * (1.0 - a)
        Om_a = om_m_of_a(a, Om, w0, wa)
        dw_dx = -wa * a  # d(1+w)/dx = d(w0+1+wa-wa*a)/dx = -wa * da/dx = -wa * a
        
        coeff = 1.5 * (1.0 + Om_a) + dw_dx
        return -f**2 - f * coeff + 1.5 * Om_a

    def integrate_growth_f(Om, w0, wa, x_grid):
        """
        Integrate the first-order ODE for f from x_grid[0] to x_grid[-1]=0.
        Returns f values on x_grid.
        """
        x0 = x_grid[0]
        a0 = math.exp(x0)
        Om_a0 = om_m_of_a(a0, Om, w0, wa)
        f0 = Om_a0 ** 0.55  # Initial condition: matter-dominated approximation
        
        t_span = (x0, 0.0)
        sol = solve_ivp(growth_ode_rhs, t_span, [f0], t_eval=x_grid,
                       args=(Om, w0, wa), rtol=1e-10, atol=1e-12,
                       method='RK45', max_step=0.05)
        
        if not sol.success:
            return None
        return sol.y[0]

    def compute_D_from_f(f_vals, x_grid):
        """
        Compute D(a)/D(1) = exp(-int_{ln a}^{0} f(x) dx) from f values.
        x_grid is ascending (negative to 0).
        """
        if len(x_grid) == 1:
            return np.array([1.0])
        
        # We need integral from x_i to 0 for each i.
        # Since x_grid is ascending and x_grid[-1] = 0,
        # integral from x_i to 0 = sum of trapezoids from i to end.
        
        dx = np.diff(x_grid)  # positive
        trap = 0.5 * (f_vals[:-1] + f_vals[1:]) * dx
        
        # cum[j] = integral from x_j to 0
        cum = np.zeros(len(x_grid))
        for j in range(len(x_grid) - 2, -1, -1):
            cum[j] = cum[j + 1] + trap[j]
        
        return np.exp(-cum)

    # ---------- Model Ansätze ----------
    
    def model_f(c, a, Om, w0, wa):
        c0, c1, c2, c3, c4, c5 = c
        Om_a = om_m_of_a(a, Om, w0, wa)
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa * (1.0 - a)
        gamma = c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q
        gamma = np.clip(gamma, 0.01, 10.0)  # Avoid numerical issues
        return Om_a ** gamma

    def model_D(c, a_grid, Om, w0, wa):
        f_vals = model_f(c, a_grid, Om, w0, wa)
        x_grid = np.log(a_grid)
        return compute_D_from_f(f_vals, x_grid)

    # ---------- Grids and Reference Data ----------
    
    # Focused cosmology grid for fitting
    Om_grid = np.array([0.20, 0.25, 0.30, 0.35, 0.40])
    w0_grid = np.array([-1.5, -1.2, -1.0, -0.8, -0.5, 0.0])
    wa_grid = np.array([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5])
    
    # Dense grid for integration
    a_int = np.linspace(0.05, 1.0, 200)
    x_int = np.log(a_int)
    
    # Build reference data
    ref_data = []
    for Om in Om_grid:
        for w0 in w0_grid:
            for wa in wa_grid:
                if time.time() > DEADLINE - 15.0:
                    break
                try:
                    f_ref = integrate_growth_f(Om, w0, wa, x_int)
                    if f_ref is None:
                        continue
                    D_ref = compute_D_from_f(f_ref, x_int)
                    ref_data.append({
                        'Om': Om, 'w0': w0, 'wa': wa,
                        'f_ref': f_ref, 'D_ref': D_ref
                    })
                except Exception:
                    pass
            if time.time() > DEADLINE - 15.0:
                break
        if time.time() > DEADLINE - 15.0:
            break

    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # ---------- Objective Functions ----------
    
    def compute_errors(c, ref_entry):
        """Compute max relative error for f and D for one cosmology."""
        Om, w0, wa = ref_entry['Om'], ref_entry['w0'], ref_entry['wa']
        f_ref, D_ref = ref_entry['f_ref'], ref_entry['D_ref']
        
        f_mod = model_f(c, a_int, Om, w0, wa)
        D_mod = model_D(c, a_int, Om, w0, wa)
        
        err_f = np.abs(f_mod - f_ref) / np.maximum(np.abs(f_ref), 1e-8)
        err_D = np.abs(D_mod - D_ref) / np.maximum(np.abs(D_ref), 1e-8)
        
        # Check gamma constraint
        Om_a = om_m_of_a(a_int, Om, w0, wa)
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa * (1.0 - a_int)
        gamma = c[0] + c[1]*q + c[2]*q*q + c[3]*u + c[4]*wa*a_int + c[5]*u*q
        gamma_violation = np.any(gamma < 0.1) or np.any(gamma > 1.2)
        
        return np.max(err_f), np.max(err_D), gamma_violation

    def objective_true(c):
        """True objective: max relative error across all cosmologies."""
        max_err = 0.0
        for entry in ref_data:
            err_f, err_D, violation = compute_errors(c, entry)
            if violation:
                return 10.0  # Heavy penalty for constraint violation
            max_err = max(max_err, err_f, err_D)
        return max_err

    def objective_smooth(c, beta=30.0):
        """Smooth surrogate using log-sum-exp of errors."""
        all_errs = []
        for entry in ref_data:
            err_f, err_D, violation = compute_errors(c, entry)
            if violation:
                return 10.0
            all_errs.append(err_f)
            all_errs.append(err_D)
        
        if not all_errs:
            return 0.0
        
        all_errs = np.array(all_errs)
        max_e = np.max(all_errs)
        if max_e < 1e-10:
            return 0.0
        lse = max_e + np.log(np.sum(np.exp(all_errs - max_e))) / beta
        return lse

    # ---------- Optimization ----------
    
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_true_val = objective_true(best_c)
    
    # Phase 1: Global search with smooth objective
    starts = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.50, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.60, -0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.05, -0.05, 0.0, 0.0, 0.0],
        [0.50, 0.0, 0.0, 0.05, 0.0, 0.0],
        [0.55, 0.1, 0.1, -0.1, 0.0, 0.0],
        [0.55, -0.1, -0.1, 0.1, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.05, 0.0],
    ]
    
    for s in starts:
        if time.time() > DEADLINE - 20.0:
            break
        s = np.asarray(s)
        try:
            res = minimize(objective_smooth, s, method='L-BFGS-B',
                           bounds=[(-10, 10)]*6,
                           options={'maxiter': 300, 'gtol': 1e-5})
            if res.fun < 1.0:
                c_candidate = res.x
                val_candidate = objective_true(c_candidate)
                if val_candidate < best_true_val:
                    best_true_val = val_candidate
                    best_c = c_candidate
        except Exception:
            pass

    # Phase 2: Local polish with true objective
    polish_starts = [best_c]
    for i in range(8):
        if time.time() > DEADLINE - 8.0:
            break
        noise = np.random.normal(0, 0.03, 6)
        polish_starts.append(best_c + noise)
    
    for s in polish_starts:
        if time.time() > DEADLINE - 3.0:
            break
        s = np.asarray(s)
        try:
            res = minimize(objective_true, s, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-7,
                                   'adaptive': True})
            if res.fun < best_true_val:
                best_true_val = res.fun
                best_c = res.x
        except Exception:
            pass

    # Final clip and constraint verification
    best_c = np.clip(best_c, -10.0, 10.0)
    
    # Verify constraints on a test grid
    valid = True
    for Om in [0.2, 0.3, 0.4]:
        for w0 in [-1.5, -1.0, 0.0]:
            for wa in [-0.5, 0.0, 0.5]:
                a_test = np.linspace(0.1, 1.0, 50)
                Om_a = om_m_of_a(a_test, Om, w0, wa)
                q = 1.0 - Om_a
                u = 1.0 + w0 + wa * (1.0 - a_test)
                g = best_c[0] + best_c[1]*q + best_c[2]*q*q + best_c[3]*u + best_c[4]*wa*a_test + best_c[5]*u*q
                if np.any(g < 0.1) or np.any(g > 1.2):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to safe constant gamma
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    return [float(x) for x in best_c]

# EVOLVE-BLOCK-END

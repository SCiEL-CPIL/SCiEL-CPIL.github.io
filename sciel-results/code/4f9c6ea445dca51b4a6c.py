"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = 1e10

    # Test cosmologies (kept same as before)
    test_models = [
        (0.30, -1.0, 0.0),
        (0.30, -1.0, 0.5),
        (0.30, -0.5, -0.5),
        (0.25, -1.0, 0.3),
        (0.35, -1.2, 0.4),
        (0.30, -0.8, -0.3),
    ]
    a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    def om_a(a, Om0, w0, wa):
        a_safe = np.maximum(a, 1e-10)
        E2 = Om0 * a_safe**(-3) + (1 - Om0) * a_safe**(-3*(1+w0)) * np.exp(3*wa*(a_safe - 1))
        return Om0 * a_safe**(-3) / E2

    def rk4_ode(x0, x1, y0, nsteps, Om0, w0, wa):
        """Fast RK4 integrator for the growth ODE in log(a) space."""
        dx = (x1 - x0) / nsteps
        y = np.array(y0, dtype=float)
        x = x0
        for _ in range(nsteps):
            a_x = np.exp(x)
            om_x = om_a(a_x, Om0, w0, wa)
            
            def rhs(yv):
                f_x = yv[0]
                D_x = yv[1]
                dfdx = -f_x*f_x - 1.5*(1 - om_x)*f_x + 1.5
                dDdx = f_x * D_x
                return np.array([dfdx, dDdx])
            
            k1 = rhs(y)
            k2 = rhs(y + 0.5*dx*k1)
            k3 = rhs(y + 0.5*dx*k2)
            k4 = rhs(y + dx*k3)
            y = y + (dx/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            x += dx
        return y

    def solve_ode_rk4(a_vals, Om0, w0, wa):
        """Solve growth ODE using RK4 with fixed step size."""
        if len(a_vals) == 0:
            return np.array([]), np.array([])
        
        a_min = max(min(a_vals), 1e-6)
        x_start = np.log(a_min)
        x_end = 0.0
        
        # Use enough steps for accuracy
        nsteps = 2000
        om_1 = om_a(1.0, Om0, w0, wa)
        y0 = [om_1**0.55, 1.0]
        
        # Solve from x_start to x_end
        # We need values at each a in a_vals
        a_sorted = np.sort(a_vals)
        x_targets = np.log(a_sorted)
        
        # Integrate from x_start to x_end, collecting values at targets
        # Use adaptive stepping: integrate segment by segment
        x_current = x_start
        y_current = np.array(y0, dtype=float)
        
        # Create a sorted list of target x values (including start and end)
        all_x = np.sort(np.concatenate([[x_start], x_targets, [x_end]]))
        
        f_vals = np.zeros(len(a_sorted))
        D_vals = np.zeros(len(a_sorted))
        
        # Map from target index to sorted position
        target_indices = []
        for i, xt in enumerate(x_targets):
            # Find position in all_x
            idx = np.searchsorted(all_x, xt)
            target_indices.append(idx)
        
        # Integrate segment by segment
        for seg in range(len(all_x) - 1):
            x0_seg = all_x[seg]
            x1_seg = all_x[seg + 1]
            if x1_seg - x0_seg < 1e-12:
                continue
            nsteps_seg = max(10, int((x1_seg - x0_seg) * 2000))
            y_end = rk4_ode(x0_seg, x1_seg, y_current.tolist(), nsteps_seg, Om0, w0, wa)
            y_current = y_end
            x_current = x1_seg
        
        # Now we have y at x_end. We need to re-integrate to get values at each target.
        # Better approach: integrate once and store values at targets
        # Let's redo with a single pass storing at targets
        
        x_current = x_start
        y_current = np.array(y0, dtype=float)
        
        # Build sorted unique target x values
        unique_targets = np.unique(np.concatenate([x_targets]))
        
        f_at_targets = {}
        D_at_targets = {}
        
        # Integrate from x_start to x_end, storing at each target
        all_points = np.sort(np.unique(np.concatenate([[x_start], x_targets, [x_end]])))
        
        for seg in range(len(all_points) - 1):
            x0_seg = all_points[seg]
            x1_seg = all_points[seg + 1]
            if x1_seg - x0_seg < 1e-12:
                continue
            nsteps_seg = max(5, int((x1_seg - x0_seg) * 1500))
            y_end = rk4_ode(x0_seg, x1_seg, y_current.tolist(), nsteps_seg, Om0, w0, wa)
            
            # Store if this is a target point
            for ti, xt in enumerate(x_targets):
                if abs(x1_seg - xt) < 1e-10:
                    f_at_targets[ti] = y_end[0]
                    D_at_targets[ti] = y_end[1]
            
            y_current = y_end
        
        # Also check if any targets were at x_start
        for ti, xt in enumerate(x_targets):
            if abs(x_start - xt) < 1e-10:
                f_at_targets[ti] = y0[0]
                D_at_targets[ti] = y0[1]
        
        # Fill in arrays
        for ti in range(len(a_sorted)):
            if ti in f_at_targets:
                f_vals[ti] = f_at_targets[ti]
                D_vals[ti] = D_at_targets[ti]
            else:
                # Should not happen, but fallback
                f_vals[ti] = om_a(a_sorted[ti], Om0, w0, wa)**0.55
                D_vals[ti] = 1.0
        
        # Normalize D to D(1)=1
        # Find which index corresponds to a=1 (or closest)
        idx_1 = np.argmin(np.abs(a_sorted - 1.0))
        D_vals = D_vals / D_vals[idx_1]
        
        # Map back to original order
        sort_idx = np.argsort(a_vals)
        f_original = np.empty_like(f_vals)
        D_original = np.empty_like(D_vals)
        f_original[sort_idx] = f_vals
        D_original[sort_idx] = D_vals
        
        return f_original, D_original

    def gamma_vec(c, a_vals, Om0, w0, wa):
        om = om_a(a_vals, Om0, w0, wa)
        q = 1 - om
        u = 1 + w0 + wa*(1 - a_vals)
        return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a_vals + c[5]*u*q

    def f_approx_vec(c, a_vals, Om0, w0, wa):
        g = gamma_vec(c, a_vals, Om0, w0, wa)
        om = om_a(a_vals, Om0, w0, wa)
        return np.power(om, g)

    def D_approx_vec(c, a_vals, Om0, w0, wa):
        if len(a_vals) == 0:
            return np.array([])
        
        a_sorted = np.sort(a_vals)
        a_sorted = np.maximum(a_sorted, 1e-6)
        
        # Use dense quadrature on log scale
        n_quad = 400
        x_vals = np.linspace(np.log(a_sorted[0]), 0, n_quad)
        a_interp = np.exp(x_vals)
        f_interp = f_approx_vec(c, a_interp, Om0, w0, wa)
        
        # Cumulative integral from right
        integral = np.zeros(n_quad)
        integral[-1] = 0.0
        for i in range(n_quad - 2, -1, -1):
            integral[i] = integral[i+1] + 0.5*(f_interp[i] + f_interp[i+1])*(x_vals[i+1] - x_vals[i])
        
        # Interpolate to get integral at each a_sorted
        D_vals = np.zeros(len(a_sorted))
        for i, a_i in enumerate(a_sorted):
            x_i = np.log(a_i)
            idx = np.searchsorted(x_vals, x_i)
            if idx == 0:
                D_vals[i] = 1.0
            elif idx >= n_quad:
                D_vals[i] = 1.0
            else:
                frac = (x_i - x_vals[idx-1]) / (x_vals[idx] - x_vals[idx-1])
                int_val = (1-frac)*integral[idx-1] + frac*integral[idx]
                D_vals[i] = np.exp(-int_val)
        
        sort_idx = np.argsort(a_vals)
        D_original = np.empty_like(D_vals)
        D_original[sort_idx] = D_vals
        return D_original

    def objective(c):
        if time.time() > deadline - 5:
            return 1e10
        
        max_err = 0.0
        
        for Om0, w0, wa in test_models:
            f_exact, D_exact = solve_ode_rk4(a_values, Om0, w0, wa)
            if f_exact is None or len(f_exact) == 0:
                continue
            
            f_app = f_approx_vec(c, a_values, Om0, w0, wa)
            D_app = D_approx_vec(c, a_values, Om0, w0, wa)
            
            g = gamma_vec(c, a_values, Om0, w0, wa)
            if np.any(g < 0.1) or np.any(g > 1.2):
                return 1e10
            
            err_f = np.abs(f_app - f_exact) / np.maximum(np.abs(f_exact), 1e-10)
            err_D = np.abs(D_app - D_exact) / np.maximum(np.abs(D_exact), 1e-10)
            
            max_err = max(max_err, np.max(err_f), np.max(err_D))
        
        return 100.0 * max_err

    try:
        # Step 1: Coarse grid search over c0 (most important parameter)
        # and small perturbations of other coefficients
        c0_grid = np.linspace(0.45, 0.65, 9)
        c1_grid = np.linspace(-0.2, 0.2, 5)
        c2_grid = np.linspace(-0.1, 0.1, 3)
        
        best_c_grid = best_c.copy()
        best_score_grid = best_score
        
        for c0v in c0_grid:
            for c1v in c1_grid:
                for c2v in c2_grid:
                    if time.time() > deadline - 40:
                        break
                    c_test = np.array([c0v, c1v, c2v, 0.0, 0.0, 0.0])
                    score = objective(c_test)
                    if score < best_score_grid:
                        best_score_grid = score
                        best_c_grid = c_test.copy()
        
        best_c = best_c_grid
        best_score = best_score_grid
        
        # Step 2: Nelder-Mead polish from best grid point
        if time.time() < deadline - 15:
            result = minimize(objective, best_c, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if result.fun < best_score:
                best_score = result.fun
                best_c = result.x
        
        # Step 3: Try a few different starting points for Nelder-Mead
        if time.time() < deadline - 10:
            starts = [
                np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0]),
                np.array([0.50, 0.1, -0.05, 0.0, 0.0, 0.0]),
                np.array([0.60, -0.1, 0.05, 0.0, 0.0, 0.0]),
            ]
            for s in starts:
                if time.time() > deadline - 5:
                    break
                res = minimize(objective, s, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                if res.fun < best_score:
                    best_score = res.fun
                    best_c = res.x
        
        best_c = np.clip(best_c, -10, 10)
        return list(best_c)
        
    except Exception as e:
        print(f"Exception in construct: {e}")
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

# EVOLVE-BLOCK-END

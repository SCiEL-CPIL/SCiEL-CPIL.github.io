"""Improved growth-index fit with min-max optimization and vectorized checks."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    """Fit six coefficients to minimize max relative error via iterative reweighting."""
    t_start = time.time()
    deadline = t_start + 85.0  # 5s safety margin before 90s kill

    # Grid of CPL parameters (Om0, w0, wa) - focus on physically reasonable range
    Om0_vals = np.array([0.20, 0.24, 0.28, 0.30, 0.32, 0.35])
    w0_vals = np.array([-1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7])
    wa_vals = np.array([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])

    # Scale factor grid for fitting - use more points near early times where gamma varies most
    a_vals = np.concatenate([
        np.linspace(0.05, 0.3, 25),
        np.linspace(0.3, 0.99, 35)
    ])

    # Build design matrix and target - vectorized for speed
    rows = []
    targets = []

    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                a_cubed = a_vals**3
                de_factor = a_vals**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a_vals))
                E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                
                valid = (E2 > 1e-12) & (E2 > 0)
                if not np.any(valid):
                    continue
                Om_a = np.where(valid, Om0 * a_cubed / E2, 0.0)
                q = 1.0 - Om_a
                u = 1.0 + w0 + wa * (1.0 - a_vals)
                
                valid &= (u > 0.01)
                if not np.any(valid):
                    continue
                
                # Linder-Van de Stark: gamma = (5/7) * (Om(a)/u)^(4/7)
                gamma_target = (5.0/7.0) * (Om_a[valid] / u[valid])**(4.0/7.0)
                
                # Design matrix rows: [1, q, q^2, u, wa*a, u*q]
                qv = q[valid]
                uv = u[valid]
                av = a_vals[valid]
                wa_a = wa * av
                
                row_batch = np.column_stack([
                    np.ones_like(qv),
                    qv,
                    qv**2,
                    uv,
                    wa_a,
                    uv * qv
                ])
                rows.append(row_batch)
                targets.append(gamma_target)
                
                if time.time() > deadline - 10.0:
                    break
            if time.time() > deadline - 10.0:
                break
        if time.time() > deadline - 10.0:
            break

    if len(rows) < 10:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    A = np.vstack(rows)
    b = np.concatenate(targets)

    # Initial guess: use the known Linder form gamma ≈ (5/7) * (Om/u)^(4/7)
    # which is approximately c0 + c1*q + c2*q^2 with c0≈0.55, c1≈-0.1, c2≈0.05
    coeffs = np.array([0.55, -0.1, 0.05, 0.0, 0.0, 0.0])

    # Iterative min-max approximation via exponential reweighting
    # This converges to a solution that minimizes the maximum absolute error
    for iteration in range(8):
        # Compute current errors
        pred = A @ coeffs
        errors = np.abs(pred - b)
        
        # Exponential reweighting: w_i = exp(errors_i / tau)
        # Small tau focuses on worst points, large tau is uniform
        tau = max(0.01, 0.5 * np.median(errors))
        w = np.exp(errors / tau)
        
        # Normalize weights to avoid numerical issues
        w = w / np.sum(w)
        
        # Weighted least squares
        Aw = A * w[:, None]
        bw = b * w
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(Aw, bw, rcond=None)
        except Exception:
            break
        
        # Clip to [-10, 10]
        coeffs = np.clip(coeffs, -10.0, 10.0)
        
        if time.time() > deadline - 5.0:
            break

    # Vectorized gamma range check
    def check_gamma_range(c):
        # Use a subset of the fitting grid for speed
        check_a = np.linspace(0.05, 0.99, 40)
        for Om0 in Om0_vals:
            for w0 in w0_vals:
                for wa in wa_vals:
                    a_cubed = check_a**3
                    de_factor = check_a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-check_a))
                    E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                    valid = E2 > 1e-12
                    if not np.any(valid):
                        continue
                    Om_a = np.where(valid, Om0 * a_cubed / E2, 0.0)
                    q = 1.0 - Om_a
                    u = 1.0 + w0 + wa * (1.0 - check_a)
                    valid &= (u > 0.01)
                    if not np.any(valid):
                        continue
                    gamma = (c[0] + c[1]*q[valid] + c[2]*q[valid]**2 + 
                             c[3]*u[valid] + c[4]*wa*check_a[valid] + c[5]*u[valid]*q[valid])
                    if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                        return False
        return True

    if not check_gamma_range(coeffs):
        # Constrained optimization with penalty
        def penalty(c):
            pen = 0.0
            check_a = np.linspace(0.1, 0.99, 15)
            for Om0 in Om0_vals[:4]:
                for w0 in w0_vals[:4]:
                    for wa in wa_vals[:4]:
                        a_cubed = check_a**3
                        de_factor = check_a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-check_a))
                        E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                        valid = E2 > 1e-12
                        if not np.any(valid):
                            continue
                        Om_a = np.where(valid, Om0 * a_cubed / E2, 0.0)
                        q = 1.0 - Om_a
                        u = 1.0 + w0 + wa * (1.0 - check_a)
                        valid &= (u > 0.01)
                        if not np.any(valid):
                            continue
                        gamma = (c[0] + c[1]*q[valid] + c[2]*q[valid]**2 + 
                                 c[3]*u[valid] + c[4]*wa*check_a[valid] + c[5]*u[valid]*q[valid])
                        below = np.maximum(0.1 - gamma, 0)
                        above = np.maximum(gamma - 1.2, 0)
                        pen += np.sum(below**2) + np.sum(above**2)
            return pen
        
        def obj_constrained(c):
            return np.sum((A @ c - b)**2) + 100.0 * penalty(c)
        
        bounds = [(-10, 10)] * 6
        try:
            result = minimize(obj_constrained, coeffs, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-10})
            if result.success:
                coeffs = np.clip(result.x, -10.0, 10.0)
        except Exception:
            pass
        
        if not check_gamma_range(coeffs):
            return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Final polish: direct local search on max relative error
    if time.time() < deadline - 3.0:
        def max_rel_error(c):
            pred = A @ c
            rel_errors = np.abs(pred - b) / np.maximum(np.abs(b), 1e-10)
            return np.max(rel_errors)
        
        def max_rel_error_obj(c):
            return max_rel_error(c)
        
        bounds = [(-10, 10)] * 6
        try:
            result = minimize(max_rel_error_obj, coeffs, method='Nelder-Mead', bounds=bounds,
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.success and check_gamma_range(result.x):
                coeffs = np.clip(result.x, -10.0, 10.0)
        except Exception:
            pass

    return [float(c) for c in coeffs]
# EVOLVE-BLOCK-END

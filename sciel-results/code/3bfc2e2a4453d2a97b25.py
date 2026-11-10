"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, dual_annealing
import time

def _cpl_E(z, Om, w0, wa):
    """Compute E(z) = H(z)/H0 for CPL dark energy in flat universe."""
    Ol = 1.0 - Om
    # E^2 = Om*(1+z)^3 + Ol*(1+z)^(3*w0) * exp(-3*wa*(1 - 1/(1+z)))
    if z < 1e-10:
        return 1.0
    a = 1.0 / (1.0 + z)
    e2 = Om * (1.0/a)**3 + Ol * (1.0/a)**(3.0*w0) * np.exp(-3.0*wa*(1.0 - a))
    return np.sqrt(np.maximum(e2, 1e-30))

def _reference_integral(Om, w0, wa, npts=20000):
    """High-precision reference: integrate (1+z)/E(z) dz from 0 to zmax."""
    # We integrate over the variable t in [0,1] where z = t, weight 1
    # Actually the formula is: dL/(c/H0) = (1+z)*z * integral_0^z dz'/E(z')
    # The quadrature approximates: integral_0^z dz'/E(z') ≈ sum_i w_i / E(a_i) * z
    # where a_i = 1/(1+z*x_i), so x_i are fractions of z.
    # So we need: for given z, integral_0^z dz'/E(z') 
    # Quadrature: z * sum_i w_i / E(1/(1+z*x_i))
    # We want this to match the true integral for many z values.
    # 
    # The integrand in the quadrature is f(t) = 1/E(1/(1+z*t)) and we integrate t from 0 to 1,
    # then multiply by z. So: integral_0^z dz'/E(z') = z * integral_0^1 dt / E(1/(1+z*t))
    # The quadrature approximates integral_0^1 dt / E(1/(1+z*t)) ≈ sum_i w_i / E(1/(1+z*x_i))
    # 
    # So the reference for a given z is:
    # true = z * ∫_0^1 dt / E(1/(1+z*t))
    # approx = z * sum_i w_i / E(1/(1+z*x_i))
    # 
    # Relative error = |approx - true| / |true| * 100
    pass

def _compute_relative_errors(x, w, models, z_vals):
    """Compute worst-case relative error (%) across all models and z values."""
    max_err = 0.0
    for (Om, w0, wa) in models:
        for z in z_vals:
            # True integral via fine Simpson
            ts = np.linspace(0, 1, 2001)
            a_vals = 1.0 / (1.0 + z * ts)
            E_vals = _cpl_E(1.0/a_vals - 1.0, Om, w0, wa)
            f_vals = 1.0 / E_vals
            # Simpson's rule
            n = len(ts) - 1
            if n % 2 == 0:
                h = 1.0 / n
                integral = h/3.0 * (f_vals[0] + f_vals[-1] + 4*sum(f_vals[1:-1:2]) + 2*sum(f_vals[2:-1:2]))
            else:
                integral = np.trapz(f_vals, ts)
            true_val = z * integral
            
            # Quadrature
            a_q = 1.0 / (1.0 + z * x)
            E_q = _cpl_E(1.0/a_q - 1.0, Om, w0, wa)
            approx = z * np.sum(w * (1.0 / E_q))
            
            if abs(true_val) > 1e-15:
                rel_err = abs(approx - true_val) / abs(true_val) * 100.0
                if rel_err > max_err:
                    max_err = rel_err
    return max_err

def _objective(params):
    """Objective: worst-case relative error with positivity and ordering constraints via penalties."""
    x = params[:8]
    w = params[8:16]
    
    # Enforce ordering: x[i] < x[i+1]
    for i in range(7):
        if x[i+1] <= x[i]:
            return 1e6
    # Enforce bounds
    if x[0] <= 0 or x[-1] >= 1:
        return 1e6
    # Enforce positivity
    if np.any(w <= 0):
        return 1e6
    # Enforce sum(w) ≈ 1
    if abs(np.sum(w) - 1.0) > 1e-6:
        return 1e6 + 1000*abs(np.sum(w) - 1.0)
    
    models = [
        (0.3, -1.0, 0.0),    # ΛCDM
        (0.3, -0.5, -0.5),   # CPL1
        (0.3, -1.5, 0.5),    # CPL2
        (0.25, -1.0, 0.0),   # ΛCDM low Om
        (0.35, -1.0, 0.0),   # ΛCDM high Om
        (0.3, -0.3, -0.7),   # CPL3
        (0.3, -1.2, 0.3),    # CPL4
        (0.3, -0.8, 0.0),    # quintessence-like
    ]
    z_vals = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    return _compute_relative_errors(x, w, models, z_vals)

def _project(params):
    """Project to feasible set: enforce ordering, bounds, positivity, sum=1."""
    x = params[:8].copy()
    w = params[8:16].copy()
    
    # Enforce ordering and bounds on x
    for i in range(1, 8):
        if x[i] <= x[i-1]:
            x[i] = x[i-1] + 1e-10
    if x[0] <= 0:
        x[0] = 1e-10
    if x[-1] >= 1:
        x[-1] = 1.0 - 1e-10
    # Re-enforce ordering after bounds
    for i in range(1, 8):
        if x[i] <= x[i-1]:
            x[i] = x[i-1] + (1.0 - x[i-1] - 1e-10) / (8 - i) * 0.5
    
    # Enforce positivity and normalize
    w = np.maximum(w, 1e-12)
    w = w / np.sum(w)
    
    return np.concatenate([x, w])

def construct():
    deadline = time.time() + 75.0  # safety margin before 90s kill
    best_params = None
    best_score = np.inf
    
    # Start from midpoint rule
    n = 8
    x0 = (np.arange(n) + 0.5) / n
    w0 = np.full(n, 1.0 / n)
    params0 = np.concatenate([x0, w0])
    
    best_params = params0.copy()
    best_score = _objective(params0)
    
    # L-BFGS-B optimization with multiple random restarts
    rng = np.random.default_rng(42)
    max_iters = 0
    while time.time() < deadline:
        max_iters += 1
        # Random perturbation
        if max_iters == 1:
            p = params0.copy()
        else:
            p = best_params + rng.normal(0, 0.05, 16)
            p = _project(p)
        
        try:
            res = minimize(_objective, p, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8},
                          timeout=30)
            if res.success or res.fun < best_score:
                candidate = _project(res.x)
                score = _objective(candidate)
                if score < best_score:
                    best_score = score
                    best_params = candidate
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline:
            break
    
    # Final projection
    final = _project(best_params)
    x_final = final[:8]
    w_final = final[8:16]
    
    # Ensure strict constraints
    for i in range(1, 8):
        if x_final[i] <= x_final[i-1]:
            x_final[i] = x_final[i-1] + 1e-12
    x_final[0] = max(x_final[0], 1e-12)
    x_final[-1] = min(x_final[-1], 1.0 - 1e-12)
    w_final = np.maximum(w_final, 1e-12)
    w_final = w_final / np.sum(w_final)
    
    return {"nodes": x_final, "weights": w_final}
# EVOLVE-BLOCK-END

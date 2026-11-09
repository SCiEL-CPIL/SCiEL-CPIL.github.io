<<<<<<< SEARCH
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _time_left():
    return _DEADLINE - time.time()

def _E_cpl(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    """
    Compute E(a) = H(a)/H0 for CPL model.
    a is a numpy array.
    """
    Om_de = 1.0 - Om_m - Om_r
    # Vectorized operations
    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    cpl_factor = 1.0 + wa * (1.0 - a)
    term_de = Om_de * a**(-3.0 * (1.0 + w0 + wa)) * (cpl_factor**3)
    val = term_m + term_r + term_de
    # Prevent division by zero or negative values due to numerical issues
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)

def _true_dL_fast(z, w0, wa, num_points=2048):
    """
    Fast reference calculation of (1+z) * integral.
    Uses a dense fixed grid to avoid adaptive overhead of quad.
    """
    if z <= 0:
        return 0.0
    # Grid in z-space from 0 to z
    zs = np.linspace(0, z, num_points)
    # Integrand is 1/E(a(z'))
    a_vals = 1.0 / (1.0 + zs)
    E_vals = _E_cpl(a_vals, w0, wa)
    integrand = 1.0 / E_vals
    
    # Simple Simpson's rule or Trapezoidal? 
    # For smooth functions, Trapezoidal with 2048 points is extremely accurate.
    # Simpson's is better if we have odd number of intervals.
    integral = np.trapz(integrand, zs)
    return (1.0 + z) * integral

def _quad_dL(z, nodes, weights, w0, wa):
    """
    Approximate dL using the 8-node quadrature rule.
    """
    if z <= 0:
        return 0.0
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _objective(params):
    """
    Objective function to minimize: worst relative error.
    params: 8 node params (t), 8 weight params (v).
    """
    if _time_left() < 1.0:
        return 1e10
        
    # Decode nodes
    # t_i -> x_i sequence
    x = np.zeros(8)
    prev = 0.0
    for i in range(8):
        # Sigmoid mapping to (0, 1)
        s = 1.0 / (1.0 + np.exp(-params[i]))
        # Scale step by remaining space
        remaining = 1.0 - prev
        step = s * remaining
        prev += step
        x[i] = prev
    # Ensure strictly increasing and in (0,1)
    # The construction above guarantees 0 < x[0] < x[1] ... < x[7] < 1
    # But we might want to clip extremely close to 0 or 1 if sigmoid is flat
    x = np.clip(x, 1e-12, 1.0 - 1e-12)
    
    # Decode weights
    w_raw = np.exp(params[8:16])
    w = w_raw / np.sum(w_raw)
    
    # Test set
    # We need a representative set of cosmologies.
    # Include LCDM, wCDM, and some extreme CPL cases.
    test_cosmos = [
        (0.0, 0.0),    # LCDM
        (-1.0, 0.0),   # Lambda
        (0.5, -1.0),   # Quintessence
        (-1.5, 0.5),   # Phantom
        (-0.5, 0.5),   
        (0.3, -0.3),   
        (-2.0, 1.0),   # More extreme phantom
        (1.0, -2.0)    # More extreme quintessence
    ]
    
    # Redshifts
    # Cover a wide range
    test_zs = np.array([0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    max_err = 0.0
    for w0, wa in test_cosmos:
        for z in test_zs:
            true_val = _true_dL_fast(z, w0, wa)
            quad_val = _quad_dL(z, x, w, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                if rel_err > max_err:
                    max_err = rel_err
                    
    return max_err

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Initial guess
    # Start with Gauss-Legendre-like distribution
    # t params for nodes:
    # We want x_i roughly uniform or GL distributed.
    # Let's just start with random or simple linear spacing in t-space.
    
    # Better initial guess: Map GL nodes to t-space?
    # x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    # x_gl is in [-1, 1], map to [0, 1]: x = 0.5*(x_gl+1)
    # We need to find t such that the sequential sigmoid produces x.
    # This is an inverse problem.
    # Simpler: Just use a reasonable spread in t.
    # If t is large, s is close to 1.
    
    # Let's try to initialize t such that x_i are roughly evenly spaced.
    # x_i = i/8
    # We can solve for t_i approximately.
    # Or just use random initialization for DE, it handles it well.
    
    # Bounds for DE
    # t_i: -5 to 5 is a good range for sigmoid.
    # v_i: -5 to 5 is a good range for log-weights.
    bounds = [(-5, 5)] * 16
    
    # Differential Evolution
    # popsize=15, maxiter=50
    # This is ~750 evaluations.
    # Each evaluation is fast.
    
    best_result = None
    best_score = 1e10
    
    try:
        if _time_left() > 20:
            res = differential_evolution(
                _objective, 
                bounds, 
                seed=42, 
                maxiter=100, 
                popsize=10, 
                tol=1e-8,
                polish=False # We will polish manually
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
    except Exception as e:
        pass
    
    # Refinement
    if best_result is not None and _time_left() > 10:
        try:
            # L-BFGS-B or Nelder-Mead
            # Nelder-Mead is robust for non-smooth worst-case objectives
            res2 = minimize(
                _objective, 
                best_result, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_result = res2.x
        except Exception as e:
            pass
            
    # If DE failed, try a simple grid or fallback
    if best_result is None:
        # Fallback to Gauss-Legendre
        x_gl, w_gl = np.polynomial.legendre.leggauss(8)
        nodes = 0.5 * (x_gl + 1.0)
        weights = 0.5 * w_gl
        weights = weights / np.sum(weights)
        return {"nodes": nodes, "weights": weights}

    # Decode final result
    t = best_result[:8]
    v = best_result[8:16]
    
    x = np.zeros(8)
    prev = 0.0
    for i in range(8):
        s = 1.0 / (1.0 + np.exp(-t[i]))
        remaining = 1.0 - prev
        step = s * remaining
        prev += step
        x[i] = prev
    x = np.clip(x, 1e-12, 1.0 - 1e-12)
    
    w_raw = np.exp(v)
    w = w_raw / np.sum(w_raw)
    
    # Final validation check
    # Ensure sum of weights is 1
    w = w / np.sum(w)
    
    return {"nodes": x, "weights": w}
=======
import time
import numpy as np
from scipy.optimize import minimize
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 87.0

def _time_left():
    return _DEADLINE - time.time()

def _E_cpl_vec(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    """
    Compute E(a) = H(a)/H0 for CPL model. Vectorized.
    a: array of shape (...,)
    """
    Om_de = 1.0 - Om_m - Om_r
    a_safe = np.maximum(a, 1e-15)
    term_m = Om_m * a_safe**(-3)
    term_r = Om_r * a_safe**(-4)
    cpl_factor = 1.0 + wa * (1.0 - a_safe)
    exponent = -3.0 * (1.0 + w0 + wa)
    term_de = Om_de * a_safe**exponent * (cpl_factor**3)
    val = term_m + term_r + term_de
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)

def _precompute_true():
    """Precompute true dL values for test set."""
    test_cosmos = [
        (0.0, 0.0),
        (-1.0, 0.0),
        (0.5, -1.0),
        (-1.5, 0.5),
        (-0.5, 0.5),
        (0.3, -0.3),
        (-2.0, 1.0),
        (1.0, -2.0),
        (-0.8, 0.8),
        (0.2, 0.5),
    ]
    test_zs = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    true_vals = []
    for w0, wa in test_cosmos:
        for z in test_zs:
            zs = np.linspace(0, z, 4096)
            a_vals = 1.0 / (1.0 + zs)
            E_vals = _E_cpl_vec(a_vals, w0, wa)
            integrand = 1.0 / E_vals
            integral = np.trapz(integrand, zs)
            true_vals.append((1.0 + z) * z * integral / z if z > 0 else 0.0)
    # Actually the formula is (1+z)*integral where integral = int_0^z dz'/E(a(z'))
    # Let me recompute properly
    true_vals = []
    for w0, wa in test_cosmos:
        for z in test_zs:
            if z <= 0:
                true_vals.append(0.0)
                continue
            zs = np.linspace(0, z, 4096)
            a_vals = 1.0 / (1.0 + zs)
            E_vals = _E_cpl_vec(a_vals, w0, wa)
            integrand = 1.0 / E_vals
            integral = np.trapz(integrand, zs)
            true_vals.append((1.0 + z) * integral)
    
    true_vals = np.array(true_vals)
    return test_cosmos, test_zs, true_vals

def _fast_objective(params, test_cosmos, test_zs, true_vals):
    """
    Fast vectorized objective. params = [x_1..x_8, log_w_1..log_w_8]
    where x_i are raw node positions (will be mapped to (0,1) via cumulative)
    and log_w are log-weights.
    """
    if _time_left() < 0.5:
        return 1e10
    
    # Decode nodes: use softplus-like cumulative to ensure increasing in (0,1)
    raw_x = params[:8]
    # Map to increasing positions using cumulative sum of exponentials
    exp_raw = np.exp(raw_x)
    cum = np.cumsum(exp_raw)
    nodes = cum / cum[-1]
    # Ensure strictly in (0,1) and increasing
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    # Force strict increase
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    
    # Decode weights
    log_w = params[8:16]
    w = np.exp(log_w)
    w = w / np.sum(w)
    
    # Vectorized evaluation over all (cosmo, z) pairs
    n_cosmo = len(test_cosmos)
    n_z = len(test_zs)
    
    # For each cosmo and z, compute quadrature
    max_err = 0.0
    for ci, (w0, wa) in enumerate(test_cosmos):
        for zi, z in enumerate(test_zs):
            idx = ci * n_z + zi
            if true_vals[idx] < 1e-10:
                continue
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = _E_cpl_vec(a_vals, w0, wa)
            quad_val = (1.0 + z) * z * np.sum(w / E_vals)
            rel_err = abs(quad_val - true_vals[idx]) / true_vals[idx]
            if rel_err > max_err:
                max_err = rel_err
    
    return max_err

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 87.0
    
    # Precompute true values
    test_cosmos, test_zs, true_vals = _precompute_true()
    
    # Initial guess: Gauss-Legendre nodes mapped to our parameterization
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = 0.5 * (x_gl + 1.0)  # in (0,1)
    weights_gl = 0.5 * w_gl
    weights_gl = weights_gl / np.sum(weights_gl)
    
    # Invert: find raw_x such that cumsum(exp(raw_x))/sum = nodes_gl
    # Approximate: raw_x[i] = log(nodes_gl[i] - nodes_gl[i-1]) with nodes_gl[-1]=0
    diffs = np.diff(np.concatenate([[0], nodes_gl]))
    raw_x_init = np.log(diffs)
    log_w_init = np.log(weights_gl + 1e-15)
    params_init = np.concatenate([raw_x_init, log_w_init])
    
    best_params = params_init.copy()
    best_score = _fast_objective(best_params, test_cosmos, test_zs, true_vals)
    
    # Multi-start Nelder-Mead
    rng = np.random.RandomState(42)
    n_starts = 20
    start_interval = 87.0 / (n_starts + 5)  # rough time per start
    
    for s in range(n_starts):
        if _time_left() < 5.0:
            break
        
        if s == 0:
            p0 = params_init.copy()
        else:
            # Perturb from best found so far
            perturbation_scale = max(0.1, 1.0 / (s + 1))
            p0 = best_params + rng.randn(16) * perturbation_scale
        
        try:
            res = minimize(
                _fast_objective,
                p0,
                args=(test_cosmos, test_zs, true_vals),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if _time_left() > 5.0:
        try:
            res = minimize(
                _fast_objective,
                best_params,
                args=(test_cosmos, test_zs, true_vals),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Decode final result
    raw_x = best_params[:8]
    exp_raw = np.exp(raw_x)
    cum = np.cumsum(exp_raw)
    nodes = cum / cum[-1]
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    
    log_w = best_params[8:16]
    w = np.exp(log_w)
    w = w / np.sum(w)
    
    return {"nodes": nodes, "weights": w}
>>>>>>> REPLACE

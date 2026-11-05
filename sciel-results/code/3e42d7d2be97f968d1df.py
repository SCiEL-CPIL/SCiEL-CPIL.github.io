import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit

    def E(a, Om, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL dark energy model."""
        a = np.asarray(a, dtype=float)
        if abs(w0 + 1.0) < 1e-8:
            if abs(wa) < 1e-8:
                return np.sqrt(Om * a**(-3.0) + (1.0 - Om))
            else:
                eps = 1e-8
                w0_eff = -1.0 + eps
                exponent = -3.0 * (1.0 + w0_eff + wa)
                exp_term = 3.0 * wa * (1.0 - a) / (1.0 + w0_eff)
                exp_term = np.clip(exp_term, -500.0, 500.0)
                term2 = (1.0 - Om) * a**exponent * np.exp(exp_term)
                return np.sqrt(Om * a**(-3.0) + term2)
        else:
            exponent = -3.0 * (1.0 + w0 + wa)
            exp_term = 3.0 * wa * (1.0 - a) / (1.0 + w0)
            exp_term = np.clip(exp_term, -500.0, 500.0)
            term2 = (1.0 - Om) * a**exponent * np.exp(exp_term)
            return np.sqrt(Om * a**(-3.0) + term2)

    # Diverse training cosmologies covering parameter space
    # Using a grid to ensure broad coverage
    Om_vals = [0.2, 0.25, 0.3, 0.35, 0.4]
    w0_vals = [-1.2, -1.0, -0.8]
    wa_vals = [-0.3, 0.0, 0.3]
    
    models = []
    for Om in Om_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                models.append({'Om': Om, 'w0': w0, 'wa': wa})
                
    # Add some specific edge cases or interesting ones
    models.append({'Om': 0.15, 'w0': -0.7, 'wa': -0.3})
    models.append({'Om': 0.45, 'w0': -1.3, 'wa': 0.4})
    models.append({'Om': 0.1, 'w0': -1.5, 'wa': 0.5})
    models.append({'Om': 0.5, 'w0': -0.5, 'wa': -0.5})

    redshifts = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    
    # Pre-compute true integrals using quad
    true_integrals = np.zeros((len(models), len(redshifts)))
    for i, m in enumerate(models):
        Om, w0, wa = m['Om'], m['w0'], m['wa']
        for j, z in enumerate(redshifts):
            def integrand(xi, Om=Om, w0=w0, wa=wa, z=z):
                a = 1.0 / (1.0 + z * xi)
                return 1.0 / E(a, Om, w0, wa)
            val, _ = quad(integrand, 0.0, 1.0, limit=200, epsabs=1e-12, epsrel=1e-12)
            true_integrals[i, j] = val

    n_models = len(models)
    n_z = len(redshifts)

    # Parameterization:
    # Nodes: x_i = sigmoid(u_i). To ensure strict ordering, we sort u_i and use the sorted values for nodes.
    # Weights: w_i = exp(v_i) / sum(exp(v_j)).
    # This ensures nodes are in (0,1) and increasing, and weights are positive and sum to 1.
    
    def decode_params(params):
        u = params[:8]
        v = params[8:]
        
        # Sort u to get ordered nodes
        u_sorted = np.sort(u)
        nodes = 1.0 / (1.0 + np.exp(-u_sorted))
        # Ensure nodes are strictly inside (0,1)
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        
        # Calculate weights
        v_max = np.max(v)
        exp_v = np.exp(v - v_max)
        weights = exp_v / np.sum(exp_v)
        
        return nodes, weights

    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        nodes, weights = decode_params(params)
        
        # Compute errors
        max_err = 0.0
        for i in range(n_models):
            Om, w0, wa = models[i]['Om'], models[i]['w0'], models[i]['wa']
            for j in range(n_z):
                z = redshifts[j]
                a_vals = 1.0 / (1.0 + z * nodes)
                e_vals = E(a_vals, Om, w0, wa)
                q_val = np.dot(weights, 1.0 / e_vals)
                true_val = true_integrals[i, j]
                rel_err = abs(q_val - true_val) / true_val
                if rel_err > max_err:
                    max_err = rel_err
                    
        return max_err

    # Initial guess: Gauss-Legendre nodes and weights
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = (nodes_gl + 1.0) / 2.0
    weights_gl = weights_gl / 2.0
    
    # Encode GL nodes and weights
    # u = logit(nodes)
    u_init = np.log(nodes_gl / (1.0 - nodes_gl))
    # v = log(weights)
    v_init = np.log(weights_gl)
    
    x0 = np.concatenate([u_init, v_init])
    
    # Bounds for u and v
    # u: sigmoid maps R to (0,1). Let's use [-10, 10]
    # v: exp maps R to (0, inf). Let's use [-10, 10]
    bounds = [(-10.0, 10.0)] * 16
    
    best_x = x0
    best_val = objective(best_x)
    
    # Differential Evolution for global search
    # This is robust to non-convexity
    try:
        def de_callback(xk, convergence):
            nonlocal best_x, best_val
            val = objective(xk)
            if val < best_val:
                best_val = val
                best_x = xk.copy()
            return time.time() > deadline

        de_res = differential_evolution(objective, bounds, maxiter=50, popsize=15, tol=1e-6, 
                                        callback=de_callback, seed=42, polish=False)
        if de_res.fun < best_val:
            best_val = de_res.fun
            best_x = de_res.x
    except Exception:
        pass

    # Local refinement with L-BFGS-B
    if time.time() < deadline - 5.0:
        try:
            # Use L-BFGS-B on the unbounded parameters (u, v)
            # Since constraints are handled by parameterization, we don't need bounds for L-BFGS-B
            # but we can provide them to keep values reasonable.
            res = minimize(objective, best_x, method='L-BFGS-B', 
                           bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
    # Final polish with Nelder-Mead (often good for final steps)
    if time.time() < deadline - 2.0:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # Decode final result
    nodes, weights = decode_params(best_x)
    
    # Ensure weights sum to 1 exactly
    weights = weights / np.sum(weights)
    
    # Final validity check
    if not np.all(np.diff(nodes) > 0) or not np.all(weights > 0) or abs(np.sum(weights) - 1.0) > 1e-12:
        # Fallback to Gauss-Legendre
        nodes = nodes_gl.copy()
        weights = weights_gl.copy()
        
    return {"nodes": nodes, "weights": weights}

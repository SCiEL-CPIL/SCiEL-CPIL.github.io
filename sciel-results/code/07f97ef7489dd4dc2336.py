import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # 1. Define test cosmologies with diverse parameters
    # E(a) = sqrt(Omega_m a^-3 + (1-Omega_m) * a^(-3*(1+w0+wa)) * exp(3*wa*(1-a)/(1+w0)))
    
    def E(a, Om, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL dark energy model."""
        # Handle the w0 = -1 case (cosmological constant)
        if abs(w0 + 1.0) < 1e-8:
            if abs(wa) < 1e-8:
                return np.sqrt(Om * a**(-3.0) + (1.0 - Om))
            else:
                # For w0=-1, wa!=0, use the limiting form
                # This is a singular case; approximate with small epsilon
                eps = 1e-8
                w0_eff = -1.0 + eps
                return np.sqrt(Om * a**(-3.0) + (1.0 - Om) * a**(-3.0 * (1.0 + w0_eff + wa)) * np.exp(3.0 * wa * (1.0 - a) / (1.0 + w0_eff)))
        
        # Standard CPL formula
        exponent = -3.0 * (1.0 + w0 + wa)
        exp_term = 3.0 * wa * (1.0 - a) / (1.0 + w0)
        # Guard against overflow/underflow
        exp_term = np.clip(exp_term, -500.0, 500.0)
        term2 = (1.0 - Om) * a**exponent * np.exp(exp_term)
        return np.sqrt(Om * a**(-3.0) + term2)
    
    # Diverse test cosmologies covering parameter space
    models = [
        {'Om': 0.3, 'w0': -1.0, 'wa': 0.0},   # LCDM
        {'Om': 0.3, 'w0': -0.9, 'wa': 0.1},
        {'Om': 0.2, 'w0': -1.1, 'wa': 0.2},
        {'Om': 0.4, 'w0': -0.8, 'wa': -0.2},
        {'Om': 0.25, 'w0': -1.0, 'wa': 0.0},
        {'Om': 0.35, 'w0': -1.2, 'wa': 0.3},
        {'Om': 0.15, 'w0': -0.7, 'wa': -0.3},
        {'Om': 0.3, 'w0': -1.05, 'wa': 0.15},
    ]
    
    redshifts = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
    
    # Pre-compute true integrals using quad
    true_integrals = np.zeros((len(models), len(redshifts)))
    for i, m in enumerate(models):
        Om, w0, wa = m['Om'], m['w0'], m['wa']
        for j, z in enumerate(redshifts):
            def integrand(xi):
                a = 1.0 / (1.0 + z * xi)
                return 1.0 / E(a, Om, w0, wa)
            val, _ = quad(integrand, 0.0, 1.0, limit=200, epsabs=1e-12, epsrel=1e-12)
            true_integrals[i, j] = val
    
    # 2. Vectorized objective function
    def compute_quadrature(nodes, weights, model_idx, z_idx):
        """Compute quadrature sum for a single model-redshift pair."""
        Om, w0, wa = models[model_idx]['Om'], models[model_idx]['w0'], models[model_idx]['wa']
        z = redshifts[z_idx]
        a_vals = 1.0 / (1.0 + z * nodes)
        e_vals = E(a_vals, Om, w0, wa)
        return np.dot(weights, 1.0 / e_vals)
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        nodes = params[:8]
        weights = params[8:]
        
        # Check validity
        if not np.all(nodes > 0) or not np.all(nodes < 1):
            return 1e10
        if not np.all(np.diff(nodes) > 1e-10):
            return 1e10
        if not np.all(weights > 1e-10):
            return 1e10
        
        w_sum = np.sum(weights)
        if abs(w_sum - 1.0) > 1e-6:
            return 1e10
        
        # Compute relative errors for all model-redshift pairs
        errors = np.zeros((len(models), len(redshifts)))
        for i in range(len(models)):
            for j in range(len(redshifts)):
                q_val = compute_quadrature(nodes, weights, i, j)
                true_val = true_integrals[i, j]
                errors[i, j] = abs(q_val - true_val) / true_val
        
        # Soft-max objective (smooth approximation to max)
        beta = 50.0
        max_err = np.log(np.sum(np.exp(beta * errors)) / (len(models) * len(redshifts))) / beta
        return max_err

    # Test Cosmologies
    models = [
        {'Om': 0.3, 'w0': -1.0, 'wa': 0.0},  # LCDM
        {'Om': 0.3, 'w0': -0.9, 'wa': 0.1},
        {'Om': 0.2, 'w0': -1.1, 'wa': 0.2},
        {'Om': 0.4, 'w0': -0.8, 'wa': -0.2},
        {'Om': 0.25, 'w0': -1.0, 'wa': 0.0},
    ]
    
    redshifts = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    # Pre-compute true integrals
    # Integral: int_0^1 1/E(1/(1+z*xi)) dxi
    true_integrals = []
    for m in models:
        Om = m['Om']
        w0 = m['w0']
        wa = m['wa']
        for z in redshifts:
            def integrand(xi):
                a = 1.0 / (1.0 + z * xi)
                # Handle w0=-1, wa=0 case
                if np.abs(w0 + 1.0) < 1e-10 and np.abs(wa) < 1e-10:
                    e_val = np.sqrt(Om * a**(-3.0) + (1.0 - Om))
                else:
                    e_val = E(a, Om, w0, wa)
                return 1.0 / e_val
            
            # quad returns (value, error)
            val, err = quad(integrand, 0.0, 1.0, limit=200)
            true_integrals.append(val)
    
    true_integrals = np.array(true_integrals)
    
    # 2. Define objective function
    def objective(params):
        # params: 16 values (8 nodes, 8 weights)
        nodes = params[:8]
        weights = params[8:]
        
        # Penalty for constraints
        penalty = 0.0
        
        # Nodes must be in [0, 1] and strictly increasing
        for i in range(8):
            if nodes[i] < 0 or nodes[i] > 1:
                penalty += 100.0 * (abs(nodes[i]) + abs(1.0 - nodes[i]))
        for i in range(7):
            if nodes[i] >= nodes[i+1]:
                penalty += 100.0 * (nodes[i] - nodes[i+1])
                
        # Weights must be positive
        for i in range(8):
            if weights[i] <= 0:
                penalty += 100.0 * abs(weights[i])
                
        # Weights sum to 1
        w_sum = np.sum(weights)
        penalty += 10.0 * abs(w_sum - 1.0)
        
        if penalty > 1.0:
            return penalty
            
        # Compute quadrature sums
        errors = []
        for idx, m in enumerate(models):
            Om = m['Om']
            w0 = m['w0']
            wa = m['wa']
            for j, z in enumerate(redshifts):
                true_val = true_integrals[idx * len(redshifts) + j]
                
                # Compute sum
                s = 0.0
                for i in range(8):
                    a = 1.0 / (1.0 + z * nodes[i])
                    if np.abs(w0 + 1.0) < 1e-10 and np.abs(wa) < 1e-10:
                        e_val = np.sqrt(Om * a**(-3.0) + (1.0 - Om))
                    else:
                        e_val = E(a, Om, w0, wa)
                    s += weights[i] / e_val
                    
                rel_err = abs(s - true_val) / true_val
                errors.append(rel_err)
                
        errors = np.array(errors)
        # Soft max
        beta = 100.0
        max_err = np.log(np.sum(np.exp(beta * errors)) / len(errors)) / beta
        
        return max_err + penalty

    # 3. Optimization with multiple restarts
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = (nodes_gl + 1.0) / 2.0
    weights_gl = weights_gl / 2.0
    
    best_x = np.concatenate([nodes_gl, weights_gl])
    best_val = objective(best_x)
    
    # Use log-space parameterization for nodes to enforce strict ordering
    # nodes[i] = sigmoid(cumsum(log_weights_nodes)) mapped to (0,1)
    # Simpler: use direct optimization with careful bounds
    
    bounds = [(1e-6, 1.0 - 1e-6)] * 8 + [(1e-8, 1.0)] * 8
    
    def callback(xk):
        nonlocal best_x, best_val
        val = objective(xk)
        if val < best_val:
            best_x = xk.copy()
            best_val = val
        return time.time() > deadline
    
    # Multiple restarts with different initial conditions
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        if r == 0:
            x0 = best_x.copy()
        else:
            # Perturb from best or use different initializations
            if r == 1:
                # Chebyshev nodes
                k = np.arange(8)
                x0_nodes = 0.5 * (1.0 + np.cos(np.pi * (2*k + 1) / 16.0))
                x0_nodes = np.sort(x0_nodes)
                x0_weights = np.ones(8) / 8.0
            elif r == 2:
                # Equally spaced
                x0_nodes = np.linspace(0.05, 0.95, 8)
                x0_weights = np.ones(8) / 8.0
            else:
                # Perturb best
                x0 = best_x.copy()
                x0 += np.random.normal(0, 0.01, 16)
                x0[:8] = np.sort(x0[:8])
                x0[:8] = np.clip(x0[:8], 1e-6, 1.0 - 1e-6)
                x0[8:] = np.clip(x0[8:], 1e-8, 1.0)
                x0[8:] /= np.sum(x0[8:])
            
            if r in [1, 2]:
                x0 = np.concatenate([x0_nodes, x0_weights])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                          callback=callback, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14})
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 2.0:
        try:
            res = minimize(objective, best_x, method='L-BFGS-B', bounds=bounds,
                          callback=callback, options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-16})
        except Exception:
            pass
    
    # Ensure final result is valid
    nodes = best_x[:8]
    weights = best_x[8:]
    
    # Sort nodes and re-associate weights (in case order was violated)
    sort_idx = np.argsort(nodes)
    nodes = nodes[sort_idx]
    weights = weights[sort_idx]
    
    # Check validity
    if (not np.all(np.diff(nodes) > 0) or not np.all(weights > 0) or 
        abs(np.sum(weights) - 1.0) > 1e-8 or nodes[0] <= 0 or nodes[-1] >= 1):
        # Fallback to Gauss-Legendre
        nodes = nodes_gl.copy()
        weights = weights_gl.copy()
    else:
        # Normalize weights to sum exactly to 1
        weights = weights / np.sum(weights)
        # Ensure strict positivity
        weights = np.maximum(weights, 1e-15)
        weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}

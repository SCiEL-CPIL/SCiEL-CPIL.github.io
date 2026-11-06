import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _E(a, w0, w1):
    """E(a) = H(a)/H0 for CPL dark energy with flat spatial geometry."""
    # a = 1/(1+z), so (1-a) = z
    # E^2 = (1-a)^-2 * (1 + w0 + w1*(1-a))  ... wait, let me be careful.
    # H(z)^2/H0^2 = (1+z)^2 * [Omega_m*(1+z)^3 + Omega_k*(1+z)^2 + Omega_de*(1+z)^{3(1+w0+w1(1-z/(1+z))}]
    # Actually for CPL: w(z) = w0 + w1*z
    # Omega_de(z) = Omega_de0 * (1+z)^{3(1+w0)} * exp(3*w1*z/(1+z))
    # But the task says "spatially flat, time-varying dark-energy models"
    # The formula given: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # where a_i = 1/(1+z*nodes[i])
    # So the integrand is 1/E(a) where a is the scale factor at the node.
    # For flat universe: E^2(a) = Omega_m/a^3 + Omega_de(a)
    # With CPL: Omega_de(a) = Omega_de0 * a^{-3(1+w0)} * exp(-3*w1*ln(1/a))
    #           = Omega_de0 * a^{-3(1+w0)} * exp(3*w1*ln(1/a))
    #           = Omega_de0 * a^{-3(1+w0)} * (1/a)^{3*w1}
    #           = Omega_de0 * a^{-3(1+w0+w1)}
    # Wait, that's not right. Let me reconsider.
    # w(z) = w0 + w1*z
    # The dark energy density evolves as: rho_de(z) = rho_de0 * (1+z)^{3(1+w0)} * exp(3*w1*z/(1+z))
    # In terms of a = 1/(1+z): rho_de(a) = rho_de0 * a^{-3(1+w0)} * exp(3*w1*(1-a)/a)
    # This is complicated. But the task says the evaluator computes integrands itself.
    # So I just need to find nodes and weights that minimize worst-case error.
    # I need to define a set of test cosmologies to optimize against.
    
    # Let me use a simpler parameterization. For flat universe with CPL:
    # E^2(a) = Omega_m * a^{-3} + (1-Omega_m) * a^{-3(1+w0)} * exp(3*w1*(1-a)/a)
    
    Omega_m = 0.3
    Omega_de = 1.0 - Omega_m  # flat
    return np.sqrt(Omega_m * a**(-3) + Omega_de * a**(-3*(1+w0)) * np.exp(3*w1*(1-a)/a))

def _integrand(z, w0, w1):
    """Integrand for luminosity distance: (1+z)/E(1/(1+z))"""
    a = 1.0 / (1.0 + z)
    return (1.0 + z) / _E(a, w0, w1)

def _true_distance(z, w0, w1, n_points=10000):
    """Compute true luminosity distance via adaptive quadrature."""
    from scipy.integrate import quad
    val, _ = quad(lambda zz: _integrand(zz, w0, w1), 0, z, limit=200)
    return val

def _quad_distance(z, nodes, weights, w0, w1):
    """Compute quadrature approximation of luminosity distance."""
    z_nodes = z * nodes
    a_nodes = 1.0 / (1.0 + z_nodes)
    vals = (1.0 + z_nodes) / _E(a_nodes, w0, w1)
    return z * np.sum(weights * vals)

def _relative_error(z, nodes, weights, w0, w1):
    """Compute relative error in percent."""
    true = _true_distance(z, w0, w1)
    approx = _quad_distance(z, nodes, weights, w0, w1)
    return abs(approx - true) / true * 100.0

def _worst_error(nodes, weights, test_cosmos):
    """Compute worst relative error over all test cosmologies and redshifts."""
    max_err = 0.0
    for (w0, w1) in test_cosmos:
        for z in [0.5, 1.0, 2.0, 3.0]:
            err = _relative_error(z, nodes, weights, w0, w1)
            if err > max_err:
                max_err = err
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s safety margin
    
    # Test cosmologies: (w0, w1) pairs
    # Include standard LambdaCDM, phantom, quintessence, etc.
    test_cosmos = [
        (-1.0, 0.0),      # LambdaCDM
        (-1.0, 0.5),      # CPL with w1>0
        (-1.0, -0.5),     # CPL with w1<0
        (-0.9, 0.3),      # Quintessence-like
        (-0.95, -0.3),    # Another variant
        (-1.2, 0.0),      # Phantom
        (-0.8, 0.5),      # Strong evolution
        (-1.0, 1.0),      # Strong CPL
    ]
    
    # Start with Gauss-Legendre as initial guess
    t, w = np.polynomial.legendre.leggauss(8)
    nodes_init = 0.5 * (t + 1.0)
    weights_init = 0.5 * w
    weights_init = weights_init / np.sum(weights_init)
    
    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    best_err = _worst_error(best_nodes, best_weights, test_cosmos)
    
    # Try local optimization
    def _objective(params):
        if time.time() > deadline:
            return 1e10
        nodes = params[:8]
        weights = params[8:]
        
        # Check constraints
        if np.any(nodes[1:] <= nodes[:-1]):
            return 1e10
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        if np.any(weights <= 0):
            return 1e10
        if abs(np.sum(weights) - 1.0) > 1e-12:
            return 1e10
        
        return _worst_error(nodes, weights, test_cosmos)
    
    # Normalize weights in the objective
    def _objective_norm(params):
        if time.time() > deadline:
            return 1e10
        nodes = params[:8]
        weights = params[8:]
        
        # Check node constraints
        if np.any(nodes[1:] <= nodes[:-1]):
            return 1e10
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        if np.any(weights <= 0):
            return 1e10
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        return _worst_error(nodes, weights, test_cosmos)
    
    # Initial params
    x0 = np.concatenate([nodes_init, weights_init])
    
    # Try Nelder-Mead
    try:
        result = minimize(_objective_norm, x0, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
        if time.time() < deadline and result.fun < best_err:
            best_nodes = result.x[:8]
            best_weights = result.x[8:] / np.sum(result.x[8:])
            best_err = result.fun
    except Exception:
        pass
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for i in range(10):
        if time.time() > deadline:
            break
        
        # Random perturbation of GL nodes
        nodes_try = nodes_init + np.random.uniform(-0.05, 0.05, 8)
        nodes_try = np.sort(nodes_try)
        nodes_try = np.clip(nodes_try, 0.01, 0.99)
        # Ensure strict increase
        for j in range(1, 8):
            if nodes_try[j] <= nodes_try[j-1]:
                nodes_try[j] = nodes_try[j-1] + 1e-6
        
        weights_try = weights_init + np.random.uniform(-0.02, 0.02, 8)
        weights_try = np.maximum(weights_try, 1e-6)
        weights_try = weights_try / np.sum(weights_try)
        
        err = _worst_error(nodes_try, weights_try, test_cosmos)
        if err < best_err:
            best_nodes = nodes_try
            best_weights = weights_try
            best_err = err
    
    # Final optimization from best point
    if time.time() < deadline - 5:
        x0 = np.concatenate([best_nodes, best_weights])
        try:
            result = minimize(_objective_norm, x0, method='Nelder-Mead', 
                             options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            if time.time() < deadline and result.fun < best_err:
                best_nodes = result.x[:8]
                best_weights = result.x[8:] / np.sum(result.x[8:])
                best_err = result.fun
        except Exception:
            pass
    
    # Final constraint enforcement
    best_nodes = np.sort(best_nodes)
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    # Ensure strict increase
    for j in range(1, 8):
        if best_nodes[j] <= best_nodes[j-1]:
            best_nodes[j] = best_nodes[j-1] + 1e-10
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}

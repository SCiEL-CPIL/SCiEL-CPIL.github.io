import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Gauss-Legendre 8-point nodes and weights on [-1, 1]
    t_gl = np.array([
        -0.9602898564975363,
        -0.7966664774136267,
        -0.5255324099163290,
        -0.1834346425004484,
         0.1834346425004484,
         0.5255324099163290,
         0.7966664774136267,
         0.9602898564975363
    ])
    w_gl = np.array([
        0.1012285362903763,
        0.2223810344533745,
        0.3137066458778873,
        0.3626837833783620,
        0.3626837833783620,
        0.3137066458778873,
        0.2223810344533745,
        0.1012285362903763
    ])
    
    nodes_init = 0.5 * t_gl + 0.5
    weights_init = 0.5 * w_gl
    weights_init = weights_init / np.sum(weights_init)

    # CPL cosmology models to cover the parameter space
    # (Om_m, w0, wa)
    cpl_models = [
        (0.3, -1.0, 0.0),   # LambdaCDM
        (0.3, -1.0, 1.0),   # w0=-1, wa=1
        (0.3, -1.0, -1.0),  # w0=-1, wa=-1
        (0.3, -0.5, 0.0),   # w0=-0.5
        (0.3, -0.5, 1.0),   # w0=-0.5, wa=1
        (0.3, -0.5, -1.0),  # w0=-0.5, wa=-1
        (0.3, 0.0, 0.0),    # w0=0
        (0.3, 0.0, 1.0),    # w0=0, wa=1
        (0.2, -1.0, 0.0),   # Om_m=0.2
        (0.4, -1.0, 0.0),   # Om_m=0.4
        (0.3, -1.5, 0.5),   # Phantom
        (0.3, -1.5, -0.5),  # Phantom with evolution
        (0.3, -0.8, 0.5),   # Intermediate
        (0.3, -1.2, 0.8),   # More phantom
    ]
    
    z_values = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0])

    def _E(z, w0, wa, Om_m):
        """H(z)/H0 for CPL model in flat universe."""
        z = np.asarray(z, dtype=float)
        one_plus_z = 1.0 + z
        matter = Om_m * one_plus_z**3
        # DE: Om_La * (1+z)^(3(1+w0+wa*(1-1/(1+z))))
        # = Om_La * (1+z)^(3(1+w0)) * (1+z)^(3*wa*(1-1/(1+z)))
        # = Om_La * (1+z)^(3(1+w0)) * ((1+z)/(z))^(3*wa) ... no
        # Standard CPL: E^2 = Om_m(1+z)^3 + Om_La(1+z)^(3(1+w0+wa*(1-1/(1+z))))
        # The exponent: 3*(1+w0+wa*(1-1/(1+z))) = 3*(1+w0+wa*z/(1+z))
        Om_La = 1.0 - Om_m
        exponent = 3.0 * (1.0 + w0 + wa * z / one_plus_z)
        de = Om_La * one_plus_z**exponent
        E_sq = matter + de
        return np.sqrt(np.maximum(E_sq, 1e-30))

    def _true_dL(z, w0, wa, Om_m):
        """Compute true dL/(c/H0) using dense quadrature."""
        if z < 1e-15:
            return 0.0
        # dL/(c/H0) = (1+z) * integral_0^z dz'/E(z')
        # Use substitution u in [0,1], z' = z*u, dz' = z*du
        # = (1+z) * z * integral_0^1 du / E(z*u)
        u = np.linspace(0, 1, 2001)
        z_prime = z * u
        E_vals = _E(z_prime, w0, wa, Om_m)
        integrand = 1.0 / E_vals
        # Handle z=0 endpoint: integrand at z'=0 is 1/E(0)=1
        # Use trapezoidal rule
        true_int = np.trapz(integrand, u)
        return (1.0 + z) * z * true_int

    def _quad_dL(nodes, weights, z, w0, wa, Om_m):
        """Compute quadrature approximation of dL/(c/H0)."""
        if z < 1e-15:
            return 0.0
        z_prime = z * nodes
        E_vals = _E(z_prime, w0, wa, Om_m)
        integral_approx = np.sum(weights / E_vals)
        return (1.0 + z) * z * integral_approx

    def _encode(nodes, weights):
        """Encode nodes and weights into unconstrained parameter vector."""
        # Nodes: use log-spacings to ensure strictly increasing in (0,1)
        spacings = np.diff(np.concatenate([[0], nodes, [1]]))
        # spacings has 9 elements, all positive
        log_spacings = np.log(spacings)
        # Weights: use log-weights
        log_weights = np.log(np.maximum(weights, 1e-30))
        return np.concatenate([log_spacings, log_weights])

    def _decode(x):
        """Decode parameter vector to nodes and weights."""
        log_spacings = x[:9]
        log_weights = x[9:17]
        
        spacings = np.exp(log_spacings)
        total_spacing = np.sum(spacings)
        # nodes are cumulative sums of first 8 spacings, normalized
        cumsum = np.cumsum(spacings[:8])
        nodes = cumsum / total_spacing
        
        exp_w = np.exp(log_weights)
        weights = exp_w / np.sum(exp_w)
        
        return nodes, weights

    def _objective(x):
        """Objective: minimize worst-case relative error across all models and z."""
        if time.time() > deadline:
            return 1e10
        
        nodes, weights = _decode(x)
        
        # Check basic validity
        if not np.all(nodes > 0) or not np.all(nodes < 1):
            return 1e10
        if not np.all(np.diff(nodes) > 1e-10):
            return 1e10
        if not np.all(weights > 0):
            return 1e10
        
        max_rel_err = 0.0
        for Om_m, w0, wa in cpl_models:
            for z in z_values:
                true_val = _true_dL(z, w0, wa, Om_m)
                quad_val = _quad_dL(nodes, weights, z, w0, wa, Om_m)
                if true_val > 1e-10:
                    rel_err = abs(quad_val - true_val) / true_val
                    if rel_err > max_rel_err:
                        max_rel_err = rel_err
        return max_rel_err

    # Initialize with GL rule
    x0 = _encode(nodes_init, weights_init)
    
    best_x = x0.copy()
    best_nodes, best_weights = _decode(x0)
    best_obj = _objective(x0)
    
    # Optimization with L-BFGS-B
    if time.time() < deadline - 15.0:
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            if res.fun < best_obj:
                best_x = res.x
                best_nodes, best_weights = _decode(res.x)
                best_obj = res.fun
        except Exception:
            pass

    # Refine with Powell if time permits
    if time.time() < deadline - 5.0:
        try:
            res2 = minimize(_objective, best_x, method='Powell',
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
            if res2.fun < best_obj:
                best_x = res2.x
                best_nodes, best_weights = _decode(res2.x)
                best_obj = res2.fun
        except Exception:
            pass

    # Final extraction and validation
    nodes = best_nodes
    weights = best_weights
    
    # Ensure constraints are met
    if not np.all(np.diff(nodes) > 0) or not np.all(nodes > 0) or not np.all(nodes < 1):
        nodes = nodes_init
        weights = weights_init
    if not np.all(weights > 0):
        weights = np.abs(weights)
    weights = weights / np.sum(weights)
    
    # Final check
    if abs(np.sum(weights) - 1.0) > 1e-12:
        weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}

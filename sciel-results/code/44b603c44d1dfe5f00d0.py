"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import roots_legendre


def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    Strategy: Jointly optimize nodes and weights using L-BFGS-B with a robust
    parameterization (cumulative differences for nodes, log-space for weights).
    Minimize worst-case relative error across representative CPL cosmologies.
    """
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill

    # Representative CPL cosmologies: (Om_m, Om_de, w0, wa)
    models = [
        (0.3, 0.7, -1.0, 0.0),    # LCDM
        (0.3, 0.7, -1.0, 0.5),    # CPL with positive wa
        (0.3, 0.7, -0.8, -0.5),   # CPL with negative wa
        (0.3, 0.7, -0.5, 0.0),    # wCDM
        (0.3, 0.7, -1.2, 0.3),    # CPL with w0 < -1
        (0.3, 0.7, -0.9, 0.2),    # Another CPL
        (0.25, 0.75, -1.0, 0.0),  # Different matter content
        (0.35, 0.65, -1.0, 0.0),  # Different matter content
    ]

    # Test z values covering the relevant range
    z_vals = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])

    def E_of_z(z, Om, OL, w0, wa):
        """Compute E(z) = H(z)/H0 for CPL model."""
        if abs(wa) < 1e-12 and abs(w0 + 1.0) < 1e-12:
            # LCDM: E(z) = sqrt(Om*(1+z)^3 + OL)
            return np.sqrt(Om * (1.0 + z)**3 + OL)
        else:
            # General CPL
            # E^2(z) = Om*(1+z)^3 + OL*(1+z)^(3*(1+w0)) * exp(3*wa*z/(1+z))
            term1 = Om * (1.0 + z)**3
            exponent = 3.0 * wa * z / (1.0 + z)
            term2 = OL * (1.0 + z)**(3.0 * (1.0 + w0)) * np.exp(exponent)
            return np.sqrt(term1 + term2)

    def true_distance(z, Om, OL, w0, wa):
        """High-accuracy reference distance using adaptive quadrature."""
        if z <= 0:
            return 0.0
        integrand = lambda zp: 1.0 / E_of_z(zp, Om, OL, w0, wa)
        val, _ = quad(integrand, 0, z, limit=100, epsabs=1e-12, epsrel=1e-12)
        return (1.0 + z) * val

    def quadrature_distance(z, nodes, weights, Om, OL, w0, wa):
        """Compute distance using the quadrature rule."""
        if z <= 0:
            return 0.0
        z_nodes = z * nodes
        E_vals = E_of_z(z_nodes, Om, OL, w0, wa)
        integral_approx = np.sum(weights / E_vals)
        return (1.0 + z) * z * integral_approx

    def params_to_nodes_weights(params):
        """
        Convert parameters to nodes and weights.
        params: [u1, d1, d2, ..., d7, y1, y2, ..., y8]
        u1 > 0, di > 0 ensure strictly increasing nodes in (0,1)
        y_i are log-weights, normalized to sum to 1
        """
        u1 = params[0]
        ds = params[1:8]
        ys = params[8:16]
        
        # Reconstruct nodes
        nodes = np.zeros(8)
        nodes[0] = u1
        for i in range(1, 8):
            nodes[i] = nodes[i-1] + ds[i-1]
        
        # Check bounds: nodes must be in (0, 1)
        if nodes[0] <= 0 or nodes[-1] >= 1.0 or not np.all(np.diff(nodes) > 0):
            return None, None
        
        # Normalize weights from log-space
        y_max = np.max(ys)
        w_exp = np.exp(ys - y_max)
        w_sum = np.sum(w_exp)
        if w_sum < 1e-15:
            return None, None
        weights = w_exp / w_sum
        
        return nodes, weights

    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        nodes, weights = params_to_nodes_weights(params)
        if nodes is None:
            return 1e10
        
        max_err = 0.0
        for z in z_vals:
            for (Om, OL, w0, wa) in models:
                d_true = true_distance(z, Om, OL, w0, wa)
                if d_true < 1e-10:
                    continue
                d_quad = quadrature_distance(z, nodes, weights, Om, OL, w0, wa)
                rel_err = abs(d_quad - d_true) / d_true
                if rel_err > max_err:
                    max_err = rel_err
                    if max_err > 1.0:  # Early exit if error is huge
                        return max_err
        return max_err

    # Initial guess: Gauss-Legendre nodes and weights
    gl_nodes, gl_weights = roots_legendre(8)
    init_nodes = 0.5 * (gl_nodes + 1.0)
    init_weights = 0.5 * gl_weights
    
    # Convert to parameter format
    u1 = init_nodes[0]
    ds = np.diff(init_nodes)
    ys = np.log(init_weights + 1e-15)
    
    init_params = np.concatenate(([u1], ds, ys))
    
    # Bounds for parameters
    bounds = []
    bounds.append((1e-6, 0.999999))  # u1
    for i in range(7):
        bounds.append((1e-6, 0.999999))  # ds
    for i in range(8):
        bounds.append((-20.0, 20.0))  # log-weights
    
    # Optimize with L-BFGS-B
    best_params = init_params.copy()
    best_score = objective(init_params)
    
    try:
        res = minimize(objective, init_params, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-6})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x.copy()
    except Exception:
        pass
    
    # Local refinement with a different starting point if time permits
    if time.time() < deadline - 10:
        # Try midpoint rule as alternative start
        mid_nodes = (np.arange(8) + 0.5) / 8.0
        mid_weights = np.full(8, 1.0/8.0)
        u1_mid = mid_nodes[0]
        ds_mid = np.diff(mid_nodes)
        ys_mid = np.log(mid_weights + 1e-15)
        mid_params = np.concatenate(([u1_mid], ds_mid, ys_mid))
        
        try:
            res2 = minimize(objective, mid_params, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 100, 'maxfun': 3000, 'ftol': 1e-10, 'gtol': 1e-6})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Final local refinement with Powell if time permits
    if time.time() < deadline - 5:
        try:
            res3 = minimize(objective, best_params, method='Powell', bounds=bounds,
                          options={'maxiter': 50, 'maxfun': 2000, 'xtol': 1e-8, 'ftol': 1e-8})
            if res3.fun < best_score:
                best_score = res3.fun
                best_params = res3.x.copy()
        except Exception:
            pass

    # Extract final nodes and weights
    nodes, weights = params_to_nodes_weights(best_params)
    if nodes is None:
        # Fallback to Gauss-Legendre
        nodes, weights = 0.5 * (gl_nodes + 1.0), 0.5 * gl_weights
    
    # Final sanity checks
    if not np.all(np.diff(nodes) > 0) or nodes[0] <= 0 or nodes[-1] >= 1.0:
        nodes, weights = 0.5 * (gl_nodes + 1.0), 0.5 * gl_weights
    
    if not np.all(weights > 0):
        weights = 0.5 * gl_weights
    
    # Normalize weights to sum exactly to 1
    weights = weights / np.sum(weights)
    
    # Verify sum is within 1e-12
    if abs(np.sum(weights) - 1.0) > 1e-12:
        weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END

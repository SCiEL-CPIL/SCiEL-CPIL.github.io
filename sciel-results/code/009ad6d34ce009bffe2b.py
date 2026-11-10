import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Gauss-Legendre 8-point nodes and weights on [-1, 1] as initial guess
    t_gl = np.array([
        -0.9602898564975363, -0.7966664774136267,
        -0.5255324099163290, -0.1834346425004484,
         0.1834346425004484,  0.5255324099163290,
         0.7966664774136267,  0.9602898564975363
    ])
    w_gl = np.array([
        0.1012285362903763, 0.2223810344533745,
        0.3137066458778873, 0.3626837833783620,
        0.3626837833783620, 0.3137066458778873,
        0.2223810344533745, 0.1012285362903763
    ])
    
    nodes_gl = 0.5 * t_gl + 0.5
    weights_gl = 0.5 * w_gl
    weights_gl = weights_gl / np.sum(weights_gl)
    
    # Best solution so far
    best_nodes = nodes_gl.copy()
    best_weights = weights_gl.copy()
    best_obj = np.inf
    
    # CPL cosmology models: (w0, wa) pairs covering typical range
    # These are representative of what might be in validation
    cpl_models = [
        (0, 0),       # Lambda CDM
        (-1, 0),      # Cosmological constant
        (-1, 1),      # Quintessence-like
        (-0.5, 0),    # Mild dark energy
        (-1, -1),     # Phantom-ish
        (0.5, 0),     # Matter-dominated-ish
        (-1, 0.5),    # Evolving DE
        (-0.8, 0.2),  # Intermediate
        (-1, -0.5),   # Slightly phantom
        (-0.7, -0.3), # Mild evolution
        (-1, 1.5),    # Strong evolution
        (0, 1),       # Matter + evolving
        (-1.2, 0.3),  # Phantom with evolution
        (-0.9, -0.4), # Near -1 with negative evolution
    ]
    
    # Redshifts to test at
    z_values = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    def _E(a, w0, wa):
        """H(a)/H0 for CPL model. a = 1/(1+z)"""
        Om_m = 0.3
        Om_La = 1.0 - Om_m
        # Exponent for DE: 3*(1+w0+wa*(1-a))
        exponent_de = 3.0 * (1.0 + w0 + wa * (1.0 - a))
        # Matter: a^-3
        term_m = Om_m * a**(-3.0)
        term_de = Om_La * a**(exponent_de)
        E_sq = term_m + term_de
        return np.sqrt(np.maximum(E_sq, 1e-30))
    
    def _true_integral(w0, wa, z):
        """Compute true integral using dense quadrature"""
        u = np.linspace(0, 1, 10001)
        a = 1.0 / (1.0 + z * u)
        E_vals = _E(a, w0, wa)
        integrand = 1.0 / E_vals
        true_int = np.trapz(integrand, u)
        return (1.0 + z) * z * true_int
    
    def _quadrature_value(nodes, weights, w0, wa, z):
        """Compute quadrature approximation"""
        a = 1.0 / (1.0 + z * nodes)
        E_vals = _E(a, w0, wa)
        integral_approx = np.sum(weights / E_vals)
        return (1.0 + z) * z * integral_approx
    
    def _objective(x):
        """Objective: minimize worst-case relative error"""
        # x = [8 node positions, 8 log-weights]
        # Nodes are directly parameterized with sigmoid to ensure (0,1)
        raw_nodes = x[:8]
        nodes = 1.0 / (1.0 + np.exp(-raw_nodes))  # sigmoid mapping to (0,1)
        nodes = np.sort(nodes)  # Ensure strictly increasing
        
        # Weights from log-weights
        log_weights = x[8:16]
        w_max = np.max(log_weights)
        exp_w = np.exp(log_weights - w_max)
        weights = exp_w / np.sum(exp_w)
        
        # Check constraints
        if not np.all(nodes > 1e-15) or not np.all(nodes < 1 - 1e-15):
            return 1e10
        if not np.all(np.diff(nodes) > 1e-15):
            return 1e10
        if not np.all(weights > 0):
            return 1e10
        
        # Compute worst-case relative error
        max_rel_err = 0.0
        for w0, wa in cpl_models:
            for z in z_values:
                true_val = _true_integral(w0, wa, z)
                quad_val = _quadrature_value(nodes, weights, w0, wa, z)
                if true_val > 1e-10:
                    rel_err = abs(quad_val - true_val) / true_val
                    if rel_err > max_rel_err:
                        max_rel_err = rel_err
        return max_rel_err
    
    # Initialize with GL rule
    def _encode(nodes, weights):
        """Encode nodes and weights into parameter vector"""
        # Nodes: use inverse sigmoid
        raw_nodes = np.log(nodes / (1.0 - nodes))
        # Weights: use log-weights
        log_weights = np.log(weights)
        return np.concatenate([raw_nodes, log_weights])
    
    x0 = _encode(best_nodes, best_weights)
    
    # Try L-BFGS-B from GL initial point
    if time.time() - start_time < deadline - 15:
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_obj:
                best_obj = res.fun
                raw_nodes = res.x[:8]
                candidate_nodes = np.sort(1.0 / (1.0 + np.exp(-raw_nodes)))
                log_weights = res.x[8:16]
                w_max = np.max(log_weights)
                exp_w = np.exp(log_weights - w_max)
                candidate_weights = exp_w / np.sum(exp_w)
                
                if np.all(candidate_nodes > 1e-15) and np.all(candidate_nodes < 1 - 1e-15) and \
                   np.all(np.diff(candidate_nodes) > 1e-15) and np.all(candidate_weights > 0):
                    best_nodes = candidate_nodes
                    best_weights = candidate_weights
        except Exception:
            pass
    
    # Try differential evolution for global search
    if time.time() - start_time < deadline - 10:
        try:
            bounds = [(-10, 10)] * 8 + [(-5, 5)] * 8
            de_res = differential_evolution(_objective, bounds, 
                                           maxiter=50, popsize=20, 
                                           tol=1e-6, seed=42, 
                                           polish=False,
                                           init='latinhypercube')
            if de_res.fun < best_obj:
                best_obj = de_res.fun
                raw_nodes = de_res.x[:8]
                candidate_nodes = np.sort(1.0 / (1.0 + np.exp(-raw_nodes)))
                log_weights = de_res.x[8:16]
                w_max = np.max(log_weights)
                exp_w = np.exp(log_weights - w_max)
                candidate_weights = exp_w / np.sum(exp_w)
                
                if np.all(candidate_nodes > 1e-15) and np.all(candidate_nodes < 1 - 1e-15) and \
                   np.all(np.diff(candidate_nodes) > 1e-15) and np.all(candidate_weights > 0):
                    best_nodes = candidate_nodes
                    best_weights = candidate_weights
        except Exception:
            pass
    
    # Final local refinement with Nelder-Mead
    if time.time() - start_time < deadline - 5:
        try:
            x_current = _encode(best_nodes, best_weights)
            nm_res = minimize(_objective, x_current, method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if nm_res.fun < best_obj:
                best_obj = nm_res.fun
                raw_nodes = nm_res.x[:8]
                candidate_nodes = np.sort(1.0 / (1.0 + np.exp(-raw_nodes)))
                log_weights = nm_res.x[8:16]
                w_max = np.max(log_weights)
                exp_w = np.exp(log_weights - w_max)
                candidate_weights = exp_w / np.sum(exp_w)
                
                if np.all(candidate_nodes > 1e-15) and np.all(candidate_nodes < 1 - 1e-15) and \
                   np.all(np.diff(candidate_nodes) > 1e-15) and np.all(candidate_weights > 0):
                    best_nodes = candidate_nodes
                    best_weights = candidate_weights
        except Exception:
            pass
    
    # Final validation and normalization
    best_weights = best_weights / np.sum(best_weights)
    
    # Verify all constraints
    if not (len(best_nodes) == 8 and len(best_weights) == 8 and
            np.all(best_nodes > 0) and np.all(best_nodes < 1) and
            np.all(np.diff(best_nodes) > 0) and
            np.all(best_weights > 0) and
            abs(np.sum(best_weights) - 1.0) < 1e-12):
        best_nodes = nodes_gl.copy()
        best_weights = weights_gl.copy()
        best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}

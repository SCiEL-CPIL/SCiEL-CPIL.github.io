"""Cosmology: universal luminosity-distance quadrature (8-node positive rule)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

_START = time.time()
_DEADLINE = _START + 85.0

def _time_left():
    return _DEADLINE - time.time()

def _E_of_z_vec(z, Om0, Ol0, w0, w1):
    """E(z) = H(z)/H0 for flat CPL model. Vectorized."""
    # z: array of redshifts
    # Om0, Ol0, w0, w1: scalars or arrays broadcastable to z
    a = 1.0 / (1.0 + z)
    a3 = a ** 3
    # CPL: w(a) = w0 + w1(1-a)
    # integral of 3(1+w(a)) da / a = 3(1+w0)ln(a) + 3w1(1-a)
    # E^2 = Om0/a^3 + Ol0 * a^{-3(1+w0)} * exp(3w1(1-a))
    exponent = 3.0 * w1 * (1.0 - a)
    de_factor = a ** (-3.0 * (1.0 + w0)) * np.exp(exponent)
    E2 = Om0 / a3 + Ol0 * de_factor
    return np.sqrt(np.maximum(E2, 1e-300))

def construct():
    # 1. Generate Design Set
    # We want to cover the plausible range of cosmological parameters
    # Om0 in [0.2, 0.4], Ol0 = 1 - Om0
    # w0 in [-1.5, -0.5], w1 in [-1.0, 1.0]
    # z_max in [0.1, 5.0]
    
    # Grid points
    Om0_vals = np.linspace(0.20, 0.40, 5)
    w0_vals = np.linspace(-1.5, -0.5, 5)
    w1_vals = np.linspace(-1.0, 1.0, 5)
    z_max_vals = np.logspace(np.log10(0.1), np.log10(5.0), 15)
    
    design_cosmos = []
    true_vals = []
    z_max_arr = []
    params_arr = []
    
    # Iterate over grid to create a dense design set
    # To keep computation fast, we limit the number of points
    # 5*5*5 = 125 cosmologies * 15 z_max = 1875 points. This is manageable.
    
    for Om0 in Om0_vals:
        Ol0 = 1.0 - Om0
        for w0 in w0_vals:
            for w1 in w1_vals:
                for z_max in z_max_vals:
                    def integrand_u(u, z_max=z_max, Om0=Om0, Ol0=Ol0, w0=w0, w1=w1):
                        z_val = z_max * u
                        if z_val < 1e-15:
                            return 1.0
                        return 1.0 / _E_of_z_vec(z_val, Om0, Ol0, w0, w1)
                    
                    result, _ = quad(integrand_u, 0, 1, limit=200, epsabs=1e-10, epsrel=1e-10)
                    true_val = z_max * result
                    
                    design_cosmos.append((Om0, Ol0, w0, w1))
                    true_vals.append(true_val)
                    z_max_arr.append(z_max)
                    params_arr.append((Om0, Ol0, w0, w1))
                    
    true_vals = np.array(true_vals)
    z_max_arr = np.array(z_max_arr)
    params_arr = np.array(params_arr)
    n_design = len(true_vals)
    
    # Precompute 1/E for all design points and all possible node positions?
    # No, nodes change. We need to evaluate 1/E(z_max * node) on the fly.
    
    def _get_inv_E(nodes):
        """
        Compute 1/E for all design points given a set of nodes.
        nodes: (8,)
        Returns: (n_design, 8) array of 1/E values.
        """
        # z_eval: (n_design, 8)
        z_eval = z_max_arr[:, None] * nodes[None, :]
        # Flatten for vectorized E calculation? Or use broadcasting.
        # E_of_z_vec expects z, and params.
        # params_arr is (n_design, 4)
        # We need to broadcast params to (n_design, 8, 4)?
        
        # Let's do it in chunks or fully vectorized.
        # z_eval shape: (N, 8)
        # a shape: (N, 8)
        a = 1.0 / (1.0 + z_eval)
        a3 = a ** 3
        
        # params_arr: (N, 4) -> (N, 1, 4)
        Om0 = params_arr[:, 0:1]
        Ol0 = params_arr[:, 1:2]
        w0 = params_arr[:, 2:3]
        w1 = params_arr[:, 3:4]
        
        exponent = 3.0 * w1 * (1.0 - a)
        de_factor = a ** (-3.0 * (1.0 + w0)) * np.exp(exponent)
        E2 = Om0 / a3 + Ol0 * de_factor
        E = np.sqrt(np.maximum(E2, 1e-300))
        return 1.0 / E

    def _objective(raw_params):
        """
        raw_params: (16,)
        First 8: logit nodes? Or just nodes with penalty?
        Let's use nodes directly with bounds and penalty for ordering.
        Last 8: log-weights (softmax).
        """
        if _time_left() < 2.0:
            return 1e10 # Time out penalty
            
        x_nodes = raw_params[:8]
        log_w = raw_params[8:]
        
        # Ensure nodes are within (0, 1)
        # We will use bounds in the optimizer, but clip here for safety
        x_nodes = np.clip(x_nodes, 1e-6, 1.0 - 1e-6)
        
        # Check ordering
        if np.any(np.diff(x_nodes) <= 0):
            return 1e10 + np.sum(np.diff(x_nodes))
            
        # Weights via softmax
        log_w = log_w - np.max(log_w) # for numerical stability
        w = np.exp(log_w)
        w = w / np.sum(w)
        
        # Compute approx integral
        inv_E = _get_inv_E(x_nodes) # (N, 8)
        approx = z_max_arr * (inv_E @ w)
        
        # Relative error
        rel_err = np.abs(approx - true_vals) / np.maximum(np.abs(true_vals), 1e-300)
        
        # Smooth max
        alpha = 50.0
        smooth_max = (1.0 / alpha) * np.log(np.sum(np.exp(alpha * rel_err)))
        
        return smooth_max

    # Initial guess
    # Gauss-Legendre nodes on [0, 1]
    gl_nodes, _ = np.polynomial.legendre.leggauss(8)
    x_init = 0.5 * (gl_nodes + 1.0)
    w_init = np.ones(8) / 8.0
    log_w_init = np.log(w_init)
    
    raw_init = np.concatenate([x_init, log_w_init])
    
    bounds = [(1e-6, 1.0 - 1e-6)] * 8 + [(-np.inf, np.inf)] * 8
    
    best_result = None
    best_err = np.inf
    
    # Optimization
    try:
        res = minimize(_objective, raw_init, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if res.success or res.fun < best_err:
            best_result = res
            best_err = res.fun
    except Exception as e:
        print(f"Optimization failed: {e}")
        
    # If L-BFGS-B failed or didn't improve, try Nelder-Mead on the smooth objective?
    # Or just use the initial guess if optimization failed.
    if best_result is None:
        best_result = res if 'res' in locals() else None
        if best_result is None:
             # Fallback to initial guess
             best_result = type('obj', (object,), {'x': raw_init, 'fun': _objective(raw_init)})()

    # Extract nodes and weights
    x_nodes = np.clip(best_result.x[:8], 1e-6, 1.0 - 1e-6)
    log_w = best_result.x[8:]
    log_w = log_w - np.max(log_w)
    w = np.exp(log_w)
    w = w / np.sum(w)
    
    # Ensure strict ordering
    order = np.argsort(x_nodes)
    x_nodes = x_nodes[order]
    w = w[order]
    
    # If ordering was violated significantly, the penalty should have prevented it,
    # but let's enforce it strictly for the return.
    if np.any(np.diff(x_nodes) <= 0):
        # This should not happen if penalty worked, but just in case
        x_nodes = np.linspace(0.05, 0.95, 8)
        w = np.ones(8) / 8.0

    # Final checks
    assert np.all(np.diff(x_nodes) > 0), "Nodes must be strictly increasing"
    assert np.all(w > 0), "Weights must be positive"
    assert abs(np.sum(w) - 1.0) < 1e-12, "Weights must sum to 1"

    return {"nodes": x_nodes, "weights": w}
# EVOLVE-BLOCK-END

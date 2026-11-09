"""Chebyshev-seeded differential evolution for luminosity distance quadrature."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.integrate import quad

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _time_left():
    return _DEADLINE - time.time()

def _E_cpl(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    Om_de = 1.0 - Om_m - Om_r
    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    cpl_factor = 1.0 + wa * (1.0 - a)
    term_de = Om_de * a**(-3.0 * (1.0 + w0 + wa)) * (cpl_factor**3)
    val = term_m + term_r + term_de
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)

def _true_dL(z, w0, wa):
    def integrand(z_prime):
        a = 1.0 / (1.0 + z_prime)
        return 1.0 / _E_cpl(a, w0, wa)
    val, _ = quad(integrand, 0, z, limit=100, epsabs=1e-12, epsrel=1e-12)
    return (1.0 + z) * val

def _quad_dL(z, nodes, weights, w0, wa):
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _compute_max_rel_error(nodes, weights, test_cosmos, test_zs):
    """Compute the maximum relative error across test cases."""
    max_err = 0.0
    for w0, wa in test_cosmos:
        for z in test_zs:
            true_val = _true_dL(z, w0, wa)
            quad_val = _quad_dL(z, nodes, weights, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                if rel_err > max_err:
                    max_err = rel_err
    return max_err

def _params_to_nodes_weights(params):
    """
    Params 0-7: sigmoid-transformed node positions in (0,1), strictly increasing.
    Params 8-15: log-weights (normalized to sum to 1).
    
    Uses cumulative sigmoid mapping to ensure strict ordering.
    """
    # Nodes: use cumulative sum of sigmoid values to ensure strict increase
    raw = params[:8]
    # Map to (0,1) via sigmoid, then create cumulative positions
    sig_vals = 1.0 / (1.0 + np.exp(-10.0 * (raw - 0.5)))
    # Create strictly increasing nodes via cumulative sum with normalization
    # Ensure each increment is positive
    increments = np.abs(sig_vals) + 1e-6
    cumsum = np.cumsum(increments)
    nodes = cumsum / cumsum[-1]
    # Shift slightly to avoid exact 0 or 1
    nodes = 1e-10 + nodes * (1.0 - 2e-10)
    
    # Weights
    log_weights = params[8:16]
    w_raw = np.exp(log_weights)
    weights = w_raw / np.sum(w_raw)
    
    return nodes, weights

def _objective(params):
    if _time_left() < 3.0:
        return 1e10
    
    nodes, weights = _params_to_nodes_weights(params)
    
    # Test set: representative cosmologies and redshifts
    # Use fewer points for speed, focus on challenging cases
    test_cosmos = [
        (0.0, 0.0),    # LCDM
        (-1.0, 0.0),   # Lambda
        (0.5, -1.0),   # Quintessence
        (-1.5, 0.5),   # Phantom
        (-0.5, 0.5),   # Another CPL
        (0.3, -0.3),   # Mild CPL
        (1.0, -1.0),   # Strong quintessence
        (-2.0, 1.0),   # Strong phantom
    ]
    test_zs = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0])
    
    return _compute_max_rel_error(nodes, weights, test_cosmos, test_zs)

def _chebyshev_init():
    """Get 8-point Chebyshev nodes and weights on [0,1]."""
    n = 8
    # Chebyshev nodes on [-1,1]: cos((2k+1)*pi/(2n))
    k = np.arange(n)
    x_cheb = np.cos((2*k + 1) * np.pi / (2*n))
    # Map to [0,1]
    nodes = 0.5 * (x_cheb + 1.0)
    nodes = np.sort(nodes)  # Ensure increasing
    
    # Weights for Chebyshev quadrature on [-1,1]
    w_cheb = np.pi / n * np.ones(n)
    # Adjust for the mapping: weights scale by 0.5
    weights = 0.5 * w_cheb
    weights = weights / np.sum(weights)
    
    return nodes, weights

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Get Chebyshev initial nodes and weights
    cheb_nodes, cheb_weights = _chebyshev_init()
    
    # Create initial parameter vector
    # For nodes: invert the sigmoid-cumsum mapping approximately
    # Start with raw values that produce nodes near Chebyshev positions
    params_init = np.zeros(16)
    
    # Set node params to values that roughly produce Chebyshev-like distribution
    # The sigmoid-cumsum mapping is complex, so use a simple heuristic
    # Set raw values spread out to create increasing nodes
    for i in range(8):
        params_init[i] = 0.5 + (cheb_nodes[i] - 0.5) * 2.0
    
    # Set weight params to log of Chebyshev weights
    params_init[8:16] = np.log(cheb_weights)
    
    # Define bounds for differential evolution
    bounds = [(-2.0, 2.0)] * 8 + [(-3.0, 3.0)] * 8
    
    best_params = params_init.copy()
    best_score = _objective(best_params)
    
    # Run differential evolution with time limit
    if _time_left() > 30.0:
        try:
            result = differential_evolution(
                _objective, bounds,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.7,
                seed=42,
                polish=True,
                init='sobol'
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Polish with local optimization if time permits
    if _time_left() > 15.0:
        try:
            from scipy.optimize import minimize
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Final decode
    nodes, weights = _params_to_nodes_weights(best_params)
    
    # Ensure validity: strictly increasing nodes in (0,1)
    nodes = np.sort(nodes)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    
    # Ensure positive weights summing to 1
    weights = np.abs(weights)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END

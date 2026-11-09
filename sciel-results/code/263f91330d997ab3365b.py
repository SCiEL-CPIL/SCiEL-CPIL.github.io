"""Combined strategy: Robust optimization of 8-node quadrature for luminosity distance."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # 5s safety margin before 90s kill

def _time_left():
    return _DEADLINE - time.time()

def _E_cpl(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    """
    Compute E(a) = H(a)/H0 for a flat CPL universe.
    Om_m + Om_r + Om_de = 1
    E^2(a) = Om_m a^-3 + Om_r a^-4 + Om_de a^(-3(1+w0+wa)) (1 + wa(1-a))^3
    """
    Om_de = 1.0 - Om_m - Om_r
    # Handle a=0 singularity if a is very small, but a_i are in (0,1)
    # a = 1/(1+z*x), z>0, x>0 => a < 1. a > 0 always.
    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    # CPL term: a^(-3(1+w0+wa)) * (1 + wa(1-a))^3
    # Note: (1+wa(1-a)) can be negative if wa is large negative and a small, but 
    # for reasonable cosmology parameters it stays positive. 
    # To be safe against sqrt of negative due to numerical issues, we clip.
    cpl_factor = 1.0 + wa * (1.0 - a)
    term_de = Om_de * a**(-3.0 * (1.0 + w0 + wa)) * (cpl_factor**3)
    
    val = term_m + term_r + term_de
    # Clip to avoid negative due to floating point errors
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)

def _true_dL(z, w0, wa):
    """
    Compute true dimensionless luminosity distance dL/(c/H0).
    dL/(c/H0) = (1+z) * integral_0^z z' / E(a') dz'
    Change variable u = z', a = 1/(1+z u).
    Integral is int_0^1 u / E(1/(1+z*u)) du * z? No.
    
    Let's stick to the definition in the prompt:
    dL/(c/H0) = (1+z) * z * sum(weights / E(a_i)) where a_i = 1/(1+z*x_i)
    This implies the integral being approximated is:
    I = int_0^1 (1 / E(1/(z*u))) du ? 
    Let's check the standard formula.
    d_L = c/H0 * (1+z) * int_0^z dz' / H(z')
    H(z') = H0 * E(1/(1+z'))
    So d_L/(c/H0) = (1+z) * int_0^z dz' / E(1/(1+z'))
    
    The prompt says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    Let u = z'/z, so z' = z*u, dz' = z du.
    int_0^z dz'/E(1/(1+z')) = int_0^1 z du / E(1/(1+z*u))
    So dL/(c/H0) = (1+z) * z * int_0^1 du / E(1/(1+z*u))
    
    Yes, the integrand for the quadrature over u in [0,1] is f(u) = 1 / E(1/(1+z*u)).
    The quadrature approximates int_0^1 f(u) du = sum w_i f(x_i).
    """
    def integrand(z_prime):
        a = 1.0 / (1.0 + z_prime)
        return 1.0 / _E_cpl(a, w0, wa)
    
    val, _ = quad(integrand, 0, z, limit=100, epsabs=1e-12, epsrel=1e-12)
    return (1.0 + z) * val

def _quad_dL(z, nodes, weights, w0, wa):
    """
    Compute quadrature approximation.
    dL/(c/H0) = (1+z) * z * sum_i weights[i] / E(1/(1+z*nodes[i]))
    """
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _params_to_nodes_weights(params):
    """
    Convert 16 params to 8 nodes and 8 weights.
    Params 0-7: log-gaps for nodes.
    Params 8-15: log-weights.
    """
    # Nodes
    log_gaps = params[:8]
    gaps = np.exp(log_gaps)
    total_gap = np.sum(gaps)
    cum_gaps = np.cumsum(gaps)
    nodes = cum_gaps / total_gap
    
    # Weights
    log_weights = params[8:16]
    w_raw = np.exp(log_weights)
    weights = w_raw / np.sum(w_raw)
    
    return nodes, weights

def _objective(params):
    if _time_left() < 1.0:
        return 1e10
    
    nodes, weights = _params_to_nodes_weights(params)
    
    # Test set: representative cosmologies and redshifts
    # We use a subset for speed during optimization
    test_cosmos = [
        (0.0, 0.0),    # LCDM
        (-1.0, 0.0),   # Lambda
        (0.5, -1.0),   # Quintessence
        (-1.5, 0.5),   # Phantom
        (-0.5, 0.5),   # Another CPL
    ]
    test_zs = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0])
    
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

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Initial guess: Midpoint rule
    n = 8
    mid_nodes = (np.arange(n) + 0.5) / n
    mid_weights = np.full(n, 1.0 / n)
    
    # Inverse transform for initial params
    # nodes_i = cumsum(exp(p_j)) / sum(exp(p_j))
    # This is hard to invert exactly for arbitrary nodes, 
    # but we can just use log-gaps derived from differences.
    # For midpoint, gaps are roughly equal.
    # Let's just start with params that produce midpoint-like nodes.
    # If we set all log_gaps to 0, gaps=1, nodes = [1/8, 2/8, ..., 8/8=1].
    # Wait, nodes must be in [0,1] strictly increasing. 
    # My parameterization nodes = cumsum(gaps)/sum(gaps) puts the last node at 1.0.
    # The prompt requires nodes in [0,1] strictly increasing. It doesn't say they can't be 1? 
    # "Nodes must be strictly increasing in [0,1]". Usually implies open interval or at least distinct.
    # If node is 1, a = 1/(1+z). E(1/(1+z)) is well defined.
    # However, standard quadrature nodes are often interior.
    # Let's adjust parameterization to ensure nodes < 1.
    # Use 9 gaps? Or normalize differently?
    # Let's use: nodes_i = (cumsum(gaps[:i+1]) - gaps[i]/2) / sum(gaps) ? No.
    # Simplest: nodes = cumsum(gaps) / (sum(gaps) + 1.0) ? No, that shifts everything.
    
    # Let's stick to the 8-gap parameterization but interpret the "nodes" as centers of bins?
    # No, the formula uses specific a_i.
    # If the last node is 1.0, it's valid.
    # Let's check if "strictly increasing in [0,1]" allows 1.0. Yes, [0,1] is closed.
    # But usually "in [0,1]" for quadrature might imply interior to avoid boundary issues if integrand is singular at 1?
    # E(a) is singular at a=0 (z=infinity). At a=1 (z=0), E(1) is finite.
    # So node=1 is fine.
    
    # Initial params for midpoint:
    # We want nodes approx [0.0625, 0.1875, ...]
    # If we use gaps, it's easier to just start with random positive gaps or equal gaps.
    # Equal gaps -> nodes [1/8, 2/8, ..., 1].
    # This is the right-endpoint rule, not midpoint.
    # To get midpoint, we can shift.
    # Let's just use the optimizer to find the best nodes.
    
    # Initial params
    # log_gaps = 0 -> gaps=1 -> nodes = [1/8, ..., 1]
    # log_weights = 0 -> weights = [1/8, ..., 1/8]
    params_init = np.zeros(16)
    
    # Run Optimization
    best_params = params_init.copy()
    best_score = _objective(best_params)
    
    # Try a few restarts or different methods
    methods = ['Nelder-Mead', 'L-BFGS-B']
    
    for method in methods:
        if _time_left() < 5.0:
            break
            
        try:
            if method == 'Nelder-Mead':
                res = minimize(_objective, params_init, method=method, 
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            else:
                res = minimize(_objective, params_init, method=method, 
                               options={'maxiter': 200, 'gtol': 1e-8})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception as e:
            pass
            
    # Final check
    nodes, weights = _params_to_nodes_weights(best_params)
    
    # Ensure weights sum to 1 exactly (within float precision)
    weights = weights / np.sum(weights)
    
    # Ensure nodes are strictly increasing in [0,1]
    # The parameterization guarantees this if gaps > 0.
    # Clip to [0,1] just in case of numerical drift
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END

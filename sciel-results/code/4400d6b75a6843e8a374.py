"""
Strategy: Deterministic local refinement of Gauss-Legendre and Gauss-Chebyshev 
initial guesses using Nelder-Mead to minimize worst-case error on a curated 
set of difficult cosmological models.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def _E_cpl(a, w0, wa, om_m, om_r, om_k=0.0):
    """E(a) = H(a)/H0 for CPL model."""
    om_de0 = 1.0 - om_m - om_r - om_k
    # H^2/H0^2 = om_m a^-3 + om_r a^-4 + om_de0 a^(-3(1+w0+wa)) + om_k
    h2 = om_m * a**(-3) + om_r * a**(-4) + om_k + om_de0 * a**(-3*(1+w0+wa))
    return np.sqrt(np.maximum(h2, 1e-30))

def _compute_worst_error(nodes, weights, models):
    """
    Compute the worst relative error for the given quadrature rule 
    against the provided models.
    models: list of (z, w0, wa, om_m, om_r, om_k)
    """
    if len(nodes) != 8 or len(weights) != 8:
        return 1e10
    
    # Check validity
    if not np.all(np.diff(nodes) > 1e-8):
        return 1e10
    if nodes[0] < 1e-8 or nodes[-1] > 1.0 - 1e-8:
        return 1e10
    if not np.all(weights > 0):
        return 1e10
    if abs(np.sum(weights) - 1.0) > 1e-6:
        return 1e10
        
    max_err = 0.0
    
    # Precompute GL reference nodes/weights for high precision reference
    # Using 128-point GL for reference to be extremely accurate
    gl_nodes_ref, gl_weights_ref = np.polynomial.legendre.leggauss(128)
    gl_nodes_ref = (gl_nodes_ref + 1.0) / 2.0
    gl_weights_ref = gl_weights_ref / 2.0
    
    for z, w0, wa, om_m, om_r, om_k in models:
        # 1. Calculate True Value (Reference)
        # Integrand: f(x) = 1 / E(a(x)) where a = 1/(1+z*x)
        # We integrate over x in [0,1]
        
        def integrand_ref(x):
            a = 1.0 / (1.0 + z * x)
            return 1.0 / _E_cpl(a, w0, wa, om_m, om_r, om_k)
        
        # Use vectorized GL for speed
        a_gl = 1.0 / (1.0 + z * gl_nodes_ref)
        E_gl = _E_cpl(a_gl, w0, wa, om_m, om_r, om_k)
        f_gl = 1.0 / E_gl
        I_true = np.dot(f_gl, gl_weights_ref)
        
        # 2. Calculate Quadrature Value
        a_nodes = 1.0 / (1.0 + z * nodes)
        E_nodes = _E_cpl(a_nodes, w0, wa, om_m, om_r, om_k)
        f_nodes = 1.0 / E_nodes
        Q_val = np.dot(f_nodes, weights)
        
        # 3. Relative Error
        if I_true == 0:
            err = 0.0
        else:
            err = abs(Q_val - I_true) / I_true
            
        if err > max_err:
            max_err = err
            
    return max_err

def construct():
    """
    Constructs an 8-node positive quadrature rule on [0,1].
    """
    deadline = time.time() + 85.0 # 5s safety margin
    
    # 1. Define a "hard" set of models for optimization
    # We want to cover the likely validation space:
    # z: 0.0 to 10 (high z is harder)
    # w0, wa: -2 to 2 (extreme dark energy)
    # om_m: 0.2 to 0.4
    
    models = []
    # Include specific challenging cases
    # Low z, standard LCDM
    models.append((0.1, -1.0, 0.0, 0.3, 9e-5, 0.0))
    models.append((1.0, -1.0, 0.0, 0.3, 9e-5, 0.0))
    models.append((5.0, -1.0, 0.0, 0.3, 9e-5, 0.0))
    models.append((10.0, -1.0, 0.0, 0.3, 9e-5, 0.0))
    
    # Extreme w0, wa
    models.append((1.0, -1.5, 1.5, 0.3, 9e-5, 0.0))
    models.append((1.0, 1.5, -1.5, 0.3, 9e-5, 0.0))
    models.append((5.0, -2.0, 2.0, 0.3, 9e-5, 0.0))
    models.append((5.0, 2.0, -2.0, 0.3, 9e-5, 0.0))
    
    # Different matter densities
    models.append((2.0, -1.0, 0.0, 0.2, 9e-5, 0.0))
    models.append((2.0, -1.0, 0.0, 0.4, 9e-5, 0.0))
    
    # Add some random sampling to ensure robustness
    np.random.seed(42)
    for _ in range(20):
        z = np.random.uniform(0.1, 8.0)
        w0 = np.random.uniform(-2.0, 2.0)
        wa = np.random.uniform(-2.0, 2.0)
        om_m = np.random.uniform(0.2, 0.4)
        models.append((z, w0, wa, om_m, 9e-5, 0.0))

    # 2. Define the objective function for the optimizer
    # We optimize nodes and log-weights to ensure positivity and sum-to-1 constraint is handled by normalization in the error calc?
    # No, better to optimize raw nodes and weights, but enforce constraints in the loss function.
    # However, Nelder-Mead works best in unconstrained space.
    # Let's use a transformation:
    # nodes: use cumulative sums of positive deltas? Or just optimize nodes directly and penalize violations heavily.
    # Given Nelder-Mead is robust, we can optimize 8 nodes and 8 weights directly, 
    # but we must ensure weights > 0 and sum = 1.
    # A common trick: optimize log_weights, then normalize.
    # For nodes, we can optimize them directly but penalize if out of [0,1] or not sorted.
    
    def objective(params):
        nodes = params[:8]
        log_weights = params[8:]
        
        # Basic validity checks with penalties
        if not np.all(nodes > 0) or not np.all(nodes < 1):
            return 1e10 + np.sum((nodes - 0.5)**2) # Push towards center
        if not np.all(np.diff(nodes) > 1e-6):
            return 1e10 + np.sum(np.diff(nodes)) # Push towards separation
            
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)
        
        err = _compute_worst_error(nodes, weights, models)
        return err

    # 3. Initial Guesses
    # Guess 1: Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0
    x0_gl = np.concatenate([gl_nodes, np.log(gl_weights)])
    
    # Guess 2: Gauss-Chebyshev (Type 1)
    # Nodes: cos((2k-1)pi/16) mapped to [0,1]?
    # Standard Chebyshev nodes on [-1,1] are cos((2k-1)pi/(2n)).
    # Map to [0,1]: x = (t+1)/2
    n = 8
    k = np.arange(1, n+1)
    t_cheb = np.cos((2*k - 1) * np.pi / (2*n))
    cheb_nodes = (t_cheb + 1.0) / 2.0
    cheb_nodes = np.sort(cheb_nodes) # Ensure increasing order
    # Weights for Chebyshev quadrature on [-1,1] are pi/n.
    # Mapped to [0,1], weights are pi/(2n).
    cheb_weights = np.full(n, np.pi / (2*n))
    cheb_weights = cheb_weights / np.sum(cheb_weights) # Normalize just in case
    x0_cheb = np.concatenate([cheb_nodes, np.log(cheb_weights)])
    
    # Guess 3: Uniform
    uni_nodes = np.linspace(1/16, 15/16, 8)
    uni_weights = np.ones(8) / 8.0
    x0_uni = np.concatenate([uni_nodes, np.log(uni_weights)])
    
    best_params = None
    best_score = 1e10
    
    # 4. Optimization Loop
    # Try Nelder-Mead from each initial guess
    initial_guesses = [x0_gl, x0_cheb, x0_uni]
    
    for x0 in initial_guesses:
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if res.success or res.fun < best_score:
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x
        except Exception:
            continue

    # 5. Final Polish / Validation
    if best_params is None:
        # Fallback to GL if optimization failed completely
        best_nodes = gl_nodes
        best_weights = gl_weights
    else:
        best_nodes = best_params[:8]
        log_weights = best_params[8:]
        best_weights = np.exp(log_weights)
        best_weights = best_weights / np.sum(best_weights)
        
        # Ensure strict increasing order and bounds
        # If Nelder-Mead slightly violated, we can try to repair
        # But usually with the penalty, it stays valid.
        # Let's do a final check.
        if not (np.all(best_nodes > 0) and np.all(best_nodes < 1) and np.all(np.diff(best_nodes) > 1e-6)):
             # If invalid, fallback to GL
             best_nodes = gl_nodes
             best_weights = gl_weights
             
    # Final normalization check
    best_weights = best_weights / np.sum(best_weights)
    
    return {
        "nodes": best_nodes,
        "weights": best_weights
    }
# EVOLVE-BLOCK-END

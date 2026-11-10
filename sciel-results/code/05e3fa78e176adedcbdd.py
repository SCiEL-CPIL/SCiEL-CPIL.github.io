"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def construct():
    t_start = time.time()
    deadline = t_start + 82.0  # 8s safety margin before 90s kill
    n = 8
    
    # Precompute a dense validation grid
    z_vals = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
    Om_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    w0_vals = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    wa_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    # Build full grid of (z, Om, w0, wa)
    Z, OM, W0, WA = np.meshgrid(z_vals, Om_vals, w0_vals, wa_vals, indexing='ij')
    Z_flat = Z.ravel()
    OM_flat = OM.ravel()
    W0_flat = W0.ravel()
    WA_flat = WA.ravel()
    N_eval = len(Z_flat)
    
    # Precompute reference integrals for the entire grid
    refs = np.zeros(N_eval)
    for i in range(N_eval):
        z = Z_flat[i]
        Om = OM_flat[i]
        w0 = W0_flat[i]
        wa = WA_flat[i]
        Ode = 1.0 - Om
        def integrand(u):
            a = 1.0 / (1.0 + u)
            E_sq = Om * a**(-3) + Ode * a**(3.0*(1.0+w0+wa))
            return 1.0 / np.sqrt(E_sq)
        val, _ = quad(integrand, 0.0, z, limit=200)
        refs[i] = (1.0 + z) * val
    
    def compute_errors(nodes, weights):
        """Vectorized error computation over the full grid."""
        # nodes: (8,), weights: (8,)
        # Z_flat: (N,), compute a_i for all combos
        # a = 1/(1 + z * node)
        # E^2 = Om * a^-3 + (1-Om) * a^(3*(1+w0+wa))
        
        # Shape: (N_eval, 8)
        a_vals = 1.0 / (1.0 + Z_flat[:, None] * nodes[None, :])
        # (N_eval, 8)
        E_sq = OM_flat[:, None] * a_vals**(-3) + (1.0 - OM_flat[:, None]) * a_vals**(3.0*(1.0 + W0_flat[:, None] + WA_flat[:, None]))
        E_vals = np.sqrt(np.maximum(E_sq, 1e-30))
        inv_E = 1.0 / E_vals  # (N_eval, 8)
        
        quad_vals = (1.0 + Z_flat) * (inv_E @ weights)  # (N_eval,)
        rel_errors = np.abs(quad_vals - refs) / np.maximum(refs, 1e-30)
        return rel_errors
    
    def objective(params):
        """params: 8 node gaps (log-spaced) + 8 log-weights. Total 16 params."""
        if time.time() > deadline:
            return 1e10
        
        # Decode nodes from gap parameters
        # g_i = log(x_{i+1} - x_i) for i=0..6, and x_0 = sigmoid(g_7)
        # Actually, let's use: x_0 = sigmoid(p0), gaps = exp(p1..p7)
        # Then x_i = x_0 + sum_{j=1}^{i} exp(p_j)
        # Need x_7 < 1
        
        p = params
        x0 = 1.0 / (1.0 + np.exp(-p[0]))  # first node
        gaps = np.exp(p[1:8])  # 7 gaps
        nodes = np.zeros(8)
        nodes[0] = x0
        for i in range(1, 8):
            nodes[i] = nodes[i-1] + gaps[i-1]
        
        # Check validity
        if nodes[-1] >= 1.0 or nodes[0] <= 0.0:
            return 1e10
        if np.any(np.diff(nodes) <= 1e-12):
            return 1e10
        
        # Decode weights
        log_w = p[8:16]
        w_unnorm = np.exp(log_w - np.max(log_w))  # numerical stability
        weights = w_unnorm / np.sum(w_unnorm)
        
        errors = compute_errors(nodes, weights)
        
        # Smooth max (inf-norm approximation)
        tau = 0.005
        max_e = np.max(errors)
        soft_max = max_e + tau * np.log(np.sum(np.exp((errors - max_e) / tau)))
        return soft_max
    
    # Multiple initializations
    rng = np.random.RandomState(42)
    
    # Strategy 1: Start from Gauss-Legendre nodes/weights
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n)
    gl_nodes = 0.5 * (gl_nodes + 1.0)  # map from [-1,1] to [0,1]
    gl_weights = gl_weights / 2.0  # rescale
    
    # Convert to gap parameterization
    p_gl = np.zeros(16)
    p_gl[0] = np.log(gl_nodes[0] / (1.0 - gl_nodes[0]))  # inverse sigmoid
    for i in range(1, 8):
        gap = gl_nodes[i] - gl_nodes[i-1]
        p_gl[i] = np.log(max(gap, 1e-12))
    for i in range(8):
        p_gl[8+i] = np.log(max(gl_weights[i], 1e-12))
    
    # Strategy 2: Midpoint rule
    mid_nodes = (np.arange(n) + 0.5) / n
    mid_weights = np.full(n, 1.0/n)
    p_mid = np.zeros(16)
    p_mid[0] = np.log(mid_nodes[0] / (1.0 - mid_nodes[0]))
    for i in range(1, 8):
        gap = mid_nodes[i] - mid_nodes[i-1]
        p_mid[i] = np.log(max(gap, 1e-12))
    for i in range(8):
        p_mid[8+i] = np.log(max(mid_weights[i], 1e-12))
    
    # Strategy 3: Random perturbations of GL
    p_rand1 = p_gl + rng.normal(0, 0.1, size=16)
    p_rand2 = p_gl + rng.normal(0, 0.05, size=16)
    
    initializations = [p_gl, p_mid, p_rand1, p_rand2]
    
    best_params = None
    best_obj = np.inf
    
    for p0 in initializations:
        if time.time() > deadline - 10.0:
            break
        
        try:
            result = minimize(
                objective, p0, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
        
        # Try Nelder-Mead polish
        if time.time() > deadline - 8.0:
            break
        if best_params is not None:
            try:
                result2 = minimize(
                    objective, best_params, method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
                )
                if result2.fun < best_obj:
                    best_obj = result2.fun
                    best_params = result2.x.copy()
            except Exception:
                pass
    
    # Final polish with tighter tolerance if time permits
    if best_params is not None and time.time() < deadline - 5.0:
        try:
            result3 = minimize(
                objective, best_params, method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 20000}
            )
            if result3.fun < best_obj:
                best_obj = result3.fun
                best_params = result3.x.copy()
        except Exception:
            pass
    
    # Extract final nodes and weights
    if best_params is None:
        # Fallback to Gauss-Legendre
        nodes = 0.5 * (gl_nodes)
        weights = gl_weights
    else:
        p = best_params
        x0 = 1.0 / (1.0 + np.exp(-p[0]))
        gaps = np.exp(p[1:8])
        nodes = np.zeros(8)
        nodes[0] = x0
        for i in range(1, 8):
            nodes[i] = nodes[i-1] + gaps[i-1]
        
        log_w = p[8:16]
        w_unnorm = np.exp(log_w - np.max(log_w))
        weights = w_unnorm / np.sum(w_unnorm)
        
        # Validate
        if nodes[-1] >= 1.0 or nodes[0] <= 0.0 or np.any(np.diff(nodes) <= 1e-14):
            nodes = 0.5 * (gl_nodes)
            weights = gl_weights
    
    # Final safety
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)
    
    # Verify sum
    assert abs(np.sum(weights) - 1.0) < 1e-12
    assert np.all(np.diff(nodes) > 0)
    assert np.all(nodes > 0) and np.all(nodes < 1)
    
    return {"nodes": nodes, "weights": weights}

# EVOLVE-BLOCK-END

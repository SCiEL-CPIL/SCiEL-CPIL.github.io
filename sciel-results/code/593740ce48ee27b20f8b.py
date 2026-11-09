import time
import numpy as np
from scipy.optimize import minimize
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _time_left():
    return _DEADLINE - time.time()

def _E_cpl(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    """
    Compute E(a) = H(a)/H0 for CPL model.
    Uses the exact CPL formulation:
    E^2(a) = Om_m a^-3 + Om_r a^-4 + Om_de a^-3(1+w0+wa) [1 + w0 + wa(1-a)]^3
    Wait, the standard CPL is p_Da = 1 + w0 + wa(1-a).
    rho_de(a) = rho_de0 * p_Da^3 * a^-3(1+w0+wa) ... no.
    
    Standard CPL: w(a) = w0 + wa(1-a).
    rho_de(a) = rho_de0 * exp(3 * integral_1^a (1+w(a'))/a' da')
    integral = 3 * [ (1+w0+wa) ln(a) + wa(a-1) ]
    So rho_de(a) = rho_de0 * a^(3(1+w0+wa)) * exp(3*wa*(a-1))
    E^2 = Om_m a^-3 + Om_r a^-4 + Om_de * a^(3(1+w0+wa)) * exp(3*wa*(a-1))
    
    Let's verify the inspiration 2 formula:
    term_de = Om_de * a**(-3.0 * D) * exp_term where D = 1+w0+wa?
    Inspiration 2 says: power_term = a**(-3.0 * D) where D = 1+w0+wa.
    This implies rho_de ~ a^-3(1+w0+wa) * exp(3*wa*(a-1)).
    This matches the standard CPL definition.
    """
    Om_de = 1.0 - Om_m - Om_r
    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    
    D = 1.0 + w0 + wa
    # Handle a=0 singularity? a is in [0,1]. 
    # In our quadrature, nodes are in (0,1), so a_i > 0.
    # For true integral, we integrate from 0 to z. At z=0, a=1.
    # The integrand 1/E(a) is smooth at a=1.
    # At a->0, E(a) ~ a^-2 (radiation) or a^-1.5 (matter). 1/E is 0.
    
    # np.power(0, negative) is inf. We should avoid a=0 if possible or handle it.
    # Our grid for true integral starts at z=0 -> a=1.
    # The quadrature nodes are in (0,1) -> a in (1/(1+z), 1).
    
    exp_term = np.exp(3.0 * wa * (a - 1.0))
    # Use np.where to handle a=0 if it appears, though it shouldn't for nodes.
    # For a=0, a**(-3*D) is inf. But term_r is also inf.
    # In practice, we don't evaluate at a=0.
    power_term = a**(-3.0 * D)
    term_de = Om_de * power_term * exp_term
    
    val = term_m + term_r + term_de
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)

def _true_dL_fast(z, w0, wa, num_points=1024):
    """
    Fast reference calculation of (1+z) * integral_0^z dz'/E(a(z')).
    Using Simpson's rule on a dense grid.
    """
    if z <= 1e-12:
        return 0.0
    # Grid in z-space from 0 to z
    # Ensure even number of intervals for Simpson's
    zs = np.linspace(0, z, num_points)
    a_vals = 1.0 / (1.0 + zs)
    E_vals = _E_cpl(a_vals, w0, wa)
    integrand = 1.0 / E_vals
    
    # Simpson's Rule
    # h = z / (num_points - 1)
    h = z / (num_points - 1)
    # Simpson: h/3 * [f0 + fN + 4*sum(odd) + 2*sum(even)]
    y = integrand
    # y[0] + y[-1] + 4*sum(y[1:-1:2]) + 2*sum(y[2:-1:2])
    integral = (h / 3.0) * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]))
    return (1.0 + z) * integral

def _quad_dL(z, nodes, weights, w0, wa):
    """
    Approximate dL using the 8-node quadrature rule.
    """
    if z <= 1e-12:
        return 0.0
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _objective(params):
    """
    Objective function to minimize: worst relative error.
    params: 8 node params (t), 8 weight params (v).
    """
    if _time_left() < 1.0:
        return 1e10
        
    # Decode nodes
    x = np.zeros(8)
    prev = 0.0
    for i in range(8):
        s = 1.0 / (1.0 + np.exp(-params[i]))
        remaining = 1.0 - prev
        step = s * remaining
        prev += step
        x[i] = prev
    x = np.clip(x, 1e-12, 1.0 - 1e-12)
    
    # Decode weights
    w_raw = np.exp(params[8:16])
    w = w_raw / np.sum(w_raw)
    
    # Test set: Broad coverage of w0, wa and z
    test_cosmos = [
        (0.0, 0.0),    # LCDM
        (-1.0, 0.0),   # Lambda
        (0.5, -1.0),   # Quintessence
        (-1.5, 0.5),   # Phantom
        (-0.5, 0.5),   
        (0.3, -0.3),   
        (-2.0, 1.0),   # More extreme phantom
        (1.0, -2.0)    # More extreme quintessence
        (-1.0, -1.0),
        (0.0, 1.0),
        (-0.5, -0.5)
    ]
    
    # Redshifts: Log-spaced to cover wide range
    test_zs = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0])
    
    max_err = 0.0
    for w0, wa in test_cosmos:
        for z in test_zs:
            true_val = _true_dL_fast(z, w0, wa)
            quad_val = _quad_dL(z, x, w, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                if rel_err > max_err:
                    max_err = rel_err
                    
    return max_err

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Multi-start local optimization
    # We will try several different initial configurations for nodes and weights.
    
    best_score = 1e10
    best_result = None
    
    # Initial guesses for nodes (in t-space, sigmoid mapped)
    # We want to cover different node distributions.
    
    def make_params_from_nodes_weights(nodes, weights):
        """
        Convert desired nodes (0,1) and weights (>0) to t and v params.
        """
        t = np.zeros(8)
        prev = 0.0
        for i in range(8):
            # x_i = prev + s_i * (1 - prev)
            # s_i = (x_i - prev) / (1 - prev)
            if prev < 1.0:
                s_i = (nodes[i] - prev) / (1.0 - prev)
                s_i = np.clip(s_i, 1e-6, 1.0 - 1e-6)
                t[i] = np.log(s_i / (1.0 - s_i))
            else:
                t[i] = 0.0
            prev = nodes[i]
        
        v = np.log(weights + 1e-12)
        return np.concatenate([t, v])

    # 1. Gauss-Legendre
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = 0.5 * (x_gl + 1.0)
    weights_gl = 0.5 * w_gl
    weights_gl = weights_gl / np.sum(weights_gl)
    
    # 2. Uniform
    nodes_uni = np.linspace(1/16, 15/16, 8)
    weights_uni = np.ones(8) / 8.0
    
    # 3. Chebyshev
    # Chebyshev nodes in [0,1]
    k = np.arange(8)
    nodes_cheb = 0.5 * (1 - np.cos(np.pi * (k + 0.5) / 8))
    nodes_cheb = np.sort(nodes_cheb)
    weights_cheb = np.ones(8) / 8.0 # Simple equal weights for start
    
    # 4. Log-spaced nodes (more density at low z/high a? or high z/low a?)
    # Luminosity distance integral is sensitive to low a (high z) for some models?
    # Actually, the integrand 1/E(a) varies.
    # Let's try nodes clustered near 0 (high z) and near 1 (low z).
    # Nodes in z-space: 0 < n < 1.
    # If we want to capture high z behavior, we want small nodes?
    # a = 1/(1+zn). Small n -> a close to 1. Large n -> a small.
    # So nodes near 1 probe high z. Nodes near 0 probe low z.
    # Uniform is balanced.
    
    starts = [
        make_params_from_nodes_weights(nodes_gl, weights_gl),
        make_params_from_nodes_weights(nodes_uni, weights_uni),
        make_params_from_nodes_weights(nodes_cheb, weights_cheb),
    ]
    
    # Add some random perturbations to GL
    rng = np.random.default_rng(42)
    for i in range(3):
        p_gl = make_params_from_nodes_weights(nodes_gl, weights_gl)
        p_pert = p_gl + rng.normal(0, 0.5, 16)
        starts.append(p_pert)
    
    # Run local optimization for each start
    for i, start in enumerate(starts):
        if _time_left() < 10:
            break
            
        try:
            # Nelder-Mead is good for non-smooth, bound-constrained (via penalty or careful init)
            # Our parameterization ensures bounds.
            res = minimize(
                _objective, 
                start, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception as e:
            pass
    
    # If no result or still very high, fallback to GL
    if best_result is None or best_score > 0.1:
        p_gl = make_params_from_nodes_weights(nodes_gl, weights_gl)
        best_result = p_gl
        best_score = _objective(p_gl)
    
    # Final refinement with L-BFGS-B? 
    # L-BFGS-B might struggle with non-smooth max error.
    # Let's do one more Nelder-Mead polish if time permits.
    if _time_left() > 5:
        try:
            res2 = minimize(
                _objective, 
                best_result, 
                method='Nelder-Mead', 
                options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_result = res2.x
        except Exception as e:
            pass
            
    # Decode final result
    t = best_result[:8]
    v = best_result[8:16]
    
    x = np.zeros(8)
    prev = 0.0
    for i in range(8):
        s = 1.0 / (1.0 + np.exp(-t[i]))
        remaining = 1.0 - prev
        step = s * remaining
        prev += step
        x[i] = prev
    x = np.clip(x, 1e-12, 1.0 - 1e-12)
    
    w_raw = np.exp(v)
    w = w_raw / np.sum(w_raw)
    
    # Final validation check
    w = w / np.sum(w)
    
    return {"nodes": x, "weights": w}

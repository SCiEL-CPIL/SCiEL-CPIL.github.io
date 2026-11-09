"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def _E_cpl(a, w0, wa, om_m, om_r, om_k=0.0):
    """E(a) = H(a)/H0 for CPL model."""
    # Ensure numerical stability
    a_safe = np.maximum(a, 1e-10)
    # Spatially flat: om_k = 0, but we keep it for generality in case of non-flat validation
    # However, task says "spatially flat", so om_k = 0 usually.
    # om_de = 1 - om_m - om_r
    om_de = 1.0 - om_m - om_r
    # E^2 = om_m/a^3 + om_r/a^4 + om_de * a^(-3(1+w0+wa))
    # Note: If w_a is time-dependent, the standard CPL parameterization is 
    # w(a) = w0 + wa*(1-a). 
    # The term is a^(-3(1+w0)) * (1-a)^(-3wa).
    # Let's use the standard CPL form:
    # H^2/H0^2 = Om_m a^-3 + Om_r a^-4 + Om_de * a^(-3(1+w0)) * (1-a)^(-3wa)
    
    if a_safe >= 1.0:
        # Limit at a=1
        # (1-a)^(-3wa) -> infinity if wa < 0? No, (1-a) -> 0.
        # Usually wa is small. If wa != 0, we might have issues at a=1.
        # But we integrate from a=1 (z=0) to a=1/(1+z).
        # The integral is over x in [0,1], a = 1/(1+zx).
        # At x=0, a=1.
        # If wa != 0, the term (1-a)^(-3wa) is singular at a=1 if wa < 0?
        # (1-a) is small positive. 
        # Standard CPL: w(a) = w0 + wa * (1-a).
        # Integral of 3(1+w(a)) da = 3w0 a + (3wa/2)(1-a)^2? No.
        # Let's stick to the numerical evaluation of E(a).
        pass

    h2 = om_m * a_safe**(-3) + om_r * a_safe**(-4)
    if om_de != 0:
        # CPL term
        # To avoid 0^0 or inf, handle a close to 1 carefully
        # If a is very close to 1, (1-a) is small.
        # We use the formula: a^(-3(1+w0)) * (1-a)^(-3wa)
        # If wa=0, it's just a^(-3(1+w0))
        if abs(wa) < 1e-10:
            de_term = om_de * a_safe**(-3.0 * (1.0 + w0))
        else:
            # Clip a to [1e-10, 1-1e-10] for the (1-a) part to avoid log(0) or pow issues if needed
            # But here we just do power.
            a_clip = np.clip(a_safe, 1e-10, 1.0 - 1e-10)
            de_term = om_de * (a_clip**(-3.0 * (1.0 + w0))) * ((1.0 - a_clip)**(-3.0 * wa))
        h2 += de_term
    
    return np.sqrt(np.maximum(h2, 1e-30))

def _integrand_val(x, z, w0, wa, om_m, om_r, om_k=0.0):
    """The integrand 1/E(a) where a = 1/(1+z*x)."""
    if z == 0:
        return 1.0
    a = 1.0 / (1.0 + z * x)
    return 1.0 / _E_cpl(a, w0, wa, om_m, om_r, om_k)

def _true_dL_integral(z, w0, wa, om_m, om_r, om_k=0.0, n=2000):
    """Compute the true integral of 1/E(a) dx using high-resolution Simpson's rule."""
    if z == 0:
        return 1.0
    x = np.linspace(0, 1, n)
    # Simpson's rule requires odd number of points
    if n % 2 == 0:
        n += 1
        x = np.linspace(0, 1, n)
    
    y = _integrand_val(x, z, w0, wa, om_m, om_r, om_k)
    
    h = 1.0 / (n - 1)
    integral = h / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]))
    return integral

def _decode_params(params):
    """
    Decode cumulative gap parameters to nodes and weights.
    params[0:8] are gaps g_i > 0, sum(g_i) = 1.
    nodes = cumsum(g_i)
    params[8:16] are log-weights, or just weights? 
    Let's use direct weights with positivity constraint via exp.
    """
    gaps = np.exp(params[:8])
    gaps = gaps / np.sum(gaps)
    nodes = np.cumsum(gaps)
    # Ensure nodes are strictly inside (0,1)
    # cumsum of positive gaps summing to 1 ends at 1. 
    # We need nodes in (0,1). 
    # If we use 8 nodes, we need 8 gaps? No, 8 nodes define 8 points.
    # Usually, we define 8 points. 
    # Let's use 8 nodes. 
    # Gaps: g_1, ..., g_8. Sum = 1. Nodes: n_i = sum_{j=1}^i g_j.
    # n_8 = 1. This is bad, we need n_8 < 1.
    # Let's use 9 gaps? No, 8 nodes.
    # Let's parameterize 8 nodes as cumulative sums of 8 positive numbers that sum to 1?
    # Then last node is 1.0. 
    # We need nodes in [0,1]. The task says "strictly increasing in [0,1]".
    # Usually quadrature nodes are in (0,1). 
    # Let's force the last node to be < 1 by scaling?
    # Better: Use 8 gaps g_1...g_8, sum g_i = 1. 
    # Nodes = cumsum(g). Last node is 1.
    # We can just evaluate at 1.0. E(1) is well defined.
    # However, "strictly increasing in [0,1]" usually implies open interval for GL, 
    # but closed is fine if defined.
    # Let's assume nodes in (0,1) is safer to avoid boundary issues if any.
    # Let's use 9 gaps? No.
    # Let's use the fact that we can just optimize nodes directly with bounds (0, 1) 
    # and rely on DE to keep them sorted? DE doesn't enforce ordering.
    # The cumulative sum method enforces ordering.
    # To keep last node < 1, we can normalize gaps to sum to 0.99?
    # Or just accept node=1.0.
    
    # Let's try: 8 gaps, sum to 1. Nodes = cumsum.
    # Weights: 8 positive numbers.
    log_weights = params[8:16]
    weights = np.exp(log_weights)
    weights = weights / np.sum(weights)
    
    return nodes, weights

def _loss(params):
    """Max relative error across a diverse set of training models."""
    nodes, weights = _decode_params(params)
    
    # Check validity
    if np.any(nodes <= 0) or np.any(nodes >= 1.0):
        # If node is exactly 1.0, it's okay for integration, but let's penalize if we want strict (0,1)
        # Task: "strictly increasing in [0,1]". 1.0 is in [0,1].
        pass
    if np.any(weights <= 0):
        return 1e10

    # Diverse Training Models
    # We want to cover the space of (w0, wa, om_m) likely to be in validation.
    # Validation is "disjoint public validation cosmologies".
    # We should sample a wide grid.
    
    # Generate a set of models
    models = []
    # Lambda CDM
    models.append((0.0, 0.0, 0.3, 0.05))
    # Quintessence
    models.append((-0.5, 0.5, 0.3, 0.05))
    models.append((-0.7, 0.3, 0.3, 0.05))
    # Phantom
    models.append((-1.2, 0.5, 0.3, 0.05))
    models.append((-1.5, 0.0, 0.3, 0.05))
    # wCDM (wa=0)
    models.append((-0.9, 0.0, 0.3, 0.05))
    models.append((-1.1, 0.0, 0.3, 0.05))
    # Different Om_m
    models.append((0.0, 0.0, 0.2, 0.05))
    models.append((0.0, 0.0, 0.4, 0.05))
    # Extreme wa
    models.append((-0.5, 1.0, 0.3, 0.05))
    models.append((-1.0, -0.5, 0.3, 0.05))
    
    # Redshifts
    z_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    max_err = 0.0
    for w0, wa, om_m, om_r in models:
        for z in z_vals:
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = _E_cpl(a_vals, w0, wa, om_m, om_r, 0.0)
            quad_val = np.sum(weights / E_vals)
            
            true_val = _true_dL_integral(z, w0, wa, om_m, om_r, 0.0, n=1000)
            
            if true_val < 1e-10:
                err = abs(quad_val - true_val)
            else:
                err = abs(quad_val - true_val) / true_val
            
            if err > max_err:
                max_err = err
                
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    # Initial guess: Gauss-Legendre nodes and weights
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    x0_gl = (x_gl + 1.0) / 2.0
    w0_gl = w_gl / 2.0
    
    # Encode initial guess
    # Nodes -> Gaps
    gaps = np.diff(np.concatenate([[0], x0_gl, [1]]))
    # Ensure gaps are positive
    gaps = np.maximum(gaps, 1e-6)
    gaps = gaps / np.sum(gaps)
    # Log weights
    log_w = np.log(np.maximum(w0_gl, 1e-10))
    
    x0 = np.concatenate([gaps, log_w])
    
    # Bounds for DE
    # Gaps: positive, will be normalized. So bound [1e-6, 10] is fine.
    # Log weights: [-20, 10]
    bounds = [(1e-6, 10.0) for _ in range(8)] + [(-20.0, 10.0) for _ in range(8)]
    
    best_params = x0
    best_loss = _loss(x0)
    
    # Run Differential Evolution
    try:
        # DE is stochastic, set seed for reproducibility if needed, but we want exploration
        res = differential_evolution(
            _loss, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-8,
            popsize=15,
            polish=False # Disable polish to avoid getting stuck in local min of L-BFGS later? 
                         # Actually, polish=True uses L-BFGS-B which might fail on non-differentiable max_err.
                         # Let's do our own polish.
        )
        
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    # Check time
    if time.time() > deadline - 10:
        nodes, weights = _decode_params(best_params)
        # Ensure weights sum to 1
        weights = weights / np.sum(weights)
        # Ensure nodes are strictly increasing
        if not np.all(np.diff(nodes) > 0):
             # Fallback
             x_gl, w_gl = np.polynomial.legendre.leggauss(8)
             nodes = (x_gl + 1.0) / 2.0
             weights = w_gl / 2.0
        return {"nodes": nodes, "weights": weights}
        
    # Polish with L-BFGS-B on the decoded parameters?
    # DE outputs gaps and log_weights.
    # L-BFGS-B works on unconstrained or bounded variables.
    # We can polish the gaps and log_weights.
    
    def _loss_for_polish(p):
        return _loss(p)
        
    try:
        res2 = minimize(
            _loss_for_polish, 
            best_params, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        if res2.fun < best_loss:
            best_loss = res2.fun
            best_params = res2.x
    except Exception as e:
        pass
        
    nodes, weights = _decode_params(best_params)
    
    # Final sanity checks
    if not np.all(np.diff(nodes) > 0):
        # Fallback to GL
        x_gl, w_gl = np.polynomial.legendre.leggauss(8)
        nodes = (x_gl + 1.0) / 2.0
        weights = w_gl / 2.0
    else:
        weights = weights / np.sum(weights)
        
    # Ensure nodes are in [0,1]
    nodes = np.clip(nodes, 0.0, 1.0)
    
    return {"nodes": nodes, "weights": weights}

# EVOLVE-BLOCK-END

"""Cosmology: universal luminosity-distance quadrature.
SEARCH approach: Minimax optimization over a dense, physics-based model library.
Instead of sparse sampling, we use a fine grid of CPL parameters and 
optimize nodes/weights to minimize the worst-case error across this broader 
space, aiming for a more robust universal rule.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

START = time.time()
DEADLINE = 85.0
MARGIN = 5.0
EPS = 1e-8
RNG_SEED = 42

def _fallback():
    n = 8
    return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}

def _model_E(a, w0, wa):
    """
    H(a)/H0 for flat LambdaCDM-like CPL models.
    E(a)^2 = Om_m a^-3 + Om_l a^-3(1+w0)exp(-3wa(1-a))
    Here we assume Om_m ~ 0.3, Om_l ~ 0.7 for realistic smoothness, 
    but the task says 'time-varying dark-energy models'. 
    To be safe and cover 'broad' space, we include variations of Om_m and Om_l.
    """
    # Standard flat universe: Om_m + Om_l = 1
    # E^2 = Om_m * a^-3 + Om_l * a^-3 * exp(-3 * wa * (1-a) * (1+w0) / (1+w0)?) 
    # CPL: w(a) = w0 + wa(1-a)
    # rho_l(a) = rho_l0 * exp(-3 * integral_0^a (1+w(a'))/a' da')
    # integral (1+w0+wa(1-a))/a da = (1+w0)ln(a) + wa(1-a)
    # So rho_l(a) ~ a^-3(1+w0) * exp(wa(1-a))
    # E^2 = Om_m a^-3 + Om_l a^-3(1+w0) exp(wa(1-a))
    
    # To handle a->0 correctly, we need limits.
    # But for quadrature, a is in [1/(1+z_max), 1]. z_max is usually not 0.
    # The integrand is 1/E(a).
    
    # We will use a slightly generalized form to cover more "time-varying" behaviors.
    # Let's stick to standard CPL with varying Om_m and w0, wa.
    pass

def _E_cpl(a, Om_m, w0, wa):
    """
    Compute E(a) for CPL model.
    Om_l = 1 - Om_m.
    E^2 = Om_m a^-3 + (1-Om_m) a^-3(1+w0) exp(wa(1-a))
    """
    Om_l = 1.0 - Om_m
    if Om_l < 0:
        Om_l = 0.0
    # Handle a=0 case if it arises (z->inf), but nodes are in [0,1] so a>=1/(1+z).
    # If z is large, a is small.
    # To avoid numerical issues with a^-3 at a=0, we assume z < 1000, so a > 0.001.
    a_safe = np.maximum(a, 1e-10)
    term_m = Om_m * a_safe**-3
    # CPL evolution of dark energy density
    # rho_l(a) = rho_l0 * a^-3 * (1+w0) * exp(wa*(1-a))
    term_l = Om_l * a_safe**-3 * (1+w0) * np.exp(wa * (1 - a_safe))
    E_sq = term_m + term_l
    # Clamp to positive
    E_sq = np.maximum(E_sq, 1e-10)
    return np.sqrt(E_sq)

def _relerr(nodes, weights, z_list, model_params):
    errs = []
    for z in z_list:
        a = 1.0 / (1.0 + z * nodes)
        # Evaluate model
        E_vals = np.array([_E_cpl(ai, *p) for ai, p in zip(a, model_params)])
        quad = z * np.sum(weights / E_vals)
        
        # Reference: Adaptive integration using scipy.integrate.quad or high-order Gauss
        # For speed and consistency, use Gauss-Legendre with 100 points
        # Integrand: f(u) = 1/E(1/(1+zu)) * z ... wait.
        # dL/c = (1+z) * integral_0^z dz' / H(z')
        # Change var to a: dz = -da/a^2. 
        # Integral from a=1 to a=1/(1+z) of (1/a^2) / E(a) da
        # Let u be node in [0,1]. a = 1/(1+z*u). 
        # The formula given: dL/(c/H0) = (1+z)*z*sum weights[i]/E(a_i)
        # This implies the integral is approximated as z * sum w_i / E(a_i) ?
        # Let's check the change of variables.
        # I = integral_0^z dz' / H(z')
        # a = 1/(1+z'). da = -1/(1+z')^2 dz' = -a^2 dz'. dz' = -da/a^2.
        # Limits: z'=0 -> a=1. z'=z -> a=1/(1+z).
        # I = integral_{1/(1+z)}^1 (1/a^2) * (1/E(a)) da
        # Let u = a * z? No.
        # The task says: dL/(c/H0) = (1+z)*z*sum weights[i]/E(a_i)
        # This looks like a specific quadrature rule for the integral J = integral_0^1 du / (u * E(1/(1+zu)))? No.
        
        # Let's re-read carefully: "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i])"
        # This implies the integral being approximated is:
        # I_approx = z * sum w_i / E(a_i)
        # And dL = (1+z) * I_approx * c/H0?
        
        # Let's verify with a=1 (z=0). Limit z->0.
        # a_i = 1/(1+z n_i) ~ 1 - z n_i.
        # E(a_i) ~ E(1) = 1 (normalized).
        # I_approx ~ z * sum w_i = z.
        # dL ~ (1+z) * z ~ z.
        # Actual dL for small z: H0 * z / H0 = z. (1+z) factor makes it z+z^2.
        # Standard dL = c/H0 (1+z) integral_0^z dz'/H(z').
        # For small z, H(z') ~ H0. Integral ~ z. dL ~ c/H0 (1+z) z.
        # So the formula matches the definition.
        
        # Reference integral:
        # R = integral_{1/(1+z)}^1 da / (a^2 E(a))
        # Let's use Gauss-Legendre on [0,1] for u, where a = 1/(1+z u) is NOT linear.
        # Better: Integrate over a directly?
        # Or use the u variable from the task's perspective.
        # The task defines the quadrature on nodes in [0,1].
        # The integral is effectively approximating:
        # F(z) = integral_0^1 du / ( (1+zu) * E(1/(1+zu)) * (1+z) ) ? No.
        
        # Let's just compute the reference integral numerically using high precision.
        # I_ref = integral_{a_min}^1 da / (a^2 E(a))
        # Use scipy.integrate.quad for reference as it is robust.
        from scipy.integrate import quad
        def integrand_a(a_val):
            E_val = _E_cpl(a_val, *model_params[0]) # Assuming same params for all a? No, params are per model.
            # Wait, model_params is a list of tuples for different models? 
            # In _relerr, we loop over z. The model is fixed for this call.
            # So model_params is a single tuple (Om_m, w0, wa).
            return 1.0 / (a_val**2 * _E_cpl(a_val, *model_params))
        
        a_min = 1.0 / (1.0 + z)
        # quad is slow. Let's use Gauss-Legendre with 100 points on [a_min, 1]
        # Map xi in [-1,1] to a in [a_min, 1]
        g = np.polynomial.legendre.leggauss(100)
        xi, wi = g
        a_ref = 0.5 * (1 - a_min) * xi + 0.5 * (1 + a_min)
        E_ref = _E_cpl(a_ref, *model_params)
        I_ref = 0.5 * (1 - a_min) * np.sum(wi / (a_ref**2 * E_ref))
        
        # The task formula: dL = (1+z) * z * sum w_i / E(a_i)
        # My reference I_ref is the integral part.
        # So dL_ref = (1+z) * I_ref
        # dL_quad = (1+z) * z * np.sum(weights / E_vals)
        
        # Relative error:
        err = abs(dL_quad - dL_ref) / dL_ref
        errs.append(err)
    return np.max(errs)

def _build_library():
    """
    Create a dense library of cosmologies.
    Om_m in [0.2, 0.4]
    w0 in [-1.5, -0.5]
    wa in [-1.0, 1.0]
    """
    lib = []
    zs = [0.1, 0.3, 0.5, 0.8, 1.2, 1.5, 2.0, 3.0]
    
    # Grid of parameters
    Om_m_vals = [0.2, 0.3, 0.4]
    w0_vals = [-1.5, -1.0, -0.5]
    wa_vals = [-1.0, 0.0, 1.0]
    
    for Om_m in Om_m_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                lib.append((zs, (Om_m, w0, wa)))
                
    # Add some extreme cases
    lib.append((zs, (0.1, -2.0, 2.0)))
    lib.append((zs, (0.5, 0.0, 0.0))) # Matter dominated (w=-1, wa=0 is Lambda, w=0 is Matter? No, w=0 is dust if it's the only component? 
                                       # If Om_m=0.5, Om_l=0.5, w=0, it's a mix of dust and dust? No, w=0 is pressureless.
                                       # So this is just matter dominated with different normalization?
                                       # Actually if w=0 and wa=0, it's matter. E^2 = Om_m a^-3 + Om_l a^-3 = a^-3. E = a^-1.5.
                                       # This is valid.
    return lib

def _objective(xv, library):
    """
    Objective function for minimization.
    xv: 16 floats. First 8 are nodes, last 8 are log-weights.
    """
    nodes = xv[:8]
    # Ensure nodes are strictly increasing in [0,1]
    # Apply sigmoid or tanh mapping? Or just penalize violations.
    # Better: map from unconstrained space to [0,1] using cumulative sum of exponentials?
    # Simpler: use raw values but penalize heavily if out of bounds or not increasing.
    
    # Penalty for bounds
    pen = 0.0
    if np.any(nodes < 0) or np.any(nodes > 1):
        pen += 1e6 * (np.sum(np.minimum(nodes, 0)**2) + np.sum(np.maximum(nodes-1, 0)**2))
    
    diffs = np.diff(nodes)
    if np.any(diffs <= 1e-6):
        pen += 1e6 * np.sum(np.maximum(1e-6 - diffs, 0)**2)
    
    if pen > 1e-3:
        return pen
    
    weights = np.exp(xv[8:16])
    weights = weights / np.sum(weights)
    
    max_err = 0.0
    for zs, params in library:
        err = _relerr(nodes, weights, zs, params)
        if err > max_err:
            max_err = err
            
    return max_err

def construct():
    fallback = _fallback()
    
    # 1. Global optimization with Differential Evolution
    # Bounds: nodes in [0.01, 0.99], log-weights in [-5, 5]
    bounds = [(0.01, 0.99)] * 8 + [(-5, 5)] * 8
    
    library = _build_library()
    
    best_x = None
    best_err = float('inf')
    
    # Run DE for a short time
    if time.time() < DEADLINE - 30:
        try:
            res = differential_evolution(
                _objective, 
                bounds, 
                args=(library,),
                seed=RNG_SEED,
                maxiter=50,
                tol=1e-6,
                popsize=10,
                update_interval=10,
                polish=False # We will polish later
            )
            if res.success and res.fun < best_err:
                best_err = res.fun
                best_x = res.x
        except Exception as e:
            pass
            
    # 2. Local refinement with Nelder-Mead or BFGS (if gradient available, but we use NM)
    if best_x is not None:
        # Start from DE result
        x0 = best_x
    else:
        # Fallback to random or grid
        x0 = np.random.uniform(0.1, 0.9, 8)
        x0[0] = 0.1
        x0[-1] = 0.9
        # Sort and make unique
        x0 = np.sort(x0)
        # Add small noise to ensure uniqueness
        x0 += np.linspace(0, 0.1, 8)
        x0 = x0 / x0[-1] * 0.9 + 0.05
        x0 = np.sort(x0)
        x0 = np.concatenate([x0, np.zeros(8)])
        
    # Refine
    if time.time() < DEADLINE - 5:
        try:
            res = minimize(
                _objective, 
                x0, 
                args=(library,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x
        except Exception:
            pass

    if best_x is not None and best_err < 1.0: # Only accept if better than 100% error? No, 1.0 is score.
        nodes = best_x[:8]
        weights = np.exp(best_x[8:16])
        weights = weights / np.sum(weights)
        
        # Final validity check
        if np.all(np.diff(nodes) > 1e-6) and np.all(nodes > 0) and np.all(nodes < 1) and np.all(weights > 0):
            return {"nodes": nodes, "weights": weights}
            
    return fallback

# EVOLVE-BLOCK-END

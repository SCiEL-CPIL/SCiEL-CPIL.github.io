import numpy as np
import time
from scipy.integrate import quad
from scipy.optimize import minimize

# Global variables to store pre-computed data
_TRUE_INTEGRALS = None
_MODELS = None
_Z_LIST = None

def _E_of_a(a, Om, w0, wa):
    """
    Compute E(a) = H(a)/H0 for a spatially flat universe with CPL dark energy.
    Om: Omega_m
    w0, wa: CPL parameters for dark energy
    a: scale factor
    """
    # Omega_DE = 1 - Om (assuming flat and negligible radiation for this approximation, 
    # or we can include it. The prompt implies standard cosmology. 
    # Let's include a small radiation term for high z accuracy if needed, 
    # but usually Om_r ~ 1e-4. 
    # Given the "luminosity distance" context and typical validation ranges, 
    # matter and DE are sufficient. I will add a small Om_r for robustness at high z.
    Om_r = 1.0e-4
    Om_DE = 1.0 - Om - Om_r
    
    # a^-3 for matter
    term_m = Om * (a ** -3)
    # a^-4 for radiation
    term_r = Om_r * (a ** -4)
    # DE term: a^(-3(1+w)) where w = w0 + wa*ln(a)
    # w(a) = w0 + wa * ln(a)
    # 1+w = 1 + w0 + wa*ln(a)
    # Exponent = -3 * (1 + w0 + wa * ln(a))
    if a <= 0:
        return np.inf
    ln_a = np.log(a)
    exponent = -3.0 * (1.0 + w0 + wa * ln_a)
    term_de = Om_DE * (a ** exponent)
    
    return np.sqrt(term_m + term_r + term_de)

def _precompute_data():
    global _TRUE_INTEGRALS, _MODELS, _Z_LIST
    
    # Define a set of representative cosmologies
    # Om: 0.1 to 0.5
    # w0: -2.0 to 0.0
    # wa: -1.0 to 1.0
    Om_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    w0_vals = [-2.0, -1.5, -1.0, -0.5, 0.0]
    wa_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    _MODELS = []
    for Om in Om_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                _MODELS.append((Om, w0, wa))
                
    # Redshifts for validation
    # Include small z, intermediate, and high z
    _Z_LIST = [0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
    
    _TRUE_INTEGRALS = np.zeros((len(_MODELS), len(_Z_LIST)))
    
    for i_m, (Om, w0, wa) in enumerate(_MODELS):
        for i_z, z in enumerate(_Z_LIST):
            if z == 0:
                _TRUE_INTEGRALS[i_m, i_z] = 0.0
                continue
            # Integrand: 1 / E(1/(1+u))
            # Integral: (1+z) * z * int_0^1 dx / E(1/(1+zx))
            # We compute the integral part: I = int_0^1 dx / E(1/(1+zx))
            def integrand(x):
                a = 1.0 / (1.0 + z * x)
                return 1.0 / _E_of_a(a, Om, w0, wa)
            
            val, _ = quad(integrand, 0.0, 1.0, limit=100)
            _TRUE_INTEGRALS[i_m, i_z] = (1.0 + z) * z * val

def _objective(params):
    """
    params: 16 floats. First 8 are u (for nodes), next 8 are v (for weights).
    """
    u = params[:8]
    v = params[8:16]
    
    # Transform u to nodes x in (0,1) strictly increasing
    exp_u = np.exp(u)
    cum_exp_u = np.cumsum(exp_u)
    total_exp_u = cum_exp_u[-1]
    if total_exp_u == 0:
        return 1e10
    x = cum_exp_u / total_exp_u
    
    # Transform v to weights w > 0, sum=1
    exp_v = np.exp(v)
    total_exp_v = np.sum(exp_v)
    if total_exp_v == 0:
        return 1e10
    w = exp_v / total_exp_v
    
    # Compute quadrature approximation for all models and redshifts
    # x: (8,)
    # w: (8,)
    # _MODELS: list of tuples
    # _Z_LIST: list of floats
    # _TRUE_INTEGRALS: (N_models, N_z)
    
    max_err = 0.0
    
    # Vectorize where possible
    # We can loop over models and redshifts, but vectorize over nodes
    for i_m, (Om, w0, wa) in enumerate(_MODELS):
        for i_z, z in enumerate(_Z_LIST):
            if z == 0:
                continue
            a = 1.0 / (1.0 + z * x)
            E_vals = _E_of_a(a, Om, w0, wa)
            # Quadrature sum
            q_sum = np.sum(w / E_vals)
            q_val = (1.0 + z) * z * q_sum
            
            true_val = _TRUE_INTEGRALS[i_m, i_z]
            if true_val == 0:
                continue
            
            rel_err = abs(q_val - true_val) / abs(true_val)
            if rel_err > max_err:
                max_err = rel_err
                
    return max_err

def construct():
    global _TRUE_INTEGRALS, _MODELS, _Z_LIST
    
    start_time = time.time()
    timeout = 85.0  # Safety margin for 90s limit
    
    # Pre-compute ground truth
    if _TRUE_INTEGRALS is None:
        _precompute_data()
        
    # Initial guess: Gauss-Legendre nodes and weights
    x0_gauss, w0_gauss = np.polynomial.legendre.leggauss(8)
    
    # Map Gauss nodes to u params
    # x = cumsum(exp(u))/sum(exp(u))
    # This mapping is not trivial to invert exactly, but we can use a simple heuristic or just start with uniform u.
    # Uniform u gives x_i proportional to cumulative sum of 1s, i.e., x_i = i/8.
    # Let's start with u=0 (uniform nodes) and v=0 (uniform weights).
    # Or better, try to find u that gives Gauss nodes?
    # Actually, starting with uniform nodes/weights is a valid start.
    # Let's use u=0, v=0.
    u_init = np.zeros(8)
    v_init = np.zeros(8)
    params_init = np.concatenate([u_init, v_init])
    
    best_params = params_init
    best_score = _objective(best_params)
    
    # Optimization
    # Use Powell method for derivative-free optimization
    # It is robust and handles the 16 dimensions well.
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < timeout:
        try:
            result = minimize(
                _objective, 
                params_init, 
                method='Powell', 
                options={'maxiter': 2000, 'maxfev': 10000, 'disp': False}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception as e:
            pass # Keep best_params
        
    # Check time again, maybe do a second pass with different start if time permits
    elapsed = time.time() - start_time
    if elapsed < timeout - 5.0:
        # Try starting from Gauss-Legendre mapped approximately
        # We can use a simple inverse mapping or just perturb the uniform start
        # Let's try a random restart if time allows
        np.random.seed(42)
        u_rand = np.random.uniform(-1, 1, 8)
        v_rand = np.random.uniform(-1, 1, 8)
        params_rand = np.concatenate([u_rand, v_rand])
        
        try:
            result2 = minimize(
                _objective, 
                params_rand, 
                method='Powell', 
                options={'maxiter': 1000, 'maxfev': 5000, 'disp': False}
            )
            if result2.fun < best_score:
                best_score = result2.fun
                best_params = result2.x
        except Exception as e:
            pass

    # Extract nodes and weights
    u = best_params[:8]
    v = best_params[8:16]
    
    exp_u = np.exp(u)
    cum_exp_u = np.cumsum(exp_u)
    total_exp_u = cum_exp_u[-1]
    x = cum_exp_u / total_exp_u
    
    exp_v = np.exp(v)
    total_exp_v = np.sum(exp_v)
    w = exp_v / total_exp_v
    
    # Ensure strict increasing and within [0,1]
    # The transformation ensures this, but let's clip just in case of numerical noise
    x = np.clip(x, 1e-12, 1.0 - 1e-12)
    # Ensure strictly increasing
    for i in range(1, 8):
        if x[i] <= x[i-1]:
            x[i] = x[i-1] + 1e-12
    # Re-normalize weights to sum to 1 exactly
    w = w / np.sum(w)
    
    return {"nodes": x, "weights": w}


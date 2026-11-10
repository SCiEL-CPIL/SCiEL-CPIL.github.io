"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import roots_legendre


def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    Strategy:
    1. Use a parameterized representation: nodes via cumulative gaps, weights via softmax.
    2. Multi-start Nelder-Mead optimization directly on worst-case relative error.
    3. Wider cosmology grid for better coverage of validation set.
    4. Final coordinate-descent polish on worst-case error.
    """
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill
    
    # 1. Gauss-Legendre nodes as initial guess
    gl_nodes, gl_weights = roots_legendre(8)
    nodes0 = 0.5 * (gl_nodes + 1.0)
    weights0 = 0.5 * gl_weights
    weights0 = weights0 / np.sum(weights0)
    
    # 2. Define representative CPL cosmologies - wider grid
    Om_m = 0.3
    Om_de = 0.7
    
    w0_values = np.linspace(-1.5, 0.8, 10)
    wa_values = np.linspace(-1.0, 1.5, 9)
    z_values = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
    
    def E_of_a(a, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL model."""
        a = np.asarray(a, dtype=float)
        matter = Om_m / a**3
        w0wa = w0 + wa
        if abs(w0wa) < 1e-12:
            de_term = Om_de * a**(-3.0*(1.0+w0))
        else:
            exponent = 3.0*wa/(1.0+w0wa)
            base = 1.0 + wa*(1.0 - a)
            # Handle potential negative base for some parameters
            de_term = Om_de * a**(-3.0*(1.0+w0wa)) * np.power(np.maximum(base, 1e-15), exponent)
        return np.sqrt(np.maximum(matter + de_term, 1e-30))
    
    def reference_integral(z, w0, wa, n_ref=2001):
        """High-accuracy reference integral using Simpson's rule."""
        u = np.linspace(0, 1, n_ref)
        a_vals = 1.0 / (1.0 + z * u)
        E_vals = E_of_a(a_vals, w0, wa)
        integrand = 1.0 / E_vals
        h = 1.0 / (n_ref - 1)
        integral = h * (integrand[0] + integrand[-1] + 
                        4.0*np.sum(integrand[1:-1:2]) + 
                        2.0*np.sum(integrand[2:-1:2]))
        return z * integral
    
    # 3. Pre-compute reference values
    ref_vals = []
    test_cases = []
    
    for w0 in w0_values:
        for wa in wa_values:
            for z in z_values:
                if time.time() > deadline:
                    break
                try:
                    rv = reference_integral(z, w0, wa)
                    if rv > 1e-12 and np.isfinite(rv):
                        ref_vals.append(rv)
                        test_cases.append((w0, wa, z))
                except Exception:
                    continue
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if not ref_vals:
        return {"nodes": nodes0, "weights": weights0}
    
    ref_vals = np.array(ref_vals)
    n_cases = len(ref_vals)
    
    # 4. Parameterized representation
    # nodes: use cumulative gaps. Let gaps be g[0..7] with g[0] = n[0], g[i] = n[i]-n[i-1]
    # All gaps must be positive, sum = n[7] < 1, and n[7] < 1
    # Parametrize: gaps = softplus(params[:8]), then scale so max node < 1
    # weights: softmax(params[8:16])
    
    def params_to_nodes_weights(params):
        """Convert unconstrained params to valid nodes and weights."""
        gap_params = params[:8]
        weight_params = params[8:16]
        
        # Gaps via softplus to ensure positivity
        gaps = np.logaddexp(0, gap_params)  # softplus
        # Scale gaps so that cumulative sum (last node) < 1
        total = np.sum(gaps)
        if total < 1e-10:
            gaps = gaps / total * 0.9
        else:
            gaps = gaps / total * 0.95  # ensure last node ~0.95 < 1
        
        nodes = np.cumsum(gaps)
        # Ensure strictly increasing and in (0,1)
        nodes = np.maximum(nodes, 1e-10)
        nodes = np.minimum(nodes, 1.0 - 1e-10)
        # Ensure strict increase
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-8
        
        # Weights via softmax
        wp = weight_params - np.max(weight_params)
        w = np.exp(wp)
        w = w / np.sum(w)
        
        return nodes, w
    
    def worst_case_error(nodes, weights):
        """Compute worst-case relative error across all test cases."""
        max_err = 0.0
        for i in range(n_cases):
            w0, wa, z = test_cases[i]
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = E_of_a(a_vals, w0, wa)
            integrand = 1.0 / E_vals
            quad_val = z * np.dot(weights, integrand)
            rel_err = abs(quad_val - ref_vals[i]) / ref_vals[i]
            if rel_err > max_err:
                max_err = rel_err
        return max_err
    
    def objective(params):
        """Worst-case relative error from unconstrained params."""
        if time.time() > deadline:
            return 1e10
        nodes, weights = params_to_nodes_weights(params)
        return worst_case_error(nodes, weights)
    
    # 5. Multi-start Nelder-Mead optimization
    from scipy.optimize import minimize
    
    best_params = None
    best_err = float('inf')
    
    # Initial params from Gauss-Legendre
    gl_gaps = np.diff(np.concatenate([[0], nodes0]))
    gl_params0 = np.concatenate([gl_gaps, np.log(weights0)])
    
    n_starts = 15
    rng = np.random.default_rng(42)
    
    for s in range(n_starts):
        if time.time() > deadline:
            break
        
        if s == 0:
            params_init = gl_params0.copy()
        else:
            # Random perturbation around GL or random init
            if s % 2 == 1:
                params_init = gl_params0 + rng.normal(0, 0.1, size=16)
            else:
                params_init = rng.uniform(-1, 1, size=16)
        
        try:
            result = minimize(objective, params_init, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                                    'adaptive': True})
            err = result.fun
            if err < best_err and np.isfinite(err):
                best_err = err
                best_params = result.x.copy()
        except Exception:
            continue
    
    # 6. Final coordinate-descent polish
    if best_params is not None:
        current_params = best_params.copy()
        current_err = best_err
        
        # Coordinate descent on each parameter
        for iteration in range(30):
            if time.time() > deadline:
                break
            improved = False
            for dim in range(16):
                # Try small perturbations
                for delta in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
                    trial = current_params.copy()
                    trial[dim] += delta
                    try:
                        trial_err = objective(trial)
                        if trial_err < current_err - 1e-12:
                            current_params = trial
                            current_err = trial_err
                            improved = True
                    except Exception:
                        pass
            if not improved:
                break
        
        if current_err < best_err:
            best_err = current_err
            best_params = current_params
    
    # 7. Extract final nodes and weights
    if best_params is not None:
        best_nodes, best_weights = params_to_nodes_weights(best_params)
    else:
        best_nodes = nodes0
        best_weights = weights0
    
    # Final validation
    if not (np.all(np.diff(best_nodes) > 1e-10) and np.all(best_nodes > 0) and np.all(best_nodes < 1)):
        best_nodes = nodes0
        best_weights = weights0
    
    if not np.all(best_weights > 0):
        best_weights = weights0
    
    # Normalize weights
    best_weights = best_weights / np.sum(best_weights)
    
    # Ensure sum is exactly 1 within tolerance
    if abs(np.sum(best_weights) - 1.0) > 1e-12:
        best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END

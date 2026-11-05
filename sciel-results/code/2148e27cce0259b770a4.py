"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import optimize
from scipy.integrate import quad


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s

    # Define cosmological E(a) functions
    def E_cpl(a, w0, wa):
        """CPL dark energy with Omega_m=0.3"""
        Om = 0.3
        Ol = 1.0 - Om
        if abs(w0 + 1.0) < 1e-10:
            if abs(wa) < 1e-10:
                return np.sqrt(Om / a**3 + Ol)
            else:
                return np.sqrt(Om / a**3 + Ol * a**(-3) * np.exp(3.0 * wa * (1.0 - a)))
        else:
            return np.sqrt(Om / a**3 + Ol * a**(-3.0*(1.0+w0+wa)) * np.exp(3.0*wa*(1.0-a)/(1.0+w0)))

    # Broad set of cosmologies to ensure universal performance
    # Include extreme values to cover the range of curvatures
    w0_vals = [-1.5, -1.2, -1.0, -0.8, -0.5, 0.0]
    wa_vals = [-0.5, 0.0, 0.5, 1.0, 1.5]
    
    test_models = []
    for w0 in w0_vals:
        for wa in wa_vals:
            # Skip some combinations if they are too similar or invalid
            if w0 == -1.0 and wa == 0.0:
                continue # Duplicate of LCDM essentially
            test_models.append(lambda a, w0=w0, wa=wa: E_cpl(a, w0, wa))
            
    # Add LCDM explicitly
    test_models.append(lambda a: E_cpl(a, -1.0, 0.0))
    
    # Representative redshifts
    test_zs = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

    # Pre-compute reference integrals
    ref_cache = {}
    def get_reference(z, E_func):
        key = (id(E_func), z)
        if key in ref_cache:
            return ref_cache[key]
        def integrand(x):
            a = 1.0 / (1.0 + z * x)
            return 1.0 / E_func(a)
        result, _ = quad(integrand, 0.0, 1.0, limit=500, epsabs=1e-14, epsrel=1e-14)
        ref_cache[key] = result
        return result

    # Compute reference values for all test cases
    ref_values = {}
    for i, E_func in enumerate(test_models):
        for z in test_zs:
            ref_values[(i, z)] = get_reference(z, E_func)

    def compute_worst_error(nodes_arr, weights_arr):
        """Compute maximum relative error across all test cases"""
        worst = 0.0
        for i, E_func in enumerate(test_models):
            for z in test_zs:
                a_vals = 1.0 / (1.0 + z * nodes_arr)
                quad_val = np.sum(weights_arr / E_func(a_vals))
                ref_val = ref_values[(i, z)]
                if abs(ref_val) > 1e-15:
                    err = abs(quad_val - ref_val) / abs(ref_val)
                    if err > worst:
                        worst = err
        return worst

    def soft_max_error(nodes_arr, weights_arr, beta=50.0):
        """Smooth approximation of worst-case error for optimization"""
        errors = []
        for i, E_func in enumerate(test_models):
            for z in test_zs:
                a_vals = 1.0 / (1.0 + z * nodes_arr)
                quad_val = np.sum(weights_arr / E_func(a_vals))
                ref_val = ref_values[(i, z)]
                if abs(ref_val) > 1e-15:
                    err = abs(quad_val - ref_val) / abs(ref_val)
                    errors.append(err)
        if not errors:
            return 0.0
        errors = np.array(errors)
        max_err = np.max(errors)
        if max_err < 1e-15:
            return 0.0
        # Log-sum-exp based soft max
        shifted = errors - max_err
        log_sum = max_err + np.log(np.sum(np.exp(beta * shifted / max_err))) / beta
        return log_sum

    # Start from Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    nodes0 = (gl_nodes + 1.0) / 2.0
    weights0 = gl_weights / 2.0
    nodes0 = np.sort(nodes0)
    order = np.argsort(gl_nodes)
    weights0 = gl_weights[order] / 2.0
    weights0 = weights0 / weights0.sum()

    best_nodes = nodes0.copy()
    best_weights = weights0.copy()
    best_err = compute_worst_error(best_nodes, best_weights)

    # Objective for optimization
    def objective(params):
        n_nodes = params[:8]
        n_weights = params[8:16]

        # Check node validity
        if np.any(n_nodes <= 0.001) or np.any(n_nodes >= 0.999):
            return 1e10
        if not np.all(np.diff(n_nodes) > 1e-4):
            return 1e10

        # Check weight validity
        if np.any(n_weights <= 0):
            return 1e10
        w_norm = n_weights / np.sum(n_weights)

        return soft_max_error(n_nodes, w_norm, beta=50.0)

    # Differential Evolution for global optimization
    # Bounds: nodes in [0.001, 0.999], weights in [1e-6, 1]
    bounds = [(0.001, 0.999) for _ in range(8)] + [(1e-6, 1.0) for _ in range(8)]
    
    # To handle the sorting constraint and positive weights, we can use a transformation
    # or rely on DE to stay in bounds and then post-process.
    # However, DE doesn't enforce ordering. 
    # We will use a penalty in the objective for disordered nodes.
    
    def objective_de(params):
        n_nodes = params[:8]
        n_weights = params[8:16]
        
        # Penalty for unordered nodes
        if not np.all(np.diff(n_nodes) > 1e-4):
            return 1e10
            
        if np.any(n_nodes <= 0.001) or np.any(n_nodes >= 0.999):
            return 1e10
        if np.any(n_weights <= 0):
            return 1e10
            
        w_norm = n_weights / np.sum(n_weights)
        return soft_max_error(n_nodes, w_norm, beta=50.0)

    try:
        # Use a smaller population and fewer iterations to fit within time
        result = optimize.differential_evolution(
            objective_de,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-10,
            polish=False # We will polish manually
        )
        
        if result.fun < 1e-10:
            opt_nodes = result.x[:8]
            opt_weights = result.x[8:]
            opt_nodes = np.sort(opt_nodes)
            if np.all(opt_nodes > 0.001) and np.all(opt_nodes < 0.999) and np.all(np.diff(opt_nodes) > 1e-4):
                if np.all(opt_weights > 0):
                    opt_weights = opt_weights / np.sum(opt_weights)
                    opt_err = compute_worst_error(opt_nodes, opt_weights)
                    if opt_err < best_err:
                        best_nodes = opt_nodes
                        best_weights = opt_weights
                        best_err = opt_err
    except Exception:
        pass

    # Local refinement using Nelder-Mead from the best DE result
    if time.time() < deadline:
        x0 = np.concatenate([best_nodes, best_weights])
        try:
            result_nm = optimize.minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result_nm.fun < 1e-10:
                opt_nodes = result_nm.x[:8]
                opt_weights = result_nm.x[8:]
                opt_nodes = np.sort(opt_nodes)
                if np.all(opt_nodes > 0.001) and np.all(opt_nodes < 0.999) and np.all(np.diff(opt_nodes) > 1e-4):
                    if np.all(opt_weights > 0):
                        opt_weights = opt_weights / np.sum(opt_weights)
                        opt_err = compute_worst_error(opt_nodes, opt_weights)
                        if opt_err < best_err:
                            best_nodes = opt_nodes
                            best_weights = opt_weights
                            best_err = opt_err
        except Exception:
            pass

    # Final validation and normalization
    best_nodes = np.sort(best_nodes)
    best_weights = best_weights / np.sum(best_weights)

    # Ensure constraints are strictly met
    assert len(best_nodes) == 8
    assert len(best_weights) == 8
    assert np.all(best_nodes > 0) and np.all(best_nodes < 1)
    assert np.all(np.diff(best_nodes) > 0)
    assert np.all(best_weights > 0)
    assert abs(np.sum(best_weights) - 1.0) < 1e-12

    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END

import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s limit
    
    n = 8  # Number of quadrature nodes
    
    # Correct E(a) for CPL model in flat universe with Omega_m=0.3, Omega_de=0.7
    def E_of_a(a, w0, wa):
        """H(a)/H0 for CPL model with Omega_m=0.3, Omega_de=0.7"""
        a_safe = np.maximum(a, 1e-10)
        omega_m = 0.3
        omega_de = 0.7
        # E^2 = Omega_m * a^{-3} + Omega_de * a^{3(1+w0+wa)} * exp(3*wa*(1-a))
        term_m = omega_m * a_safe**(-3)
        term_de = omega_de * a_safe**(3*(1+w0+wa)) * np.exp(3*wa*(1-a_safe))
        return np.sqrt(term_m + term_de)
    
    # The integrand for the quadrature: f(t) = 1/E(a(t)) where a = 1/(1+t)
    def integrand_f(t, w0, wa):
        """f(t) = 1/E(a(t)) where a = 1/(1+t), so t is in [0, z]"""
        a = 1.0 / (1.0 + t)
        return 1.0 / E_of_a(a, w0, wa)
    
    # Reference integral using dense sampling
    def reference_integral(z, w0, wa, n_points=1000):
        """High-accuracy reference: (1+z)*z*∫_0^1 f(t)dt where f(t)=1/E(1/(1+zt))"""
        t_arr = np.linspace(0, 1, n_points)
        f = integrand_f(t_arr, w0, wa)
        integral = np.trapz(f, t_arr)
        return (1.0 + z) * z * integral
    
    # Quadrature approximation
    def quadrature_value(z, nodes, weights, w0, wa):
        """Compute quadrature: (1+z)*z*sum_i w_i * f(nodes_i)"""
        f_vals = integrand_f(nodes, w0, wa)
        return (1.0 + z) * z * np.sum(weights * f_vals)
    
    # Define diverse test cosmologies
    model_list = [
        (-1.0, 0.0),    # Lambda CDM
        (-1.0, 0.5),    # Quintessence-like
        (-0.5, -0.5),   # Phantom-like
        (0.0, 0.0),     # w=0 constant
        (1.0, 0.0),     # w=1 constant
        (-1.5, 0.5),    # Strong phantom
        (-0.9, 0.3),    # Mildly varying
        (-1.1, -0.2),   # Another variation
        (-0.7, 0.7),    # Strongly varying
        (-1.3, 0.1),    # Phantom mild
        (-0.3, -1.0),   # Another model
        (0.5, 0.5),     # High w model
    ]
    
    # Test redshifts
    z_list = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
    
    # Compute reference integrals for all (z, model) pairs
    ref_values = {}
    for z in z_list:
        for w0, wa in model_list:
            ref_values[(z, w0, wa)] = reference_integral(z, w0, wa, n_points=2000)
    
    # Chebyshev basis for moment matching
    def chebyshev_basis(x, n_max=7):
        """Evaluate Chebyshev polynomials T_0 through T_n_max at x"""
        x = np.atleast_1d(x)
        basis = np.zeros((len(x), n_max+1))
        basis[:, 0] = 1.0
        if n_max >= 1:
            basis[:, 1] = x
        for k in range(2, n_max+1):
            basis[:, k] = 2*x*basis[:, k-1] - basis[:, k-2]
        return basis
    
    # Moment matching: select representative models and match moments
    moment_models = [
        (-1.0, 0.0),
        (-0.5, -0.5),
        (-1.5, 0.5),
        (0.0, 0.0),
        (-0.9, 0.3),
    ]
    
    def moment_residuals(nodes, weights):
        """Compute moment matching residuals for representative models"""
        residuals = []
        for w0, wa in moment_models:
            for k in range(4):  # Match first 4 moments
                # Reference moment: ∫_0^1 f(t) * T_k(t) dt
                t_arr = np.linspace(0, 1, 500)
                f_arr = integrand_f(t_arr, w0, wa)
                basis_vals = chebyshev_basis(t_arr, 3)[:, k]
                ref_moment = np.trapz(f_arr * basis_vals, t_arr)
                
                # Quadrature approximation
                f_i = integrand_f(nodes, w0, wa)
                basis_i = chebyshev_basis(nodes, 3)[:, k]
                quad_moment = np.sum(weights * f_i * basis_i)
                
                residuals.append((quad_moment - ref_moment) / max(abs(ref_moment), 1e-10))
        
        return residuals
    
    def worst_case_error(nodes, weights):
        """Compute worst relative error across all test (z, model) pairs"""
        max_err = 0.0
        for z in z_list:
            for w0, wa in model_list:
                ref = ref_values[(z, w0, wa)]
                quad = quadrature_value(z, nodes, weights, w0, wa)
                if abs(ref) > 1e-15:
                    rel_err = abs(quad - ref) / abs(ref) * 100.0
                    max_err = max(max_err, rel_err)
        return max_err
    
    def combined_objective(params):
        """Combined objective: worst-case error + moment matching penalty"""
        if time.time() > deadline - 5:
            return 1e10
        
        # Extract and transform parameters
        node_params = params[:8]
        weight_params = params[8:16]
        
        # Map node_params to (0,1) using sigmoid, then enforce ordering
        raw_nodes = 1.0 / (1.0 + np.exp(-node_params))
        sorted_nodes = np.sort(raw_nodes)
        # Add small offsets to ensure strict ordering
        nodes = sorted_nodes + np.arange(8) * 0.001
        nodes = np.clip(nodes, 0.01, 0.99)
        
        # Weights via softmax
        exp_w = np.exp(weight_params - np.max(weight_params))
        weights = exp_w / np.sum(exp_w)
        
        # Worst-case error
        wc_err = worst_case_error(nodes, weights)
        
        # Moment matching penalty
        res = moment_residuals(nodes, weights)
        moment_penalty = np.max(np.abs(res)) if len(res) > 0 else 0
        
        # Combine: prioritize worst-case error, use moments as regularizer
        return wc_err + 0.1 * moment_penalty
    
    # Initial guess from Gauss-Legendre quadrature
    from numpy.polynomial.legendre import leggauss
    nodes_init, weights_init = leggauss(8)
    nodes_init = 0.5 * (nodes_init + 1.0)  # Map from [-1,1] to [0,1]
    
    # Invert sigmoid to get initial node_params
    nodes_init_clipped = np.clip(nodes_init, 0.01, 0.99)
    node_params_init = np.log(nodes_init_clipped / (1.0 - nodes_init_clipped))
    weight_params_init = np.log(weights_init)
    params_init = np.concatenate([node_params_init, weight_params_init])
    
    best_params = params_init.copy()
    best_score = combined_objective(params_init)
    
    # Run differential evolution for global search
    try:
        bounds = [(-3, 3)] * 8 + [(-2, 2)] * 8
        result = differential_evolution(
            combined_objective,
            bounds,
            maxiter=50,
            tol=1e-6,
            popsize=15,
            seed=42,
            polish=False,
            init='sobol',
            updating='deferred'
        )
        
        if result.fun < best_score:
            best_params = result.x.copy()
            best_score = result.fun
            
    except Exception:
        pass
    
    # Check time for local refinement
    if time.time() < deadline - 20:
        try:
            result_local = minimize(
                combined_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if result_local.fun < best_score:
                best_score = result_local.fun
                best_params = result_local.x.copy()
                
        except Exception:
            pass
    
    # Final polish with Powell if time permits
    if time.time() < deadline - 10:
        try:
            result_powell = minimize(
                combined_objective,
                best_params,
                method='Powell',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if result_powell.fun < best_score:
                best_score = result_powell.fun
                best_params = result_powell.x.copy()
                
        except Exception:
            pass
    
    # Extract final nodes and weights
    node_params = best_params[:8]
    weight_params = best_params[8:16]
    
    raw_nodes = 1.0 / (1.0 + np.exp(-node_params))
    sorted_nodes = np.sort(raw_nodes)
    nodes = sorted_nodes + np.arange(8) * 0.001
    nodes = np.clip(nodes, 0.01, 0.99)
    
    exp_w = np.exp(weight_params - np.max(weight_params))
    weights = exp_w / np.sum(exp_w)
    
    # Ensure nodes are strictly increasing in (0,1)
    if np.any(nodes <= 0) or np.any(nodes >= 1) or np.any(np.diff(nodes) <= 0):
        nodes = (np.arange(n) + 0.5) / n
        weights = np.ones(n) / n
    else:
        # Ensure weights are positive and sum to 1
        if np.any(weights <= 0) or abs(np.sum(weights) - 1.0) > 1e-12:
            weights = weights / np.sum(weights)
            if abs(np.sum(weights) - 1.0) > 1e-12:
                nodes = (np.arange(n) + 0.5) / n
                weights = np.ones(n) / n
    
    # Final constraint check
    if np.any(nodes <= 0) or np.any(nodes >= 1) or np.any(np.diff(nodes) <= 0):
        nodes = (np.arange(n) + 0.5) / n
        weights = np.ones(n) / n
    if np.any(weights <= 0):
        weights = np.ones(n) / n
    if abs(np.sum(weights) - 1.0) > 1e-12:
        weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}

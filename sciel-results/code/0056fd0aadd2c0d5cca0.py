"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def _E_of_z_vec(z, Om0, Ol0, w0, w1):
    """E(z) = H(z)/H0 for flat CPL model. Vectorized."""
    a = 1.0 / (1.0 + z)
    a3 = a ** 3
    exponent = -3.0 * w1 * (1.0 - a)
    de_factor = a ** (-3.0 * (1.0 + w0)) * np.exp(exponent)
    E2 = Om0 / a3 + Ol0 * de_factor
    return np.sqrt(np.maximum(E2, 1e-300))

def construct():
    t_start = time.time()
    deadline = t_start + 85.0

    # Design cosmologies: broader coverage in parameter space
    design_cosmos = [
        (0.30, 0.70, -1.00,  0.00),
        (0.27, 0.73, -1.00,  0.00),
        (0.30, 0.70, -1.00,  1.00),
        (0.30, 0.70, -1.00, -1.00),
        (0.20, 0.80, -1.20,  0.50),
        (0.35, 0.65, -0.80,  0.30),
        (0.25, 0.75, -1.50,  1.00),
        (0.30, 0.70, -0.50, -0.50),
        (0.30, 0.70, -1.00,  0.50),
        (0.30, 0.70, -1.50, -0.50),
        (0.30, 0.70, -0.80, -1.00),
        (0.25, 0.75, -1.00,  0.80),
        (0.28, 0.72, -0.90,  0.20),
        (0.32, 0.68, -1.10, -0.30),
        (0.30, 0.70, -1.00, -0.50),
    ]
    z_max_list = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]

    # Precompute true integrals using quad
    true_vals = []
    for (Om0, Ol0, w0, w1) in design_cosmos:
        for z_max in z_max_list:
            def integrand_u(u, z_max=z_max, Om0=Om0, Ol0=Ol0, w0=w0, w1=w1):
                z_val = z_max * u
                if z_val < 1e-15:
                    return 1.0
                return 1.0 / _E_of_z_vec(z_val, Om0, Ol0, w0, w1)
            result, _ = quad(integrand_u, 0, 1, limit=200)
            true_val = z_max * result
            true_vals.append((Om0, Ol0, w0, w1, z_max, true_val))

    n_design = len(true_vals)
    true_vals_arr = np.array([tv[5] for tv in true_vals])
    cosmos_params = np.array([tv[:4] for tv in true_vals])
    z_max_arr = np.array([tv[4] for tv in true_vals])

    def _compute_inv_E(nodes, cosmos_params, z_max_arr):
        """Compute 1/E for all design points and nodes. Returns (n_design, 8) array."""
        z_eval = z_max_arr[:, None] * nodes[None, :]
        a = 1.0 / (1.0 + z_eval)
        a3 = a ** 3
        exponent = -3.0 * cosmos_params[:, 3:4] * (1.0 - a)
        de_factor = a ** (-3.0 * (1.0 + cosmos_params[:, 2:3])) * np.exp(exponent)
        E2 = cosmos_params[:, 0:1] / a3 + cosmos_params[:, 1:2] * de_factor
        E = np.sqrt(np.maximum(E2, 1e-300))
        return 1.0 / E

    def _worst_rel_error(nodes, weights):
        """Compute worst relative error in percent."""
        inv_E = _compute_inv_E(nodes, cosmos_params, z_max_arr)
        approx = z_max_arr * (inv_E @ weights)
        rel_err = np.abs(approx - true_vals_arr) / np.maximum(np.abs(true_vals_arr), 1e-300) * 100.0
        return np.max(rel_err)

    def _smooth_max_rel_error(nodes, weights, beta=50.0):
        """Compute smoothed max relative error using log-sum-exp."""
        inv_E = _compute_inv_E(nodes, cosmos_params, z_max_arr)
        approx = z_max_arr * (inv_E @ weights)
        rel_err = np.abs(approx - true_vals_arr) / np.maximum(np.abs(true_vals_arr), 1e-300) * 100.0
        # Soft max via log-sum-exp
        max_val = np.max(rel_err)
        if max_val < 1e-15:
            return 0.0
        scaled = beta * rel_err
        max_scaled = np.max(scaled)
        logsumexp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        return logsumexp / beta

    def _param_to_nodes_weights(raw_params):
        """Convert raw parameters to nodes and weights.
        raw_params[0:8] -> nodes via tanh
        raw_params[8:16] -> weights via softmax
        """
        nodes_raw = raw_params[:8]
        weights_raw = raw_params[8:]
        
        nodes = 0.5 + 0.5 * np.tanh(nodes_raw)
        # Ensure strictly increasing
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        order = np.argsort(nodes)
        nodes = nodes[order]
        
        # Softmax for weights
        weights = np.exp(weights_raw - np.max(weights_raw))
        weights = weights / np.sum(weights)
        weights = weights[order]
        
        return nodes, weights

    def _objective_smooth(raw_params, beta=50.0):
        """Smooth objective for optimization."""
        nodes, weights = _param_to_nodes_weights(raw_params)
        if np.any(np.diff(nodes) <= 1e-12):
            return 1e10
        return _smooth_max_rel_error(nodes, weights, beta)

    # Phase 1: Multiple random restarts with smooth objective
    best_nodes = None
    best_weights = None
    best_err = np.inf
    best_params = None

    # Initializations: Chebyshev, Gauss-Legendre, uniform, random
    n_init = 4
    init_params_list = []

    # Chebyshev nodes on [0,1]
    k = np.arange(8)
    cheb_nodes = 0.5 * (1.0 - np.cos(np.pi * (k + 0.5) / 8))
    cheb_nodes = np.sort(cheb_nodes)
    cheb_raw = np.arctanh(2.0 * cheb_nodes - 1.0)
    init_params_list.append(np.concatenate([cheb_raw, np.zeros(8)]))

    # Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    gl_01 = 0.5 * (gl_nodes + 1.0)
    gl_raw = np.arctanh(2.0 * gl_01 - 1.0)
    init_params_list.append(np.concatenate([gl_raw, np.zeros(8)]))

    # Uniform
    unif_nodes = np.linspace(0.05, 0.95, 8)
    unif_raw = np.arctanh(2.0 * unif_nodes - 1.0)
    init_params_list.append(np.concatenate([unif_raw, np.zeros(8)]))

    # Random
    rng = np.random.default_rng(42)
    rand_nodes = np.sort(rng.uniform(0.02, 0.98, 8))
    rand_raw = np.arctanh(2.0 * rand_nodes - 1.0)
    init_params_list.append(np.concatenate([rand_raw, rng.normal(0, 0.5, 8)]))

    for init_idx, x0 in enumerate(init_params_list):
        if time.time() > deadline - 20:
            break

        # Try different betas for annealing
        betas = [100.0, 50.0, 20.0, 10.0]
        
        for beta in betas:
            if time.time() > deadline - 15:
                break

            def obj_anneal(params, beta=beta):
                return _objective_smooth(params, beta)

            try:
                res = minimize(obj_anneal, x0, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                
                nodes, weights = _param_to_nodes_weights(res.x)
                err = _worst_rel_error(nodes, weights)
                
                if err < best_err:
                    best_err = err
                    best_nodes = nodes.copy()
                    best_weights = weights.copy()
                    best_params = res.x.copy()
                    
            except Exception:
                pass

    if best_nodes is None:
        # Fallback to Gauss-Legendre
        best_nodes, best_weights = gl_01, gl_weights.copy()
        best_err = _worst_rel_error(best_nodes, best_weights)

    # Phase 2: Fine-tune with Nelder-Mead on the best solution
    if time.time() < deadline - 15:
        def _nm_objective(raw_params):
            nodes, weights = _param_to_nodes_weights(raw_params)
            if np.any(np.diff(nodes) <= 1e-12):
                return 1e10
            return _worst_rel_error(nodes, weights)

        try:
            res_nm = minimize(_nm_objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True})
            
            nodes, weights = _param_to_nodes_weights(res_nm.x)
            err = _worst_rel_error(nodes, weights)
            
            if err < best_err:
                best_err = err
                best_nodes = nodes.copy()
                best_weights = weights.copy()
        except Exception:
            pass

    # Phase 3: Final refinement with L-BFGS-B using high beta
    if time.time() < deadline - 10:
        def _final_objective(raw_params):
            return _objective_smooth(raw_params, beta=200.0)

        try:
            res_final = minimize(_final_objective, best_params, method='L-BFGS-B',
                                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13})
            
            nodes, weights = _param_to_nodes_weights(res_final.x)
            err = _worst_rel_error(nodes, weights)
            
            if err < best_err:
                best_err = err
                best_nodes = nodes.copy()
                best_weights = weights.copy()
        except Exception:
            pass

    # Final cleanup and validation
    nodes = best_nodes
    weights = best_weights

    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)

    order = np.argsort(nodes)
    nodes = nodes[order]
    weights = weights[order]

    assert np.all(np.diff(nodes) > 0), "Nodes must be strictly increasing"
    assert np.all(weights > 0), "Weights must be positive"
    assert abs(np.sum(weights) - 1.0) < 1e-12, "Weights must sum to 1"

    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END

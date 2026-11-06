import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    # Define E(a) for CPL dark energy in flat universe
    def _E(a, w0, w1, Omega_m=0.3):
        Omega_de = 1.0 - Omega_m
        a_safe = np.maximum(a, 1e-10)
        return np.sqrt(Omega_m * a_safe**(-3) + Omega_de * a_safe**(-3*(1+w0)) * np.exp(3*w1*(1-a_safe)/a_safe))

    def _true_distance(z_max, w0, w1):
        def f(z):
            a = 1.0 / (1.0 + z)
            return (1.0 + z) / _E(a, w0, w1)
        val, _ = quad(f, 0, z_max, limit=200, epsabs=1e-12, epsrel=1e-12)
        return val

    def _quad_distance(z_max, nodes, weights, w0, w1):
        z_nodes = z_max * nodes
        a_nodes = 1.0 / (1.0 + z_nodes)
        vals = (1.0 + z_nodes) / _E(a_nodes, w0, w1)
        return z_max * np.sum(weights * vals)

    # Test cosmologies: diverse CPL models
    test_cosmos = [
        (-1.0, 0.0),      # LambdaCDM
        (-1.0, 0.5),      # CPL w1>0
        (-1.0, -0.5),     # CPL w1<0
        (-0.9, 0.3),      # Quintessence-like
        (-0.95, -0.3),    # Another variant
        (-1.2, 0.0),      # Phantom
        (-0.8, 0.5),      # Strong evolution
        (-1.0, 1.0),      # Strong CPL
        (-1.1, 0.4),      # Additional
        (-0.85, -0.4),    # Additional
        (-0.7, 0.8),      # Very strong evolution
        (-1.3, -0.2),     # Phantom with slight evolution
    ]

    # Redshifts to test
    test_zs = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

    # Pre-compute true distances for speed
    true_vals = np.zeros((len(test_cosmos), len(test_zs)))
    for i, (w0, w1) in enumerate(test_cosmos):
        for j, z in enumerate(test_zs):
            true_vals[i, j] = _true_distance(z, w0, w1)

    def _worst_error(nodes, weights):
        """Compute worst relative error over all test cosmologies and redshifts."""
        max_err = 0.0
        for i, (w0, w1) in enumerate(test_cosmos):
            for j, z in enumerate(test_zs):
                approx = _quad_distance(z, nodes, weights, w0, w1)
                true_val = true_vals[i, j]
                err = abs(approx - true_val) / true_val * 100.0
                if err > max_err:
                    max_err = err
        return max_err

    # Start with Gauss-Legendre as initial guess
    t, w = np.polynomial.legendre.leggauss(8)
    nodes_init = 0.5 * (t + 1.0)
    weights_init = 0.5 * w
    weights_init = weights_init / np.sum(weights_init)

    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    best_err = _worst_error(best_nodes, best_weights)

    # Phase 1: Differential Evolution for global exploration
    # Use log-transformed parameters to ensure constraints are naturally satisfied
    # nodes: use cumulative sum of exps to ensure strict increase in (0,1)
    # weights: use exps to ensure positive, then normalize
    
    def _transform_nodes(params):
        # params are 8 values, we map them to strictly increasing nodes in (0,1)
        # Using softmax-like cumulative sum
        exp_u = np.exp(params)
        cumsum_u = np.cumsum(exp_u)
        total_u = cumsum_u[-1]
        nodes = cumsum_u / total_u
        return nodes

    def _transform_weights(params):
        exp_v = np.exp(params)
        weights = exp_v / np.sum(exp_v)
        return weights

    def _objective_de(params):
        if time.time() > deadline - 2:
            return 1e10
        nodes = _transform_nodes(params[:8])
        weights = _transform_weights(params[8:])
        return _worst_error(nodes, weights)

    # Initialize DE parameters: 
    # For nodes, we want them to roughly map to nodes_init.
    # This is an inverse transform problem. 
    # Alternatively, we can just use bounds that cover the space.
    # Let's use a simpler parameterization for DE: 
    # Actually, the current program uses Nelder-Mead on raw nodes/weights.
    # Let's stick to the raw parameterization for consistency with the current best, 
    # but use a smarter search strategy.
    
    # The current program's strength is the local refinement. 
    # The inspiration's strength is the global search (DE).
    # Let's combine: Use DE to find a good starting point, then refine with Nelder-Mead.
    
    # However, DE on raw nodes/weights is tricky due to constraints.
    # Let's use the transformed parameterization for DE.
    
    # We need to find initial params for DE that map close to nodes_init.
    # This is hard to invert analytically. 
    # Instead, let's just run DE with random initial bounds and hope it finds a good basin.
    # Or, better: use the current best (GL) as a seed for local search, 
    # and use DE to explore other basins.
    
    # Let's try a hybrid: 
    # 1. Run DE with transformed params.
    # 2. Take the best result from DE.
    # 3. Refine with Nelder-Mead on raw params.
    
    bounds_de = [(-5, 5)] * 16  # 8 for nodes, 8 for weights
    
    try:
        # Run DE with limited iterations to save time
        result_de = differential_evolution(
            _objective_de,
            bounds_de,
            maxiter=50,
            popsize=15,
            tol=1e-5,
            seed=42,
            polish=False,  # We will polish manually
            init='sobol'
        )
        
        if result_de.fun < best_err:
            de_nodes = _transform_nodes(result_de.x[:8])
            de_weights = _transform_weights(result_de.x[8:])
            best_nodes = de_nodes
            best_weights = de_weights
            best_err = result_de.fun
    except Exception:
        pass

    # Phase 2: Local Refinement with Nelder-Mead on raw params
    # This is the "refinement machinery" from the current program.
    
    def _objective(params):
        if time.time() > deadline - 2:
            return 1e10
        nodes = params[:8]
        weights = params[8:]

        # Check node constraints: strictly increasing in (0,1)
        if np.any(nodes[1:] <= nodes[:-1]):
            return 1e10
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        if np.any(weights <= 0):
            return 1e10

        # Normalize weights
        weights = weights / np.sum(weights)

        return _worst_error(nodes, weights)

    # Refine from current best
    x0 = np.concatenate([best_nodes, best_weights])
    try:
        result_nm = minimize(_objective, x0, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
        if result_nm.fun < best_err:
            best_nodes = result_nm.x[:8]
            best_weights = result_nm.x[8:] / np.sum(result_nm.x[8:])
            best_err = result_nm.fun
    except Exception:
        pass

    # Phase 3: Random Restarts + Local Refinement
    # This is the "search" part from the current program, but we keep it.
    np.random.seed(42)
    for i in range(30):
        if time.time() > deadline - 5:
            break

        # Random perturbation of GL nodes
        nodes_try = nodes_init + np.random.uniform(-0.1, 0.1, 8)
        nodes_try = np.sort(nodes_try)
        nodes_try = np.clip(nodes_try, 0.005, 0.995)
        for j in range(1, 8):
            if nodes_try[j] <= nodes_try[j-1]:
                nodes_try[j] = nodes_try[j-1] + 1e-6

        weights_try = weights_init + np.random.uniform(-0.05, 0.05, 8)
        weights_try = np.maximum(weights_try, 1e-6)
        weights_try = weights_try / np.sum(weights_try)

        err = _worst_error(nodes_try, weights_try)
        if err < best_err:
            best_nodes = nodes_try.copy()
            best_weights = weights_try.copy()
            best_err = err

            # Refine from this better point
            if time.time() < deadline - 3:
                x0 = np.concatenate([best_nodes, best_weights])
                try:
                    result = minimize(_objective, x0, method='Nelder-Mead',
                                     options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                    if time.time() < deadline and result.fun < best_err:
                        best_nodes = result.x[:8]
                        best_weights = result.x[8:] / np.sum(result.x[8:])
                        best_err = result.fun
                except Exception:
                    pass

    # Final constraint enforcement
    best_nodes = np.sort(best_nodes)
    best_nodes = np.clip(best_nodes, 1e-8, 1.0 - 1e-8)
    for j in range(1, 8):
        if best_nodes[j] <= best_nodes[j-1]:
            best_nodes[j] = best_nodes[j-1] + 1e-8
    best_weights = np.maximum(best_weights, 1e-12)
    best_weights = best_weights / np.sum(best_weights)

    return {"nodes": best_nodes, "weights": best_weights}

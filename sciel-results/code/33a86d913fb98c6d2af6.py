import numpy as np
import time
import scipy.optimize as opt
from scipy.integrate import quad

def _E_cpl(z, w0, w1, Om):
    one_plus_z = 1.0 + z
    exp_out = 3.0 * (1.0 + w0) / 2.0
    exp_in = 3.0 * (1.0 - w0 - w1)
    term_m = Om * one_plus_z ** exp_in
    term_de = 1.0 - Om
    return (one_plus_z ** exp_out) * np.sqrt(term_m + term_de)

def _exact_integral(z, w0, w1, Om):
    f = lambda zp: 1.0 / _E_cpl(zp, w0, w1, Om)
    result, _ = quad(f, 0, z, limit=100)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin

    models = [
        (-1.0, 0.0, 0.3),
        (-1.2, 0.0, 0.3),
        (-0.8, 0.0, 0.3),
        (-1.0, -0.5, 0.3),
        (-1.0, 0.5, 0.3),
        (-0.8, -1.0, 0.3),
        (-1.2, 1.0, 0.3),
        (-1.0, 0.0, 0.2),
        (-1.0, 0.0, 0.35),
        (-0.9, 0.5, 0.25),
        (-1.1, -0.3, 0.28),
        (-0.9, -0.8, 0.32),
    ]
    z_test = np.array([0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])

    n_models = len(models)
    n_z = len(z_test)
    n_nodes = 8

    # Precompute exact integrals
    exact_vals = np.zeros((n_models, n_z))
    for i, (w0, w1, Om) in enumerate(models):
        for j, z in enumerate(z_test):
            exact_vals[i, j] = _exact_integral(z, w0, w1, Om)

    # Precompute 1/E values for each model, z, and node position
    # We'll compute these on-the-fly since they depend on node positions

    def _compute_E_inv(nodes, model_idx, z_idx):
        """Compute 1/E(z*node) for given nodes, model, and z."""
        w0, w1, Om = models[model_idx]
        z = z_test[z_idx]
        z_vals = z * nodes
        E_vals = _E_cpl(z_vals, w0, w1, Om)
        return 1.0 / E_vals

    def _solve_weights(nodes):
        """
        Given fixed nodes, solve for optimal weights that minimize worst-case relative error.
        The quadrature value for model i, z j is: z_j * sum_k w_k * (1/E_{i,j,k})
        Error: |z_j * sum_k w_k * invE_{i,j,k} - exact_{i,j}| / exact_{i,j}
        
        We solve: minimize max_{i,j} |sum_k A_{i,j,k} * w_k - b_{i,j}|
        where A_{i,j,k} = z_j * invE_{i,j,k} / exact_{i,j}
        and b_{i,j} = 1.0
        
        This is a linear programming problem.
        """
        n_constraints = n_models * n_z
        # A matrix: (n_constraints, n_nodes)
        A = np.zeros((n_constraints, n_nodes))
        for i in range(n_models):
            for j in range(n_z):
                invE = _compute_E_inv(nodes, i, j)
                A[i * n_z + j, :] = z_test[j] * invE / exact_vals[i, j]
        
        b = np.ones(n_constraints)
        
        # Minimize t subject to:
        # -t <= A*w - b <= t
        # sum(w) = 1
        # w >= 0
        
        # Variables: w_0,...,w_7, t  (9 variables)
        # c = [0,0,0,0,0,0,0,0,1]  (minimize t)
        c = np.zeros(n_nodes + 1)
        c[-1] = 1.0
        
        # Inequality constraints: A*w - t <= b  and  -(A*w) - t <= -b
        # i.e., A*w - t <= b  =>  [A, -1] * [w,t] <= b
        #      -A*w - t <= -b =>  [-A, -1] * [w,t] <= -b
        A_ub = np.zeros((2 * n_constraints, n_nodes + 1))
        b_ub = np.zeros(2 * n_constraints)
        
        for k in range(n_constraints):
            A_ub[k, :n_nodes] = A[k, :]
            A_ub[k, n_nodes] = -1.0
            b_ub[k] = b[k]
            
            A_ub[n_constraints + k, :n_nodes] = -A[k, :]
            A_ub[n_constraints + k, n_nodes] = -1.0
            b_ub[n_constraints + k] = -b[k]
        
        # Equality constraint: sum(w) = 1
        A_eq = np.zeros((1, n_nodes + 1))
        A_eq[0, :n_nodes] = 1.0
        b_eq = np.array([1.0])
        
        # Bounds: w_i >= 0, t >= 0
        bounds = [(0, None)] * n_nodes + [(0, None)]
        
        try:
            res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')
            if res.success:
                weights = res.x[:n_nodes]
                # Ensure positivity and normalization
                weights = np.maximum(weights, 1e-15)
                weights = weights / np.sum(weights)
                return weights, res.x[-1]
            else:
                # Fallback: equal weights
                return np.ones(n_nodes) / n_nodes, np.inf
        except:
            return np.ones(n_nodes) / n_nodes, np.inf

    def _worst_rel_error(nodes):
        """Compute worst relative error for given nodes with optimal weights."""
        weights, t_val = _solve_weights(nodes)
        # Verify by computing actual errors
        max_err = 0.0
        for i in range(n_models):
            for j in range(n_z):
                invE = _compute_E_inv(nodes, i, j)
                quad_val = z_test[j] * np.dot(weights, invE)
                rel_err = abs(quad_val - exact_vals[i, j]) / abs(exact_vals[i, j])
                if rel_err > max_err:
                    max_err = rel_err
        return max_err

    def _node_objective(node_params):
        """Objective for differential evolution: node_params in (0,1) space."""
        nodes = np.sort(node_params)
        # Ensure strict separation
        min_sep = 1e-6
        for i in range(1, n_nodes):
            if nodes[i] - nodes[i-1] < min_sep:
                nodes[i] = nodes[i-1] + min_sep
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        return _worst_rel_error(nodes)

    # Initialize with Gauss-Legendre nodes
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_nodes)
    nodes_gl = (x_gl + 1.0) / 2.0
    nodes_gl = np.sort(nodes_gl)
    
    best_nodes = nodes_gl.copy()
    best_err = _worst_rel_error(best_nodes)
    
    # Differential evolution for nodes
    if time.time() < deadline - 20:
        bounds_de = [(1e-6, 1.0 - 1e-6)] * n_nodes
        try:
            de_res = opt.differential_evolution(
                _node_objective, bounds_de,
                seed=42, maxiter=50, popsize=15,
                tol=1e-8, mutation=(0.3, 1.0), recombination=0.8,
                polish=True
            )
            if de_res.fun < best_err:
                best_nodes = np.sort(de_res.x)
                best_err = de_res.fun
        except:
            pass
    
    # Local refinement with Nelder-Mead on sorted nodes
    if time.time() < deadline - 10:
        # Parameterize as strictly increasing: x_1, x_2-x_1, ..., x_8-x_7
        def _param_to_nodes(params):
            x1 = params[0]
            diffs = params[1:]
            nodes = np.zeros(n_nodes)
            nodes[0] = x1
            for i in range(1, n_nodes):
                nodes[i] = nodes[i-1] + diffs[i-1]
            return nodes
        
        def _param_objective(params):
            nodes = _param_to_nodes(params)
            if nodes[0] < 1e-6 or nodes[-1] > 1.0 - 1e-6:
                return 1.0
            if np.any(np.diff(nodes) < 1e-6):
                return 1.0
            return _worst_rel_error(nodes)
        
        # Initial params from best_nodes
        init_params = np.zeros(n_nodes)
        init_params[0] = best_nodes[0]
        init_params[1:] = np.diff(best_nodes)
        
        try:
            nm_res = opt.minimize(_param_objective, init_params, method='Nelder-Mead',
                                  options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            if nm_res.fun < best_err:
                candidate_nodes = _param_to_nodes(nm_res.x)
                if candidate_nodes[0] > 1e-6 and candidate_nodes[-1] < 1.0 - 1e-6:
                    if np.all(np.diff(candidate_nodes) > 1e-6):
                        best_nodes = candidate_nodes
                        best_err = nm_res.fun
        except:
            pass
    
    # Get optimal weights for best nodes
    best_weights, _ = _solve_weights(best_nodes)
    
    # Ensure strict increasing and valid range
    best_nodes = np.sort(best_nodes)
    for i in range(1, n_nodes):
        if best_nodes[i] <= best_nodes[i-1]:
            best_nodes[i] = best_nodes[i-1] + 1e-8
    best_nodes = np.clip(best_nodes, 1e-8, 1.0 - 1e-8)
    
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.integrate import quad

def _E_of_z(z, Om0, Ol0, w0, w1):
    """E(z) = H(z)/H0 for flat CPL model."""
    a = 1.0 / (1.0 + z)
    a3 = a ** 3
    exponent = -3.0 * w1 * (1.0 - a)
    de_factor = a ** (-3.0 * (1.0 + w0)) * np.exp(exponent)
    E2 = Om0 / a3 + Ol0 * de_factor
    return np.sqrt(np.maximum(E2, 1e-300))

def _integrand(z, Om0, Ol0, w0, w1):
    """f(z) = z/(1+z) * 1/E(z)"""
    if z < 1e-15:
        return 0.0
    E = _E_of_z(z, Om0, Ol0, w0, w1)
    return z / (1.0 + z) / E

def _true_integral(z_max, Om0, Ol0, w0, w1):
    """High-accuracy adaptive integration for reference."""
    result, _ = quad(lambda z: _integrand(z, Om0, Ol0, w0, w1), 0, z_max, limit=200)
    return result

def _quadrature_at_nodes(nodes, weights, z_max, Om0, Ol0, w0, w1):
    """Compute quadrature approximation for given nodes/weights."""
    z_eval = z_max * nodes
    vals = np.array([_integrand(z_eval[i], Om0, Ol0, w0, w1) for i in range(len(nodes))])
    return z_max * np.dot(weights, vals)

def _solve_weights_lp(nodes, test_data):
    """
    Solve for optimal positive weights minimizing worst-case relative error using Linear Programming.
    Variables: w (n) + t (1)
    Minimize t
    Subject to:
      sum(w) = 1
      w >= 0
      |A_i @ w - true_i| <= t * true_i
    """
    n = len(nodes)
    m = len(test_data)
    
    # Precompute A and true_vals
    A = np.zeros((m, n))
    true_vals = np.zeros(m)
    for i, (Om0, Ol0, w0, w1, z_max, tv) in enumerate(test_data):
        z_eval = z_max * nodes
        for j in range(n):
            A[i, j] = z_max * _integrand(z_eval[j], Om0, Ol0, w0, w1)
        true_vals[i] = tv
    
    # Filter out zero true values (shouldn't happen for z_max > 0)
    mask = np.abs(true_vals) > 1e-15
    A = A[mask]
    true_vals = true_vals[mask]
    m = len(A)
    
    if m == 0:
        return np.ones(n) / n

    # Variables: w_0 ... w_{n-1}, t
    # Objective: minimize t -> c = [0, ..., 0, 1]
    c = np.zeros(n + 1)
    c[-1] = 1.0
    
    # Equality constraints: sum(w) = 1
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    
    # Inequality constraints:
    # 1) A_i @ w - t * true_i <= true_i  =>  [A_i, -true_i] @ v <= true_i
    # 2) -A_i @ w + t * true_i <= true_i  =>  [-A_i, true_i] @ v <= true_i
    # 3) -w_j <= 0 (handled by bounds, but linprog handles bounds separately)
    
    A_ub = np.zeros((2 * m, n + 1))
    b_ub = np.zeros(2 * m)
    
    for i in range(m):
        A_ub[2*i, :n] = A[i]
        A_ub[2*i, n] = -true_vals[i]
        b_ub[2*i] = true_vals[i]
        
        A_ub[2*i+1, :n] = -A[i]
        A_ub[2*i+1, n] = true_vals[i]
        b_ub[2*i+1] = true_vals[i]
    
    # Bounds: w_j >= 0, t >= 0
    bounds = [(0, None)] * n + [(0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            w = res.x[:n]
            # Ensure strictly positive and normalized
            w = np.maximum(w, 1e-15)
            w = w / np.sum(w)
            return w
        else:
            # Fallback: equal weights
            return np.ones(n) / n
    except Exception:
        return np.ones(n) / n

def _eval_worst_error(nodes, weights, test_data):
    """Evaluate worst-case relative error."""
    max_err = 0.0
    for (Om0, Ol0, w0, w1, z_max, tv) in test_data:
        if abs(tv) < 1e-15:
            continue
        approx = _quadrature_at_nodes(nodes, weights, z_max, Om0, Ol0, w0, w1)
        rel_err = abs(approx - tv) / abs(tv) * 100.0
        if rel_err > max_err:
            max_err = rel_err
    return max_err

def construct():
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin before 90s
    
    # Design cosmologies: span the expected parameter space
    design_cosmos = [
        (0.3, 0.7, -1.0, 0.0),
        (0.27, 0.73, -1.0, 0.0),
        (0.3, 0.7, -1.0, 1.0),
        (0.3, 0.7, -1.0, -1.0),
        (0.2, 0.8, -1.2, 0.5),
        (0.35, 0.65, -0.8, 0.3),
        (0.25, 0.75, -1.5, 1.0),
        (0.3, 0.7, -0.5, -0.5),
        (0.3, 0.7, -1.0, 0.5),
        (0.3, 0.7, -1.5, -0.5),
        (0.3, 0.7, -1.0, 0.8),
        (0.3, 0.7, -1.2, -0.8),
        (0.25, 0.75, -0.9, 0.6),
        (0.32, 0.68, -1.1, -0.6),
    ]
    z_max_list = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    # Precompute true integrals
    test_data = []
    for (Om0, Ol0, w0, w1) in design_cosmos:
        for z_max in z_max_list:
            tv = _true_integral(z_max, Om0, Ol0, w0, w1)
            test_data.append((Om0, Ol0, w0, w1, z_max, tv))
    
    # Initialize with Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    best_nodes = 0.5 * (gl_nodes + 1.0)
    best_weights = _solve_weights_lp(best_nodes, test_data)
    best_score = _eval_worst_error(best_nodes, best_weights, test_data)
    
    # Optimization loop:
    # We will use Nelder-Mead to optimize the nodes.
    # The objective function for Nelder-Mead will:
    # 1. Take a vector of 8 parameters representing nodes.
    # 2. Map them to valid nodes in (0,1) strictly increasing.
    # 3. Solve LP for optimal weights.
    # 4. Return worst-case error.
    
    def _node_objective(params):
        # Map params to nodes
        # Use a transformation that ensures nodes are in (0,1) and sorted.
        # Let params be u. 
        # x_1 = 0.5 * (1 + tanh(u_1)) / 8 + 1/16 ? No, simpler:
        # Let's just use the raw params as nodes and penalize if invalid.
        # Or better, use cumulative sums of positive values to ensure ordering.
        # Let u_i be unconstrained.
        # Let d_i = exp(u_i).
        # Let x_1 = d_1 / S, x_2 = (d_1+d_2)/S, ..., x_8 = 1.
        # This ensures x_i are in (0,1) and strictly increasing.
        # However, this might bias the nodes towards the edges if not careful.
        # Let's try a simpler mapping: x_i = 0.5 + 0.5 * tanh(u_i) but then sort?
        # Sorting breaks the correspondence between weight and node index if we swap them.
        # But weights are optimized for the specific node positions.
        # If we sort nodes, we must keep weights associated with the sorted nodes.
        # The LP solver assumes w[i] corresponds to nodes[i].
        # So if we change the order of nodes, we must change the order of weights?
        # No, the LP solver takes `nodes` and `test_data` and returns `weights` corresponding to `nodes`.
        # So if we define the objective as:
        # 1. Generate candidate nodes from params.
        # 2. Sort nodes.
        # 3. Solve LP for weights corresponding to these sorted nodes.
        # 4. Evaluate error.
        # This is valid.
        
        nodes = 0.5 + 0.5 * np.tanh(params)
        # Ensure they are in (0,1)
        nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
        # Sort them
        sort_idx = np.argsort(nodes)
        nodes_sorted = nodes[sort_idx]
        
        # Check if strictly increasing
        if np.any(np.diff(nodes_sorted) < 1e-12):
            return 1e6
            
        # Solve for weights
        weights = _solve_weights_lp(nodes_sorted, test_data)
        
        # Evaluate error
        err = _eval_worst_error(nodes_sorted, weights, test_data)
        return err

    # Initial params for Nelder-Mead
    # Inverse tanh of current nodes
    init_params = np.arctanh(2.0 * best_nodes - 1.0)
    
    # Run Nelder-Mead
    try:
        res = minimize(_node_objective, init_params, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        if res.fun < best_score:
            # Extract nodes
            nodes_opt = 0.5 + 0.5 * np.tanh(res.x)
            nodes_opt = np.clip(nodes_opt, 1e-8, 1.0 - 1e-8)
            sort_idx = np.argsort(nodes_opt)
            nodes_opt = nodes_opt[sort_idx]
            
            # Solve for final weights
            weights_opt = _solve_weights_lp(nodes_opt, test_data)
            score_opt = _eval_worst_error(nodes_opt, weights_opt, test_data)
            
            if score_opt < best_score:
                best_nodes = nodes_opt
                best_weights = weights_opt
                best_score = score_opt
    except Exception:
        pass
    
    # Final refinement: maybe do a few more iterations of alternating optimization
    # with a different random start for nodes?
    # Given time constraints, let's just do a local polish with Nelder-Mead again from the best found.
    if time.time() < deadline - 5:
        init_params2 = np.arctanh(2.0 * best_nodes - 1.0)
        try:
            res2 = minimize(_node_objective, init_params2, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if res2.fun < best_score:
                nodes_opt = 0.5 + 0.5 * np.tanh(res2.x)
                nodes_opt = np.clip(nodes_opt, 1e-8, 1.0 - 1e-8)
                sort_idx = np.argsort(nodes_opt)
                nodes_opt = nodes_opt[sort_idx]
                weights_opt = _solve_weights_lp(nodes_opt, test_data)
                score_opt = _eval_worst_error(nodes_opt, weights_opt, test_data)
                if score_opt < best_score:
                    best_nodes = nodes_opt
                    best_weights = weights_opt
                    best_score = score_opt
        except Exception:
            pass

    # Ensure validity
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    order = np.argsort(best_nodes)
    best_nodes = best_nodes[order]
    best_weights = best_weights[order]
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END

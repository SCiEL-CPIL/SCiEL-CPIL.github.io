import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 84.0  # 6s safety margin for 90s limit

    def E(a, Om, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL dark energy model. Vectorized."""
        a = np.asarray(a, dtype=float)
        matter = Om * a**(-3.0)
        
        if abs(w0 + 1.0) < 1e-10:
            # w0 = -1 case: DE density = (1-Om) * a^(-3(1+w0+wa)) * exp(3wa(1-a)/(1+w0))
            # Limit as w0->-1: DE = (1-Om) * exp(3*wa*(1-a))
            if abs(wa) < 1e-10:
                de = (1.0 - Om)
            else:
                de = (1.0 - Om) * np.exp(3.0 * wa * (1.0 - a))
        else:
            exponent = -3.0 * (1.0 + w0 + wa)
            exp_arg = 3.0 * wa * (1.0 - a) / (1.0 + w0)
            exp_arg = np.clip(exp_arg, -500.0, 500.0)
            de = (1.0 - Om) * np.power(a, exponent) * np.exp(exp_arg)
        
        return np.sqrt(matter + de)

    # Build a diverse training set
    Om_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    w0_vals = np.array([-2.0, -1.5, -1.2, -1.0, -0.8, -0.5, 0.0, 0.5])
    wa_vals = np.array([-1.0, -0.5, -0.3, 0.0, 0.3, 0.5, 1.0])
    z_vals = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
    
    # Build model list
    models = []
    for Om in Om_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                models.append((Om, w0, wa))
    
    n_models = len(models)
    n_z = len(z_vals)
    
    # Precompute true integrals using Gauss-Legendre with many points
    # Use 200-point Gauss-Legendre for high accuracy reference
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(200)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0
    
    true_integrals = np.zeros((n_models, n_z))
    for i, (Om, w0, wa) in enumerate(models):
        for j, z in enumerate(z_vals):
            a_gl = 1.0 / (1.0 + z * gl_nodes)
            e_gl = E(a_gl, Om, w0, wa)
            true_integrals[i, j] = np.dot(gl_weights, 1.0 / e_gl)
    
    # Precompute a_i for each z and candidate node set
    # Strategy: optimize nodes first with a fixed reasonable weight set, then optimize weights
    
    # Phase 1: Optimize nodes using a grid + local refinement
    # Start from Gauss-Legendre 8-point nodes
    gl8_nodes, gl8_weights = np.polynomial.legendre.leggauss(8)
    gl8_nodes = (gl8_nodes + 1.0) / 2.0
    gl8_weights = gl8_weights / 2.0
    
    def compute_max_error(nodes, weights):
        """Compute worst-case relative error over all training models and redshifts."""
        max_err = 0.0
        for i, (Om, w0, wa) in enumerate(models):
            for j, z in enumerate(z_vals):
                a_vals = 1.0 / (1.0 + z * nodes)
                e_vals = E(a_vals, Om, w0, wa)
                q_val = np.dot(weights, 1.0 / e_vals)
                true_val = true_integrals[i, j]
                if true_val > 0:
                    rel_err = abs(q_val - true_val) / true_val
                    if rel_err > max_err:
                        max_err = rel_err
        return max_err
    
    # Phase 1: Coarse search over node configurations
    # Use a structured grid perturbation around GL nodes
    best_nodes = gl8_nodes.copy()
    best_weights = gl8_weights.copy()
    best_err = compute_max_error(best_nodes, best_weights)
    
    # Try several initial node configurations
    candidate_node_sets = [gl8_nodes.copy()]
    
    # Add some alternative node distributions
    # Chebyshev nodes
    cheb_nodes = 0.5 * (1 - np.cos(np.pi * (np.arange(8) + 0.5) / 8))
    candidate_node_sets.append(cheb_nodes.copy())
    
    # Equally spaced
    eq_nodes = (np.arange(8) + 0.5) / 8.0
    candidate_node_sets.append(eq_nodes.copy())
    
    # Slightly biased toward lower x (since integrand may be larger near x=0)
    for bias in [0.3, 0.5, 0.7, 1.0, 1.5]:
        t = np.linspace(0, 1, 8)
        biased = t**bias
        # Ensure strictly in (0,1)
        biased = np.clip(biased, 1e-6, 1-1e-6)
        # Make strictly increasing
        for k in range(1, 8):
            if biased[k] <= biased[k-1]:
                biased[k] = biased[k-1] + 1e-4
        candidate_node_sets.append(biased)
    
    # For each candidate node set, find optimal weights via least squares
    # Since weights enter linearly: minimize max_j |sum_i w_i f_i(j) - true_j|
    # This is a linear programming problem, but we can use a simpler approach:
    # Minimize sum of squared errors, then check max error
    
    def find_optimal_weights(nodes):
        """Find weights that minimize worst-case error given fixed nodes."""
        # Build the design matrix: A[i,j] = f(node_i, model_j, z_j)
        # We want w such that A @ w ≈ true
        # A has shape (n_models*n_z, 8)
        n_samples = n_models * n_z
        A = np.zeros((n_samples, 8))
        true_vec = np.zeros(n_samples)
        
        idx = 0
        for i, (Om, w0, wa) in enumerate(models):
            for j, z in enumerate(z_vals):
                a_vals = 1.0 / (1.0 + z * nodes)
                e_vals = E(a_vals, Om, w0, wa)
                A[idx, :] = 1.0 / e_vals
                true_vec[idx] = true_integrals[i, j]
                idx += 1
        
        # Solve least squares: minimize ||Aw - true||^2
        # With constraint sum(w) = 1 and w > 0
        # First, get unconstrained LS solution
        w_ls, residuals, rank, sv = np.linalg.lstsq(A, true_vec, rcond=None)
        
        # Normalize to sum to 1
        w_sum = np.sum(w_ls)
        if w_sum > 0:
            w_ls = w_ls / w_sum
        
        # If any weights are negative, use a projection approach
        # Simple iterative projection
        w = w_ls.copy()
        for iteration in range(100):
            # Project onto sum(w) = 1
            w = w - (np.sum(w) - 1.0) / 8.0
            # Project onto w >= 0
            w = np.maximum(w, 0.0)
            # Re-normalize
            w_sum = np.sum(w)
            if w_sum > 1e-15:
                w = w / w_sum
            else:
                w = np.ones(8) / 8.0
                break
        
        # Also try uniform weights and a few other heuristics
        w_uniform = np.ones(8) / 8.0
        
        # Evaluate both
        err_ls = compute_max_error(nodes, w)
        err_unif = compute_max_error(nodes, w_uniform)
        
        if err_ls < err_unif:
            return w, err_ls
        else:
            return w_uniform, err_unif
    
    # Evaluate each candidate node set
    for nodes_cand in candidate_node_sets:
        if time.time() > deadline - 10:
            break
        # Ensure nodes are strictly in (0,1) and increasing
        nodes_cand = np.sort(nodes_cand)
        nodes_cand = np.clip(nodes_cand, 1e-6, 1 - 1e-6)
        for k in range(1, 8):
            if nodes_cand[k] <= nodes_cand[k-1]:
                nodes_cand[k] = nodes_cand[k-1] + 1e-4
        
        w_cand, err_cand = find_optimal_weights(nodes_cand)
        if err_cand < best_err:
            best_err = err_cand
            best_nodes = nodes_cand.copy()
            best_weights = w_cand.copy()
    
    # Phase 2: Local optimization of both nodes and weights jointly
    # Parameterize: nodes via sigmoid of parameters, weights via softmax
    
    def decode(params):
        u = params[:8]
        v = params[8:]
        nodes = 1.0 / (1.0 + np.exp(-u))
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        # Sort and make strictly increasing
        sort_idx = np.argsort(nodes)
        nodes = nodes[sort_idx]
        for k in range(1, 8):
            if nodes[k] <= nodes[k-1]:
                nodes[k] = nodes[k-1] + 1e-6
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        
        v_max = np.max(v)
        exp_v = np.exp(v - v_max)
        weights = exp_v / np.sum(exp_v)
        # Reorder weights to match sorted nodes
        weights = weights[sort_idx]
        
        return nodes, weights
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        nodes, weights = decode(params)
        return compute_max_error(nodes, weights)
    
    # Initialize from best found
    u_init = np.log(best_nodes / (1.0 - best_nodes))
    v_init = np.log(best_weights + 1e-15)
    x0 = np.concatenate([u_init, v_init])
    
    bounds = [(-10.0, 10.0)] * 16
    
    # Multiple restarts with L-BFGS-B
    best_x = x0.copy()
    best_val = objective(best_x)
    
    # Try a few perturbations
    rng = np.random.RandomState(42)
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 5:
            break
        if r == 0:
            x_try = x0.copy()
        else:
            x_try = x0 + rng.randn(16) * 0.5
        
        try:
            res = minimize(objective, x_try, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Decode final result
    nodes, weights = decode(best_x)
    
    # Final validation and cleanup
    nodes = np.sort(nodes)
    nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
    for k in range(1, 8):
        if nodes[k] <= nodes[k-1]:
            nodes[k] = nodes[k-1] + 1e-8
    nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
    
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)
    
    # Check validity
    if (not np.all(np.diff(nodes) > 1e-8) or not np.all(weights > 0) or
        abs(np.sum(weights) - 1.0) > 1e-10 or nodes[0] <= 0 or nodes[-1] >= 1):
        # Fallback to Gauss-Legendre
        nodes = gl8_nodes.copy()
        weights = gl8_weights.copy()
        weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _gauss_legendre_rule(n=8):
    """Standard Gauss-Legendre rule on [0,1]."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    nodes = (nodes + 1.0) / 2.0
    weights = weights / 2.0
    return nodes, weights

def _E_cpl(a, w0, wa, om_m, om_r, om_k=0.0):
    """E(a) = H(a)/H0 for CPL model."""
    # For flat universe, om_k = 0
    # om_de0 = 1 - om_m - om_r
    om_de = 1.0 - om_m - om_r
    h2 = om_m * a**(-3) + om_r * a**(-4) + om_de * a**(-3*(1+w0+wa))
    return np.sqrt(np.maximum(h2, 1e-30))

def _integrand(x, z, w0, wa, om_m, om_r, om_k):
    """The integrand 1/E(a) where a = 1/(1+z*x)."""
    a = 1.0 / (1.0 + z * x)
    return 1.0 / _E_cpl(a, w0, wa, om_m, om_r, om_k)

def _true_dL(z, w0, wa, om_m, om_r, om_k, n=4000):
    """Compute the true integral using high-resolution trapezoidal rule."""
    x = np.linspace(0, 1, n)
    f = _integrand(x, z, w0, wa, om_m, om_r, om_k)
    return np.trapz(f, x)

def _build_training_models():
    """Generate a diverse set of training cosmologies."""
    models = []
    z_vals = [0.5, 1.0, 2.0, 3.0]
    w0_vals = [-0.3, -0.1, 0.1, 0.3]
    wa_vals = [-0.3, -0.1, 0.1, 0.3]
    om_m_vals = [0.25, 0.30]
    
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                for om_m in om_m_vals:
                    om_r = 9e-5
                    om_k = 0.0
                    models.append((z, w0, wa, om_m, om_r, om_k))
    return models

def _compute_true_values(models):
    """Pre-compute true integral values for all models."""
    true_vals = []
    for (z, w0, wa, om_m, om_r, om_k) in models:
        true_vals.append(_true_dL(z, w0, wa, om_m, om_r, om_k))
    return np.array(true_vals)

def _get_integrand_matrix(nodes, models):
    """
    Compute the matrix of integrand values at the given nodes for all models.
    Shape: (num_models, num_nodes)
    """
    num_models = len(models)
    num_nodes = len(nodes)
    matrix = np.zeros((num_models, num_nodes))
    
    for i, (z, w0, wa, om_m, om_r, om_k) in enumerate(models):
        for j, node in enumerate(nodes):
            matrix[i, j] = _integrand(node, z, w0, wa, om_m, om_r, om_k)
    return matrix

def _solve_weights(matrix, true_vals):
    """
    Solve for weights that best approximate the true values.
    We want w such that matrix @ w ≈ true_vals.
    We also want w to be positive and sum to 1.
    Using least squares with constraints is complex, so we use a simple 
    non-negative least squares approach or just standard LSQ and then project.
    For now, let's use standard LSQ and check positivity.
    """
    # Standard least squares
    w, residuals, rank, sv = np.linalg.lstsq(matrix, true_vals, rcond=None)
    
    # Check if weights are positive
    if np.any(w < 0):
        # If not, try to project to positive cone? 
        # Or just return a large error penalty in the loss function.
        # For the inner optimization, we can just return the raw LSQ solution 
        # and let the outer loop handle validity, or we can use a penalty.
        # Let's use a penalty approach in the loss function instead of hard projection here.
        pass
        
    return w

def _loss_nodes(nodes, models, true_vals):
    """
    Loss function for node optimization.
    Computes the optimal weights for these nodes and returns the worst-case error.
    """
    # Check node validity
    if not np.all(np.diff(nodes) > 1e-8):
        return 1e10
    if nodes[0] < 1e-6 or nodes[-1] > 1.0 - 1e-6:
        return 1e10
        
    matrix = _get_integrand_matrix(nodes, models)
    
    # Solve for weights using least squares
    # We want to minimize ||matrix @ w - true_vals||^2
    # Subject to sum(w) = 1 and w > 0.
    # For speed in the inner loop, we'll just use unconstrained LSQ and penalize violations.
    w, _, _, _ = np.linalg.lstsq(matrix, true_vals, rcond=None)
    
    # Penalize negative weights and incorrect sum
    penalty = 0.0
    if np.any(w < 0):
        penalty += 1e5 * np.sum(np.clip(-w, 0, None))
    if abs(np.sum(w) - 1.0) > 1e-6:
        penalty += 1e5 * abs(np.sum(w) - 1.0)
        
    # Compute errors
    approx_vals = matrix @ w
    errors = np.abs(approx_vals - true_vals) / true_vals
    
    max_err = np.max(errors)
    
    return max_err + penalty

def construct():
    """
    Constructs an 8-node positive quadrature rule on [0,1] for luminosity distance.
    
    Strategy:
    1. Pre-compute true values for a set of representative CPL cosmologies.
    2. Use Nelder-Mead to optimize the 8 nodes, where the loss function 
       internally solves for the optimal weights via least squares.
    3. Once optimal nodes are found, solve for the optimal weights.
    4. Optionally polish with SLSQP if time permits.
    """
    n = 8
    deadline = time.time() + 55.0  # 55s budget to be safe
    
    # Step 1: Setup
    gl_nodes, gl_weights = _gauss_legendre_rule(n)
    models = _build_training_models()
    true_vals = _compute_true_values(models)
    
    # Step 2: Optimize Nodes
    # Start from GL nodes
    x0 = gl_nodes.copy()
    
    # Bounds for nodes (Nelder-Mead doesn't support bounds directly, 
    # so we rely on the loss function penalizing out-of-bounds or 
    # we can use a transformation. 
    # Let's use a transformation: nodes = (1 + tanh(y))/2 
    # This maps y in R to nodes in (0,1).
    # However, we need strictly increasing nodes. 
    # Simpler: just use Nelder-Mead on the nodes directly and hope it stays in bounds,
    # or use a penalty. The penalty in _loss_nodes handles bounds.
    
    # To ensure strict increase, we can parameterize nodes as cumulative sums of positive deltas.
    # Let nodes[0] = d0, nodes[i] = nodes[i-1] + d_i.
    # All d_i > 0 and sum(d_i) < 1.
    # This is 8 parameters.
    
    def _nodes_from_deltas(deltas):
        # deltas: 8 positive values
        # nodes[0] = deltas[0]
        # nodes[i] = nodes[i-1] + deltas[i]
        # We need sum(deltas) < 1.
        # Let's normalize deltas to sum to 1, then scale by (1 - epsilon) and add small offset?
        # Actually, simpler: just use raw nodes and let the optimizer handle it, 
        # but Nelder-Mead might step out of bounds.
        # Let's stick to raw nodes and use a strong penalty for violations.
        return deltas

    # Initial guess for Nelder-Mead
    initial_nodes = gl_nodes
    
    # Run Nelder-Mead
    best_nodes = initial_nodes
    best_loss = _loss_nodes(best_nodes, models, true_vals)
    
    try:
        res = minimize(
            _loss_nodes,
            initial_nodes,
            args=(models, true_vals),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if res.x is not None and time.time() < deadline - 5.0:
            candidate_nodes = res.x
            # Ensure nodes are sorted and within bounds
            candidate_nodes = np.sort(candidate_nodes)
            if candidate_nodes[0] < 1e-6:
                candidate_nodes[0] = 1e-6
            if candidate_nodes[-1] > 1.0 - 1e-6:
                candidate_nodes[-1] = 1.0 - 1e-6
            
            # Check if this is an improvement
            cand_loss = _loss_nodes(candidate_nodes, models, true_vals)
            if cand_loss < best_loss:
                best_nodes = candidate_nodes
                best_loss = cand_loss
                
    except Exception:
        pass

    # Step 3: Solve for Weights
    matrix = _get_integrand_matrix(best_nodes, models)
    weights, _, _, _ = np.linalg.lstsq(matrix, true_vals, rcond=None)
    
    # Check validity of weights
    if np.any(weights < 0) or abs(np.sum(weights) - 1.0) > 1e-6:
        # Fallback to GL if weights are invalid
        return {
            "nodes": gl_nodes,
            "weights": gl_weights
        }
    
    # Normalize weights just in case
    weights = weights / np.sum(weights)
    
    # Step 4: Optional Polish with SLSQP
    if time.time() < deadline - 10.0:
        try:
            def _full_loss(params):
                nodes = params[:8]
                log_weights = params[8:]
                w = np.exp(log_weights)
                w = w / np.sum(w)
                
                if not np.all(np.diff(nodes) > 1e-8):
                    return 1e10
                if nodes[0] < 1e-6 or nodes[-1] > 1.0 - 1e-6:
                    return 1e10
                    
                matrix = _get_integrand_matrix(nodes, models)
                approx_vals = matrix @ w
                errors = np.abs(approx_vals - true_vals) / true_vals
                return np.max(errors)
                
            x0 = np.concatenate([best_nodes, np.log(weights)])
            
            bounds = [(1e-6, 1.0 - 1e-6) for _ in range(8)] + [(-10, 10) for _ in range(8)]
            cons = []
            for i in range(7):
                def make_con(i):
                    def con(params):
                        return params[i+1] - params[i] - 1e-8
                    return con
                cons.append({'type': 'ineq', 'fun': make_con(i)})
                
            res2 = minimize(
                _full_loss,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 50, 'ftol': 1e-10}
            )
            
            if res2.success and time.time() < deadline - 2.0:
                final_nodes = res2.x[:8]
                final_log_weights = res2.x[8:]
                final_weights = np.exp(final_log_weights)
                final_weights = final_weights / np.sum(final_weights)
                
                if np.all(np.diff(final_nodes) > 1e-8) and np.all(final_weights > 0):
                    best_nodes = final_nodes
                    weights = final_weights
                    
        except Exception:
            pass

    return {
        "nodes": best_nodes,
        "weights": weights
    }
# EVOLVE-BLOCK-END

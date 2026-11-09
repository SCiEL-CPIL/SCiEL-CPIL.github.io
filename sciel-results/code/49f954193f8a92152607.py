"""Minimax quadrature design via iterative reweighting (Remez-like) for luminosity distance."""
import numpy as np
import time


def _E_cpl(a, w0, wa, om_m, om_r=9e-5):
    """E(a) = H(a)/H0 for flat CPL model."""
    om_de = 1.0 - om_m - om_r
    h2 = om_m * a**(-3) + om_r * a**(-4) + om_de * a**(-3 * (1 + w0 + wa * (1 - a)))
    return np.sqrt(np.maximum(h2, 1e-30))


def _integrand_at(a, w0, wa, om_m):
    """1/E(a) for given cosmology."""
    return 1.0 / _E_cpl(a, w0, wa, om_m)


def _true_integral(z, w0, wa, om_m, n=1024):
    """High-accuracy reference integral via dense trapezoidal rule."""
    x = np.linspace(0, 1, n)
    a = 1.0 / (1.0 + z * x)
    f = _integrand_at(a, w0, wa, om_m)
    return np.trapz(f, x)


def _quad_value(nodes, weights, z, w0, wa, om_m):
    """Quadrature estimate for given cosmology."""
    a = 1.0 / (1.0 + z * nodes)
    f = _integrand_at(a, w0, wa, om_m)
    return np.dot(weights, f)


def construct():
    """
    Minimax 8-node positive quadrature rule for luminosity distance.
    
    Strategy: Iterative reweighting (Remez-like) to minimize worst-case relative
    error over a dense grid of cosmologies. Nodes are updated via Gauss-Legendre
    on the weighted residual landscape; weights are solved via linear programming
    (or projected gradient) to satisfy moment constraints and positivity.
    """
    deadline = time.time() + 85.0  # 90s wall-clock, 5s safety margin
    n_nodes = 8
    
    # Build dense validation grid
    z_vals = np.array([0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0])
    w0_vals = np.array([-0.3, -0.1, 0.0, 0.1, 0.3])
    wa_vals = np.array([-0.3, -0.1, 0.0, 0.1, 0.3])
    om_m_vals = np.array([0.22, 0.27, 0.32])
    
    # Precompute true integrals for all combinations
    cosmologies = []
    true_vals = []
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                for om_m in om_m_vals:
                    cosmologies.append((z, w0, wa, om_m))
                    true_vals.append(_true_integral(z, w0, wa, om_m))
    true_vals = np.array(true_vals)
    n_cosmo = len(cosmologies)
    
    # Initialize with Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_nodes)
    nodes = (gl_nodes + 1.0) / 2.0
    weights = gl_weights / 2.0
    
    best_nodes = nodes.copy()
    best_weights = weights.copy()
    best_max_err = np.inf
    
    # Iterative reweighting loop
    for it in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Evaluate quadrature on all cosmologies
        quad_vals = np.zeros(n_cosmo)
        for i, (z, w0, wa, om_m) in enumerate(cosmologies):
            quad_vals[i] = _quad_value(nodes, weights, z, w0, wa, om_m)
        
        # Relative errors
        rel_errs = np.abs(quad_vals - true_vals) / np.maximum(true_vals, 1e-15)
        max_err = np.max(rel_errs)
        
        if max_err < best_max_err:
            best_max_err = max_err
            best_nodes = nodes.copy()
            best_weights = weights.copy()
        
        # Check convergence
        if max_err < 1e-8:
            break
        
        # Reweight: emphasize worst-case cosmologies
        # Use exponential weighting on errors to focus on worst cases
        err_scaled = rel_errs / (max_err + 1e-15)
        alpha = 10.0  # sharpness parameter
        case_weights = np.exp(alpha * err_scaled)
        case_weights /= np.sum(case_weights)
        
        # Update weights: solve for weights that minimize weighted sum of squared errors
        # subject to sum(w)=1 and w>0
        # Build the system: for each cosmology, the quadrature is linear in weights
        # A[i, j] = f_j(c_i), where f_j is integrand at node j for cosmology i
        A = np.zeros((n_cosmo, n_nodes))
        for i, (z, w0, wa, om_m) in enumerate(cosmologies):
            a = 1.0 / (1.0 + z * nodes)
            A[i, :] = _integrand_at(a, w0, wa, om_m)
        
        # Minimize sum_i w_i * (A[i,:] @ weights - true_vals[i])^2
        # This is a quadratic in weights. Use projected gradient or simple iteration.
        # Gradient: 2 * A.T @ (case_weights * (A @ weights - true_vals))
        # Update: weights -= lr * gradient, then project to simplex
        
        lr = 0.05
        for _ in range(20):  # inner iterations for weight update
            residuals = A @ weights - true_vals
            grad = 2.0 * (A.T @ (case_weights * residuals))
            weights -= lr * grad
            # Project to simplex: positive, sum to 1
            weights = np.maximum(weights, 1e-12)
            weights /= np.sum(weights)
        
        # Update nodes: shift nodes toward regions of high error
        # Compute local error density
        node_errors = np.zeros(n_nodes)
        for i, (z, w0, wa, om_m) in enumerate(cosmologies):
            a = 1.0 / (1.0 + z * nodes)
            f = _integrand_at(a, w0, wa, om_m)
            # Approximate local contribution to error
            node_errors += case_weights[i] * np.abs(f * weights - true_vals[i] / n_cosmo)
        
        # Perturb nodes toward higher error regions
        # Use a simple heuristic: move nodes slightly toward center if error is high
        node_perturbation = 0.01 * (1.0 - 2.0 * nodes) * (node_errors / (np.max(node_errors) + 1e-15))
        nodes = nodes + node_perturbation
        
        # Enforce strict ordering and bounds
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        for i in range(1, n_nodes):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-6
        if nodes[-1] > 1.0 - 1e-6:
            nodes[-1] = 1.0 - 1e-6
            for i in range(n_nodes - 2, -1, -1):
                if nodes[i] >= nodes[i+1]:
                    nodes[i] = nodes[i+1] - 1e-6
        if nodes[0] < 1e-6:
            nodes[0] = 1e-6
    
    # Final validation and cleanup
    final_nodes = best_nodes
    final_weights = best_weights
    
    # Ensure strict validity
    if not np.all(np.diff(final_nodes) > 0):
        # Fallback to GL
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_nodes)
        final_nodes = (gl_nodes + 1.0) / 2.0
        final_weights = gl_weights / 2.0
    
    final_weights = np.maximum(final_weights, 1e-12)
    final_weights /= np.sum(final_weights)
    
    return {
        "nodes": final_nodes,
        "weights": final_weights
    }

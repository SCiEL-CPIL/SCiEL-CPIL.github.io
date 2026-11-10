# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _cpl_e(a, wa0, wa1):
    """E(a) = H(a)/H0 for CPL model with Omega_m=0.3, Omega_L=0.7."""
    omega_m = 0.3
    omega_l = 0.7
    # E^2 = omega_m/a^3 + omega_l*(a/(1+a*wa1))^(3*(1+wa0))
    term_m = omega_m / (a**3)
    base = a / (1.0 + a * wa1)
    term_l = omega_l * (base ** (3.0 * (1.0 + wa0)))
    return np.sqrt(term_m + term_l)


def _compute_error(params, z_grid, wa0_grid, wa1_grid, n_dense=1000):
    """
    Compute the sum of squared relative errors for a given set of parameters.
    params: 16-vector [x1..x8, y1..y8]
    x defines nodes via softmax cumulative sum.
    y defines weights via softmax.
    """
    x = params[:8]
    y = params[8:]
    
    # Compute nodes
    exp_x = np.exp(x - np.max(x)) # for numerical stability
    cumsum_x = np.cumsum(exp_x)
    nodes = cumsum_x / cumsum_x[-1]
    
    # Compute weights
    exp_y = np.exp(y - np.max(y))
    weights = exp_y / np.sum(exp_y)
    
    total_error = 0.0
    count = 0
    
    # Pre-compute dense integration values for all grid points?
    # It's better to loop and compute, as n_dense is high.
    # To speed up, we can vectorize over z for fixed wa0, wa1 or vice versa.
    # Let's vectorize over z for each (wa0, wa1) pair.
    
    for wa0 in wa0_grid:
        for wa1 in wa1_grid:
            # True integral for all z in z_grid
            # Integral I(z) = ∫_0^1 du / E(a(z*u))
            # a(z*u) = 1/(1+z*u)
            
            # Dense integration setup
            u_dense = np.linspace(0, 1, n_dense + 1)
            u_mid = 0.5 * (u_dense[:-1] + u_dense[1:])
            du = u_dense[1] - u_dense[0]
            
            # Vectorize over z
            # z_grid shape (Nz,), u_mid shape (N_d,)
            # We need a matrix of a(z, u)
            # a = 1 / (1 + z * u)
            # z[:, None] * u_mid[None, :]
            
            z_col = z_grid[:, None]
            u_row = u_mid[None, :]
            
            a_matrix = 1.0 / (1.0 + z_col * u_row)
            E_matrix = _cpl_e(a_matrix, wa0, wa1)
            integrand_matrix = 1.0 / E_matrix
            
            # True values: sum over u for each z
            true_vals = np.sum(integrand_matrix, axis=1) * du
            
            # Quadrature approximation
            # a_quad = 1 / (1 + z * nodes)
            # nodes shape (8,)
            # z_col shape (Nz, 1)
            # nodes_row shape (1, 8)
            
            nodes_row = nodes[None, :]
            a_quad = 1.0 / (1.0 + z_col * nodes_row)
            E_quad = _cpl_e(a_quad, wa0, wa1)
            # Approx val for each z: sum_i weights[i] / E(a(z, n_i))
            # weights shape (8,)
            # E_quad shape (Nz, 8)
            approx_vals = np.sum(weights[None, :] / E_quad, axis=1)
            
            # Relative errors
            # Avoid division by zero
            mask = true_vals > 1e-12
            if np.any(mask):
                rel_errs = (approx_vals[mask] - true_vals[mask]) / true_vals[mask]
                total_error += np.sum(rel_errs**2)
                count += np.sum(mask)
                
    return total_error / max(count, 1)


def construct():
    # Define a dense grid of z values and CPL parameters to optimize over
    # We want to cover the range of interest.
    # z: 0.05 to 2.0
    # wa0: -1.0 to 1.0
    # wa1: -1.0 to 1.0
    
    z_grid = np.linspace(0.05, 2.0, 20)
    wa0_grid = np.linspace(-1.0, 1.0, 10)
    wa1_grid = np.linspace(-1.0, 1.0, 10)
    
    # Initial guess
    # Nodes: roughly uniform
    # Weights: roughly uniform
    # x_i such that nodes are uniform
    # If nodes are uniform, x_i should be constant?
    # n_i = i/8.
    # cumsum(exp(x)) / sum(exp(x)) = i/8.
    # If x is constant, n_i = i/8.
    x_init = np.zeros(8)
    y_init = np.zeros(8)
    
    params_init = np.concatenate([x_init, y_init])
    
    # Optimization
    # L-BFGS-B
    res1 = minimize(
        _compute_error,
        params_init,
        args=(z_grid, wa0_grid, wa1_grid),
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    # Powell refinement
    res2 = minimize(
        _compute_error,
        res1.x,
        args=(z_grid, wa0_grid, wa1_grid),
        method='Powell',
        options={'maxiter': 100, 'ftol': 1e-12, 'xtol': 1e-12}
    )
    
    params_opt = res2.x
    x_opt = params_opt[:8]
    y_opt = params_opt[8:]
    
    # Extract nodes and weights
    exp_x = np.exp(x_opt - np.max(x_opt))
    cumsum_x = np.cumsum(exp_x)
    nodes_opt = cumsum_x / cumsum_x[-1]
    
    exp_y = np.exp(y_opt - np.max(y_opt))
    weights_opt = exp_y / np.sum(exp_y)
    
    # Ensure strict monotonicity and bounds (should be handled by transform, but double check)
    # The transform guarantees 0 < n_1 < ... < n_8 < 1
    # Weights are positive and sum to 1
    
    return {"nodes": nodes_opt, "weights": weights_opt}
# EVOLVE-BLOCK-END

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import quad
import time

# Constants for cosmology
OM = 0.3
OR = 1e-4
ODE = 1.0 - OM - OR

def E_of_a(a, w0, w1):
    """
    Compute E(a) = H(a)/H0 for CPL dark energy model.
    a: scale factor
    w0, w1: CPL parameters
    """
    # Handle a=0 or very small
    if a < 1e-10:
        return np.inf
    
    # Exponent part for DE
    # exp(3 * w1 * (a - 1 - ln(a)))
    # Power part: a^(-3 * (1 + w0))
    
    # Numerical stability for ln(a) when a is close to 1
    # Use np.log
    log_term = np.log(a)
    exp_arg = 3.0 * w1 * (a - 1.0 - log_term)
    
    # Avoid overflow in exp
    if exp_arg > 700:
        exp_val = np.exp(700)
    else:
        exp_val = np.exp(exp_arg)
        
    power_term = a ** (-3.0 * (1.0 + w0))
    
    de_term = ODE * power_term * exp_val
    
    m_term = OM * a ** (-3.0)
    r_term = OR * a ** (-4.0)
    
    sum_term = m_term + r_term + de_term
    
    if sum_term < 0:
        return 0.0 # Should not happen for physical models
        
    return np.sqrt(sum_term)

def exact_dL(z, w0, w1):
    """
    Compute exact dimensionless luminosity distance (without c/H0 factor, but including the integral part).
    The formula is dL/(c/H0) = (1+z) * Integral_0^z dz'/E(a')
    """
    if z <= 0:
        return 0.0
        
    def integrand(z_prime):
        a_prime = 1.0 / (1.0 + z_prime)
        return 1.0 / E_of_a(a_prime, w0, w1)
        
    try:
        val, err = quad(integrand, 0, z, limit=100)
        return (1.0 + z) * val
    except Exception:
        return 0.0

def chebyshev_quadrature(w0, w1, n_nodes=8, n_cheb=20):
    """
    Construct an 8-node quadrature rule using Chebyshev spectral collocation.
    
    The idea: approximate 1/E(a) using a Chebyshev series, then find nodes and
    weights that exactly integrate the truncated series.
    
    We work in the variable t = a (scale factor), mapping to z-space via
    a = 1/(1+z*t) where t in [0,1] is the node parameter.
    
    Actually, let's work directly in the integral:
    dL/(c/H0) = (1+z) * integral_0^z dz'/E(1/(1+z'))
    
    Let f(z') = 1/E(1/(1+z'))
    Then dL/(c/H0) = (1+z) * integral_0^z f(z') dz'
    
    For a fixed z, we want: integral_0^z f(z') dz' ≈ z * sum_i w_i f(z_i)
    where z_i = z * nodes[i]
    
    This means: (1/z) * integral_0^z f(z') dz' ≈ sum_i w_i f(z_i)
    
    So we're approximating the average value of f over [0,z].
    
    Using Chebyshev nodes in [0,1] for the normalized variable t = z'/z:
    integral_0^z f(z') dz' = z * integral_0^1 f(z*t) dt
    
    So we need: integral_0^1 g(t) dt ≈ sum_i w_i g(t_i)
    where g(t) = f(z*t) = 1/E(1/(1+z*t))
    
    The challenge: g depends on z, so nodes/weights can't be universal in t-space
    unless we use a different approach.
    
    Alternative: Work in a-space directly.
    a = 1/(1+z), so as z goes from 0 to z_max, a goes from 1 to 1/(1+z_max).
    
    Let's use the substitution: for the quadrature, we evaluate at a_i = 1/(1+z*nodes[i]).
    The integral becomes:
    integral_0^z dz'/E(1/(1+z')) 
    
    With u = 1/(1+z'), du = -dz'/(1+z')^2 = -u^2 dz'
    So dz' = -du/u^2
    
    When z'=0: u=1
    When z'=z: u=1/(1+z)
    
    integral_0^z dz'/E(1/(1+z')) = integral_{1/(1+z)}^1 du/(u^2 * E(u))
    
    So: dL/(c/H0) = (1+z) * integral_{1/(1+z)}^1 du/(u^2 * E(u))
    
    This is an integral over u from a to 1 where a = 1/(1+z).
    
    For a universal quadrature, we can use Gauss-Chebyshev or similar on [0,1]
    and adapt. But the integration limits change with z.
    
    Let me try a different approach: use fixed nodes in the "normalized" space
    and solve for weights that minimize worst-case error across a set of test
    cosmologies and redshifts.
    """
    n = n_nodes
    
    # Use Chebyshev-Gauss-Lobatto nodes mapped to (0,1)
    # t_k = (1 + cos(k*pi/(n-1)))/2 for k=0,...,n-1
    k = np.arange(n)
    t = 0.5 * (1 + np.cos(np.pi * k / (n - 1)))
    t = t[::-1]  # Reverse to get increasing order
    t = 0.5 * t  # Scale to (0, 0.5) to avoid endpoints
    
    # Actually, let's use interior Chebyshev nodes
    k = np.arange(1, n)
    t = 0.5 * (1 + np.cos(np.pi * k / (n + 1)))
    t = t[::-1]  # Increasing order
    
    # Now find optimal weights by solving a linear system
    # We want the quadrature to be exact for Chebyshev polynomials up to degree n-1
    
    # Define test cosmologies
    test_zs = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0]
    test_w0s = [-1.5, -1.0, -0.5, 0.0, 0.5]
    test_w1s = [-1.0, 0.0, 1.0, 2.0]
    
    test_data = []
    for z in test_zs:
        for w0 in test_w0s:
            for w1 in test_w1s:
                test_data.append((z, w0, w1))
    
    # Build the system: for each test case, compute A_row and b
    # approx = (1+z) * z * sum_i w_i / E(a_i)
    # exact = (1+z) * integral
    # So we need: sum_i w_i / E(a_i) ≈ exact / ((1+z) * z)
    
    A = np.zeros((len(test_data), n))
    b = np.zeros(len(test_data))
    
    for idx, (z, w0, w1) in enumerate(test_data):
        a_vals = 1.0 / (1.0 + z * t)
        e_vals = E_of_a(a_vals, w0, w1)
        
        if np.any(e_vals <= 0) or np.any(np.isinf(e_vals)):
            continue
            
        A[idx, :] = 1.0 / e_vals
        exact = exact_dL(z, w0, w1)
        if exact > 0 and z > 0:
            b[idx] = exact / ((1.0 + z) * z)
    
    # Solve least squares with constraint sum(w) = 1
    # Use augmented system or project after solving
    
    try:
        w_ls, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return t, np.full(n, 1.0/n)
    
    # Project onto simplex: w >= 0, sum(w) = 1
    w_pos = np.clip(w_ls, 1e-12, None)
    w_pos = w_pos / np.sum(w_pos)
    
    return t, w_pos

def quad_dL(z, nodes, weights, w0, w1):
    """
    Compute approximate dimensionless luminosity distance using the quadrature rule.
    Formula: (1+z) * z * sum_i weights[i] / E(a_i)
    where a_i = 1 / (1 + z * nodes[i])
    """
    if z <= 0:
        return 0.0
        
    a_vals = 1.0 / (1.0 + z * nodes)
    e_vals = E_of_a(a_vals, w0, w1)
    
    # Handle potential zeros or infs in E
    if np.any(e_vals <= 0) or np.any(np.isinf(e_vals)):
        return 0.0
        
    sum_term = np.sum(weights / e_vals)
    return (1.0 + z) * z * sum_term

def evaluate_worst_error(nodes, weights, test_data):
    """
    Compute worst-case relative error for given nodes and weights.
    """
    max_err = 0.0
    
    for z, w0, w1 in test_data:
        exact = exact_dL(z, w0, w1)
        if exact == 0:
            continue
            
        approx = quad_dL(z, nodes, weights, w0, w1)
        rel_err = abs((approx - exact) / exact)
        
        if rel_err > max_err:
            max_err = rel_err
            
    return max_err

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    # Define comprehensive test data
    zs = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    w0s = [-1.5, -1.2, -1.0, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5]
    w1s = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    test_data = []
    for z in zs:
        for w0 in w0s:
            for w1 in w1s:
                test_data.append((z, w0, w1))
    
    # Strategy 1: Chebyshev-based initialization
    try:
        nodes_cheb, weights_cheb = chebyshev_quadrature(0.0, 0.0, n_nodes=8)
        elapsed = time.time() - start_time
        if elapsed > deadline:
            raise TimeoutError()
    except Exception:
        # Fallback to evenly spaced nodes
        nodes_cheb = (np.arange(8) + 0.5) / 8.0
        weights_cheb = np.full(8, 1.0/8)
    
    # Strategy 2: Try multiple node configurations and pick best
    candidate_configs = []
    
    # Config 1: Chebyshev nodes
    candidate_configs.append((nodes_cheb.copy(), weights_cheb.copy()))
    
    # Config 2: Evenly spaced
    nodes_even = (np.arange(8) + 0.5) / 8.0
    weights_even = np.full(8, 1.0/8)
    candidate_configs.append((nodes_even, weights_even))
    
    # Config 3: Gauss-Legendre nodes mapped to (0,1)
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = 0.5 * (x_gl + 1)  # Map [-1,1] to [0,1]
    weights_gl = w_gl / 2.0  # Scale weights
    weights_gl = weights_gl / np.sum(weights_gl)
    candidate_configs.append((nodes_gl, weights_gl))
    
    # Config 4: Gauss-Lobatto (Chebyshev) nodes
    k = np.arange(8)
    nodes_lob = 0.5 * (1 + np.cos(np.pi * k / 7.0))
    nodes_lob = nodes_lob[::-1]  # Increasing
    # Interior points only for positive weights
    nodes_lob_int = nodes_lob[1:-1]
    if len(nodes_lob_int) == 6:
        # Add endpoints slightly inside
        nodes_lob_int = np.concatenate([[nodes_lob_int[0] - 0.01], nodes_lob_int, [nodes_lob_int[-1] + 0.01]])
        # This gives 8 nodes
        # Use equal weights as initial guess
        weights_lob = np.full(8, 1.0/8)
        candidate_configs.append((nodes_lob_int, weights_lob))
    
    # Config 5: Log-spaced nodes
    nodes_log = np.geomspace(0.01, 0.99, 8)
    weights_log = np.full(8, 1.0/8)
    candidate_configs.append((nodes_log, weights_log))
    
    # Evaluate each configuration
    best_nodes = None
    best_weights = None
    best_error = np.inf
    
    for nodes, weights in candidate_configs:
        # Ensure nodes are strictly increasing in (0,1)
        if not np.all(np.diff(nodes) > 1e-12):
            continue
        if nodes[0] <= 0 or nodes[-1] >= 1:
            continue
            
        err = evaluate_worst_error(nodes, weights, test_data)
        
        if err < best_error:
            best_error = err
            best_nodes = nodes.copy()
            best_weights = weights.copy()
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
    
    # If we have time, refine the best configuration using optimization
    elapsed = time.time() - start_time
    if best_nodes is not None and elapsed < deadline - 10:
        try:
            # Refine using scipy minimize on the worst-case error
            def obj_to_minimize(params):
                v = params[:8]
                q = params[8:]
                
                # Sigmoid for nodes
                x = 1.0 / (1.0 + np.exp(-v))
                x = np.sort(x)
                
                # Softmax for weights
                q_max = np.max(q)
                exp_q = np.exp(q - q_max)
                w = exp_q / np.sum(exp_q)
                
                return evaluate_worst_error(x, w, test_data)
            
            # Initialize from best found
            v_init = np.log(best_nodes / (1 - best_nodes))
            q_init = np.log(best_weights + 1e-15)
            params_init = np.concatenate([v_init, q_init])
            
            result = minimize(
                obj_to_minimize,
                params_init,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            v_best = result.x[:8]
            q_best = result.x[8:]
            
            x_best = 1.0 / (1.0 + np.exp(-v_best))
            x_best = np.sort(x_best)
            
            q_max = np.max(q_best)
            exp_q = np.exp(q_best - q_max)
            w_best = exp_q / np.sum(exp_q)
            
            # Check validity
            if np.all(np.diff(x_best) > 1e-12) and x_best[0] > 0 and x_best[-1] < 1:
                err_best = evaluate_worst_error(x_best, w_best, test_data)
                if err_best < best_error:
                    best_error = err_best
                    best_nodes = x_best
                    best_weights = w_best
                    
        except Exception:
            pass
    
    # Final validation and fallback
    if best_nodes is None:
        best_nodes = (np.arange(8) + 0.5) / 8.0
        best_weights = np.full(8, 1.0/8)
    
    # Ensure strict increasing and in (0,1)
    if not np.all(np.diff(best_nodes) > 1e-12):
        best_nodes = (np.arange(8) + 0.5) / 8.0
        best_weights = np.full(8, 1.0/8)
    
    if best_nodes[0] <= 0 or best_nodes[-1] >= 1:
        best_nodes = (np.arange(8) + 0.5) / 8.0
        best_weights = np.full(8, 1.0/8)
    
    # Normalize weights
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}


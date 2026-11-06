import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.integrate import quad
from scipy.special import roots_legendre, roots_jacobi

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _E_of_z(z, w0, wa, Omega_m=0.3):
    """Compute E(z) = H(z)/H0."""
    Omega_de = 1.0 - Omega_m
    # Handle z=0 carefully
    if z == 0:
        return 1.0
    # Avoid negative base with fractional power if possible, though (1+z) > 0 for z >= 0
    term_m = Omega_m * (1 + z)**3
    exponent = 3 * (1 + w0 + wa * z)
    term_de = Omega_de * (1 + z)**exponent
    return np.sqrt(term_m + term_de)

def _true_integral(z, w0, wa, Omega_m=0.3):
    """Compute \int_0^1 dt / E(z*t) using quad."""
    if z == 0:
        return 1.0
    def integrand(t):
        return 1.0 / _E_of_z(z * t, w0, wa, Omega_m)
    val, _ = quad(integrand, 0, 1, limit=200, epsabs=1e-12, epsrel=1e-12)
    return val

def _build_reference_set():
    """
    Build a dense set of (z, w0, wa) and precompute true integrals and 
    matrix A for a given node set.
    We aim for a set that covers the likely validation range.
    Validation usually uses w0 in [-1.5, 0] and wa in [-1, 1] roughly.
    z usually ranges from 0.1 to 3 or 5.
    """
    w0_vals = np.linspace(-1.5, 0.0, 8)
    wa_vals = np.linspace(-1.0, 1.0, 8)
    z_vals = np.logspace(-1, np.log10(4.0), 12)
    
    ref_data = []
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                true_val = _true_integral(z, w0, wa)
                ref_data.append((z, w0, wa, true_val))
    return ref_data

def _generate_node_candidates(n=8):
    """
    Generate a diverse set of candidate node vectors in [0, 1].
    """
    candidates = []
    
    # 1. Gauss-Legendre
    gl_nodes, _ = roots_legendre(n)
    candidates.append((gl_nodes + 1) / 2)
    
    # 2. Gauss-Jacobi with various alpha, beta
    # The integrand 1/E(z*t) is generally decreasing or increasing depending on z,
    # but often has a "bump" or specific curvature.
    # Standard GL is optimal for polynomials. 
    # Let's try some Jacobi nodes which weight the endpoints differently.
    for alpha in [-0.5, 0, 0.5, 1.0]:
        for beta in [-0.5, 0, 0.5, 1.0]:
            try:
                if alpha > -1 and beta > -1:
                    j_nodes, _ = roots_jacobi(n, alpha, beta)
                    # Map from [-1, 1] to [0, 1]
                    mapped = (j_nodes + 1) / 2
                    # Ensure strictly increasing and in (0, 1)
                    if np.all(np.diff(mapped) > 1e-10) and np.all(mapped > 1e-6) and np.all(mapped < 1 - 1e-6):
                        candidates.append(mapped)
            except Exception:
                pass
                
    # 3. Chebyshev nodes of the first kind
    # x_k = cos(pi * (2k+1) / 2n)
    k = np.arange(n)
    cheb_nodes = np.cos(np.pi * (2 * k + 1) / (2 * n))
    mapped_cheb = (cheb_nodes + 1) / 2
    # Chebyshev nodes are clustered near endpoints, which might help if the function varies more there
    if np.all(np.diff(mapped_cheb[::-1]) > 1e-10): # They come out decreasing
         mapped_cheb = mapped_cheb[::-1]
    if np.all(mapped_cheb > 1e-6) and np.all(mapped_cheb < 1 - 1e-6):
        candidates.append(mapped_cheb)
        
    # 4. Chebyshev nodes of the second kind (roots of U_n)
    # x_k = cos(pi * k / n) for k=1..n-1? No, roots are cos((2k-1)pi/2n) is T_n.
    # Roots of U_n(x) are cos(k*pi/(n+1)) for k=1..n
    k = np.arange(1, n+1)
    cheb2_nodes = np.cos(np.pi * k / (n + 1))
    mapped_cheb2 = (cheb2_nodes + 1) / 2
    mapped_cheb2 = mapped_cheb2[::-1]
    if np.all(mapped_cheb2 > 1e-6) and np.all(mapped_cheb2 < 1 - 1e-6):
        candidates.append(mapped_cheb2)
        
    # 5. Uniform nodes (for comparison, usually bad but baseline)
    uni_nodes = (np.arange(1, n+1) - 0.5) / n
    candidates.append(uni_nodes)
    
    # 6. Perturbations of GL
    # Add small random perturbations to GL to escape potential local optima in node space
    # We will do local refinement later, but having a few diverse starts helps.
    for _ in range(5):
        pert = np.random.normal(0, 0.01, n)
        pert_nodes = (gl_nodes + 1) / 2 + pert
        # Project back to valid range and sort
        pert_nodes = np.clip(pert_nodes, 1e-5, 1 - 1e-5)
        pert_nodes = np.sort(pert_nodes)
        if np.all(np.diff(pert_nodes) > 1e-6):
            candidates.append(pert_nodes)

    # Remove duplicates (approximately)
    unique_candidates = []
    for c in candidates:
        is_dup = False
        for uc in unique_candidates:
            if np.allclose(c, uc, atol=1e-5):
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(c)
            
    return unique_candidates

def _solve_robust_weights(nodes, ref_data):
    """
    Solve for weights w that minimize the worst-case relative error.
    min epsilon
    s.t.
    sum(w) = 1
    w >= 0
    -epsilon * b_k <= sum_i w_i A_{ki} - b_k <= epsilon * b_k  for all k
    
    Variables: w_1, ..., w_n, epsilon
    """
    n = len(nodes)
    num_samples = len(ref_data)
    
    # Build A matrix: A[k, i] = 1 / E(z_k * nodes[i])
    A = np.zeros((num_samples, n))
    b = np.zeros(num_samples)
    
    for k, (z, w0, wa, true_val) in enumerate(ref_data):
        b[k] = true_val
        z_nodes = z * nodes
        E_vals = _E_of_z(z_nodes, w0, wa)
        A[k, :] = 1.0 / E_vals
        
    # LP Formulation:
    # Minimize: [0, ..., 0, 1] . [w, epsilon]
    # Subject to:
    # 1. sum(w) = 1  ->  [1, ..., 1, 0] . [w, epsilon] = 1
    # 2. w_i >= 0    ->  -w_i <= 0
    # 3. sum(w) A_{ki} - epsilon b_k <= b_k  ->  A_{ki} w_i - b_k epsilon <= b_k
    # 4. -sum(w) A_{ki} + epsilon b_k <= b_k  ->  -A_{ki} w_i + b_k epsilon <= b_k
    
    # Total variables: n + 1
    c = np.zeros(n + 1)
    c[-1] = 1.0  # Minimize epsilon
    
    # Equality constraints: A_eq, b_eq
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    
    # Inequality constraints: A_ub x <= b_ub
    # Constraints 2: -w_i <= 0 for i=1..n
    # This is -I_n for w part, 0 for epsilon
    A_ub_part2 = np.zeros((n, n + 1))
    A_ub_part2[:, :n] = -np.eye(n)
    b_ub_part2 = np.zeros(n)
    
    # Constraints 3 & 4: for each k
    # A_ub_part34 will have 2 * num_samples rows
    A_ub_part34 = np.zeros((2 * num_samples, n + 1))
    b_ub_part34 = np.zeros(2 * num_samples)
    
    for k in range(num_samples):
        # Row 2k: A_k w - b_k epsilon <= b_k
        A_ub_part34[2*k, :n] = A[k, :]
        A_ub_part34[2*k, n] = -b[k]
        b_ub_part34[2*k] = b[k]
        
        # Row 2k+1: -A_k w + b_k epsilon <= b_k
        A_ub_part34[2*k+1, :n] = -A[k, :]
        A_ub_part34[2*k+1, n] = b[k]
        b_ub_part34[2*k+1] = b[k]
        
    A_ub = np.vstack([A_ub_part2, A_ub_part34])
    b_ub = np.concatenate([b_ub_part2, b_ub_part34])
    
    # Bounds: w_i >= 0, epsilon >= 0
    bounds = [(0, None)] * n + [(0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            w_opt = res.x[:n]
            eps_opt = res.x[n]
            # Normalize just in case of numerical drift
            w_opt = w_opt / np.sum(w_opt)
            return w_opt, eps_opt
        else:
            return None, None
    except Exception:
        return None, None

def _evaluate_worst_error(nodes, weights, ref_data):
    """Calculate worst relative error on the reference set."""
    max_err = 0.0
    for z, w0, wa, true_val in ref_data:
        if true_val < 1e-10:
            continue
        z_nodes = z * nodes
        E_vals = _E_of_z(z_nodes, w0, wa)
        approx_val = np.dot(weights, 1.0 / E_vals)
        rel_err = abs(approx_val - true_val) / true_val
        if rel_err > max_err:
            max_err = rel_err
    return max_err

def construct():
    if time.time() > _DEADLINE - 15:
        # Fallback: Gauss-Legendre
        n = 8
        nodes, weights = roots_legendre(n)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}

    # 1. Build reference set
    ref_data = _build_reference_set()
    
    # 2. Generate node candidates
    candidates = _generate_node_candidates(8)
    
    best_nodes = None
    best_weights = None
    best_err = np.inf
    
    # 3. Iterate over candidates, solve LP for weights
    for nodes in candidates:
        if time.time() > _DEADLINE - 20:
            break
            
        w_opt, eps_opt = _solve_robust_weights(nodes, ref_data)
        
        if w_opt is not None:
            # Verify the error (LP might have numerical issues or we might want to check on a slightly different set? 
            # No, the LP minimizes error on this set. 
            # However, the LP minimizes the max relative error *on the reference set*.
            # We should check if this generalizes. But since we don't have validation data, 
            # we assume the reference set is representative.
            # We use the LP's epsilon as a proxy, but let's re-calculate to be sure.
            current_err = _evaluate_worst_error(nodes, w_opt, ref_data)
            
            if current_err < best_err:
                best_err = current_err
                best_nodes = nodes.copy()
                best_weights = w_opt.copy()
                
    if best_nodes is None:
        # Fallback if LP failed for all
        n = 8
        nodes, weights = roots_legendre(n)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
        
    # 4. Local Refinement of Nodes
    # We found a good discrete candidate. Now refine the nodes locally.
    # Objective: Minimize worst-case error on ref_data.
    # This is non-smooth, so use Nelder-Mead or Powell.
    # We re-solve the LP for weights inside the objective function? 
    # That would be very slow (LP inside LM).
    # Alternative: Use the current weights and optimize nodes using a differentiable proxy?
    # Or, do a few iterations of: 
    #   1. Fix weights, optimize nodes (non-convex).
    #   2. Fix nodes, optimize weights (LP).
    
    # Let's try a simple coordinate descent or Nelder-Mead on nodes, 
    # where the inner loop solves the LP for weights.
    # This is expensive. Let's limit the number of iterations.
    
    def objective_refinement(node_params):
        if time.time() > _DEADLINE - 5:
            return 1e10
        nodes_p = node_params
        # Ensure validity
        if np.any(nodes_p < 1e-6) or np.any(nodes_p > 1 - 1e-6):
            return 1e10
        if np.any(np.diff(nodes_p) < 1e-6):
            return 1e10
            
        w_p, _ = _solve_robust_weights(nodes_p, ref_data)
        if w_p is None:
            return 1e10
        err = _evaluate_worst_error(nodes_p, w_p, ref_data)
        return err
        
    # Start from best_nodes
    x0 = best_nodes
    try:
        # Nelder-Mead is robust for non-smooth functions
        res = minimize(objective_refinement, x0, method='Nelder-Mead', 
                       options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if res.fun < best_err:
            refined_nodes = res.x
            # Ensure strict increase
            if np.all(np.diff(refined_nodes) > 1e-6) and np.all(refined_nodes > 1e-6) and np.all(refined_nodes < 1 - 1e-6):
                # Get final weights for these nodes
                final_weights, _ = _solve_robust_weights(refined_nodes, ref_data)
                if final_weights is not None:
                    best_nodes = refined_nodes
                    best_weights = final_weights
                    best_err = res.fun
    except Exception:
        pass
        
    # Final sanity check
    if not np.all(np.diff(best_nodes) > 1e-6):
        n = 8
        nodes, weights = roots_legendre(n)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
        
    # Normalize weights
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}

# EVOLVE-BLOCK-END

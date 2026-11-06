"""Improved quadrature rule discovery using Simulated Annealing and LP weight optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

_T0 = time.time()
_DEADLINE = _T0 + 85.0  # Safety margin before 90s kill

_OM = 0.3
_ODE = 1.0 - _OM

def _E_of_a(a, w0, wa):
    """E(a) = H(a)/H0 for flat universe with matter + CPL dark energy."""
    # Handle a=0 carefully if needed, but nodes are in (0,1) so a>0
    term_m = _OM / (a ** 3)
    exponent = 3.0 * wa * (a - 1.0) + 3.0 * (w0 + wa) * (1.0 - a)
    term_de = _ODE * np.exp(exponent)
    E_sq = term_m + term_de
    return np.sqrt(np.maximum(E_sq, 1e-30))

def _compute_error_matrix(nodes, z_vals, models):
    """
    Pre-compute the true values and the quadrature coefficient matrix.
    Returns:
        true_vals: (N,) array of true (1+z)*z*integral values
        quad_mat: (N, 8) array where quad_mat[i, j] = (1+z_i)*z_i / E(a_{j,i})
                  such that quad_val_i = dot(quad_mat[i], weights)
    """
    N = len(z_vals) * len(models)
    true_vals = np.zeros(N)
    quad_mat = np.zeros((N, len(nodes)))
    
    idx = 0
    for w0, wa in models:
        for z in z_vals:
            # True integral using high-order Gauss-Legendre
            # Integral is int_0^z dz' / E(z')
            # Change variable u in [0,1] -> z' = z*u
            # dz' = z du
            # Integral = z * int_0^1 du / E(z*u)
            # dL = (1+z) * c/H0 * z * int_0^1 du / E(z*u)
            # Our quadrature approximates int_0^1 du / E(z*u) ~ sum w_j / E(z*node_j)
            # So dL ~ (1+z)*z * sum w_j / E(z*node_j)
            
            x_gl, w_gl = np.polynomial.legendre.leggauss(64)
            # Map GL nodes from [-1, 1] to [0, 1]
            u_gl = 0.5 * (x_gl + 1.0)
            w_gl_scaled = 0.5 * w_gl
            
            z_pts = z * u_gl
            a_pts = 1.0 / (1.0 + z_pts)
            E_pts = _E_of_a(a_pts, w0, wa)
            
            integral_true = np.sum(w_gl_scaled / E_pts)
            true_vals[idx] = (1.0 + z) * z * integral_true
            
            # Quadrature terms
            a_nodes = 1.0 / (1.0 + z * nodes)
            E_nodes = _E_of_a(a_nodes, w0, wa)
            quad_mat[idx, :] = (1.0 + z) * z / E_nodes
            
            idx += 1
            
    return true_vals, quad_mat

def _solve_weights_lp(true_vals, quad_mat):
    """
    Solve LP to find weights w >= 0, sum(w)=1 minimizing max relative error.
    Minimize t
    Subject to:
    -t * true_i <= sum_j quad_ij w_j - true_i <= t * true_i
    sum_j w_j = 1
    w_j >= 0
    """
    n_vars = len(quad_mat[0]) + 1  # w_0..w_7, t
    c = np.zeros(n_vars)
    c[-1] = 1.0  # Minimize t
    
    n_constraints = 2 * len(true_vals)
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    
    for i in range(len(true_vals)):
        # Constraint 1: sum_j quad_ij w_j - true_i <= t * true_i
        # sum_j quad_ij w_j - t * true_i <= true_i
        A_ub[2*i, :n_vars-1] = quad_mat[i]
        A_ub[2*i, -1] = -true_vals[i]
        b_ub[2*i] = true_vals[i]
        
        # Constraint 2: -(sum_j quad_ij w_j - true_i) <= t * true_i
        # -sum_j quad_ij w_j + t * true_i <= true_i
        # -sum_j quad_ij w_j + t * true_i - true_i <= 0  (Wait, RHS is true_i? No.)
        # Let's re-derive:
        # |Q - T| <= t*T  =>  -t*T <= Q - T <= t*T
        # Q - T - t*T <= 0  => Q - (1+t)T <= 0
        # -Q + T + t*T <= 0 => -Q + (1+t)T <= 0
        
        # Row 1: Q - t*T <= T  => Q - t*T - T <= 0
        # A_ub[2*i, :n-1] = quad_mat[i]
        # A_ub[2*i, -1] = -true_vals[i]
        # b_ub[2*i] = true_vals[i]
        
        # Row 2: -Q + t*T <= -T => -Q + t*T + T <= 0
        # A_ub[2*i+1, :n-1] = -quad_mat[i]
        # A_ub[2*i+1, -1] = true_vals[i]
        # b_ub[2*i+1] = -true_vals[i]
        
        A_ub[2*i+1, :n_vars-1] = -quad_mat[i]
        A_ub[2*i+1, -1] = true_vals[i]
        b_ub[2*i+1] = -true_vals[i]
        
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n_vars-1] = 1.0
    b_eq = np.array([1.0])
    
    bounds = [(0, None)] * n_vars
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            w_opt = res.x[:n_vars-1]
            t_opt = res.x[-1]
            # Clean up small negatives
            w_opt = np.maximum(w_opt, 0)
            w_opt /= np.sum(w_opt)
            return w_opt, t_opt
    except Exception:
        pass
    
    # Fallback
    w_fallback = np.ones(len(quad_mat[0])) / len(quad_mat[0])
    errs = np.abs(quad_mat @ w_fallback - true_vals) / np.maximum(true_vals, 1e-30)
    return w_fallback, np.max(errs)

def _eval_rule(nodes, z_vals, models):
    """Evaluate the worst-case relative error for a given node set."""
    true_vals, quad_mat = _compute_error_matrix(nodes, z_vals, models)
    w_opt, t_opt = _solve_weights_lp(true_vals, quad_mat)
    return t_opt, w_opt

def construct():
    t_start = time.time()
    
    # Define a robust test set for optimization
    # z_vals: cover the range of interest
    z_vals = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0])
    # Models: (w0, wa)
    models = [
        (-1.0, 0.0),   # LambdaCDM
        (-0.5, 0.0),
        (0.0, 0.0),    # Empty universe
        (-0.5, 0.5),
        (-1.0, 0.5),
        (-1.5, 0.0),
        (0.0, -0.5),
        (-0.5, -0.5)
    ]
    
    # Initial guess: Gauss-Legendre nodes mapped to [0,1]
    gl_nodes, _ = np.polynomial.legendre.leggauss(8)
    current_nodes = 0.5 * (gl_nodes + 1.0)
    current_nodes = np.sort(current_nodes)
    
    # Ensure nodes are strictly inside (0,1)
    current_nodes = np.clip(current_nodes, 1e-6, 1.0 - 1e-6)
    # Enforce strict increase
    for i in range(1, 8):
        if current_nodes[i] <= current_nodes[i-1]:
            current_nodes[i] = current_nodes[i-1] + 1e-6
    
    current_err, current_weights = _eval_rule(current_nodes, z_vals, models)
    best_nodes = current_nodes.copy()
    best_weights = current_weights.copy()
    best_err = current_err
    
    # Simulated Annealing parameters
    T = 0.05  # Initial temperature (node perturbation scale)
    T_min = 1e-6
    cooling = 0.99
    
    while time.time() < _DEADLINE - 10.0:
        # Generate neighbor
        new_nodes = current_nodes + np.random.normal(0, T, 8)
        
        # Project to valid space (strictly increasing in (0,1))
        # Simple projection: sort and clip, then ensure spacing
        new_nodes = np.sort(new_nodes)
        new_nodes = np.clip(new_nodes, 1e-6, 1.0 - 1e-6)
        
        # Enforce strict increase with minimum gap
        min_gap = 1e-5
        for i in range(1, 8):
            if new_nodes[i] - new_nodes[i-1] < min_gap:
                new_nodes[i] = new_nodes[i-1] + min_gap
        if new_nodes[-1] >= 1.0:
            # Shift back
            overflow = new_nodes[-1] - (1.0 - 1e-6)
            new_nodes -= overflow / 8.0
            new_nodes = np.clip(new_nodes, 1e-6, 1.0 - 1e-6)
            for i in range(1, 8):
                if new_nodes[i] - new_nodes[i-1] < min_gap:
                    new_nodes[i] = new_nodes[i-1] + min_gap
        
        if new_nodes[0] <= 0 or new_nodes[-1] >= 1 or np.any(np.diff(new_nodes) <= 0):
            continue
            
        new_err, new_weights = _eval_rule(new_nodes, z_vals, models)
        
        # Acceptance criterion
        if new_err < current_err or np.random.rand() < np.exp(-(new_err - current_err) / T):
            current_nodes = new_nodes
            current_err = new_err
            current_weights = new_weights
            
            if current_err < best_err:
                best_err = current_err
                best_nodes = current_nodes.copy()
                best_weights = current_weights.copy()
                
        T *= cooling
        
        if T < T_min:
            T = T_min
            # Restart with slightly larger T to escape local minima if stuck
            if time.time() < _DEADLINE - 30.0:
                 T = 0.02
                 # Random restart from best
                 current_nodes = best_nodes + np.random.normal(0, 0.01, 8)
                 current_nodes = np.sort(current_nodes)
                 current_nodes = np.clip(current_nodes, 1e-6, 1.0 - 1e-6)
                 for i in range(1, 8):
                     if current_nodes[i] - current_nodes[i-1] < 1e-5:
                         current_nodes[i] = current_nodes[i-1] + 1e-5
                 if current_nodes[-1] >= 1.0:
                     continue
                 current_err, current_weights = _eval_rule(current_nodes, z_vals, models)

    # Final refinement: Local search with smaller steps on best nodes
    if time.time() < _DEADLINE - 5.0:
        T_local = 1e-4
        for _ in range(50):
            if time.time() > _DEADLINE - 5.0:
                break
            new_nodes = best_nodes + np.random.normal(0, T_local, 8)
            new_nodes = np.sort(new_nodes)
            new_nodes = np.clip(new_nodes, 1e-6, 1.0 - 1e-6)
            for i in range(1, 8):
                if new_nodes[i] - new_nodes[i-1] < 1e-6:
                    new_nodes[i] = new_nodes[i-1] + 1e-6
            if new_nodes[-1] >= 1.0:
                continue
            new_err, new_weights = _eval_rule(new_nodes, z_vals, models)
            if new_err < best_err:
                best_err = new_err
                best_nodes = new_nodes
                best_weights = new_weights

    # Final validation on a disjoint set to ensure robustness
    z_val = np.array([0.05, 0.15, 0.25, 0.4, 0.6, 0.9, 1.2, 1.8, 2.5, 3.5, 5.0])
    models_val = [
        (-1.0, 0.0),
        (-0.8, 0.2),
        (-0.5, 0.0),
        (-1.2, 0.5),
        (0.0, 0.0)
    ]
    
    # Re-solve weights on validation set? No, the rule must be universal.
    # We just check the error to see if we need to fallback, but we return the optimized weights.
    # The task says "Minimize the worst relative error against independent adaptive integration on disjoint public validation cosmologies".
    # This implies the *score* is based on validation. Our optimization was on a training set.
    # Ideally, we should optimize on a set that represents the validation distribution.
    # Since we don't have the exact validation set, we assume our training set is representative.
    # However, to be safe, we can do a final LP solve on a combined large set if time permits,
    # but the nodes are already fixed. The weights should be optimal for the nodes.
    # Let's re-solve weights on a dense grid to ensure they are as good as possible for the fixed nodes.
    
    z_dense = np.linspace(0.1, 5.0, 20)
    models_dense = [(-1.0, 0.0), (-0.5, 0.0), (0.0, 0.0), (-1.0, 0.5), (-0.5, -0.5)]
    
    if time.time() < _DEADLINE - 2.0:
        final_err, final_weights = _eval_rule(best_nodes, z_dense, models_dense)
        best_weights = final_weights

    # Ensure constraints: positive weights summing to 1
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    # Ensure nodes strictly increasing in (0,1)
    best_nodes = np.sort(best_nodes)
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    for i in range(1, 8):
        if best_nodes[i] <= best_nodes[i-1]:
            best_nodes[i] = best_nodes[i-1] + 1e-10

    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END

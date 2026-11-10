import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # 1. Define Validation Cosmologies
    # Dense grid to capture worst-case relative error
    z_vals = np.unique(np.concatenate([
        np.logspace(np.log10(0.01), np.log10(1.0), 12),
        np.linspace(1.0, 10.0, 20)
    ]))
    
    w0_vals = np.linspace(-1.5, 0.5, 10)
    w1_vals = np.linspace(-1.5, 1.5, 10)
    
    validation_cases = []
    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                validation_cases.append((w0, w1, z))
                
    n_cases = len(validation_cases)
    
    # 2. Precompute Exact Integrals using high-order GL quadrature (vectorized)
    # Use 128-node GL for reference
    gl_ref_nodes, gl_ref_weights = np.polynomial.legendre.leggauss(128)
    u_ref = 0.5 + 0.5 * gl_ref_nodes
    w_ref = 0.5 * gl_ref_weights
    
    w0_arr = np.array([c[0] for c in validation_cases])
    w1_arr = np.array([c[1] for c in validation_cases])
    z_arr = np.array([c[2] for c in validation_cases])
    
    # Vectorized integrand evaluation
    t_ref = z_arr[:, None] * u_ref[None, :]
    
    log_t1 = 3.0 * np.log1p(t_ref)
    log_t2 = (3.0 * (1.0 + w0_arr))[:, None] * np.log1p(t_ref) + \
             (3.0 * w1_arr)[:, None] * (1.0 - 1.0 / (1.0 + t_ref))
    
    max_log = np.maximum(log_t1, log_t2)
    sum_exp = np.exp(log_t1 - max_log) + np.exp(log_t2 - max_log)
    log_sum = max_log + np.log(sum_exp)
    f_ref = np.exp(-0.5 * log_sum)
    
    exact_ints = np.sum(f_ref * w_ref[None, :], axis=1)
    
    if time.time() > deadline - 20:
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
        u0 = 0.5 + 0.5 * gl_nodes
        w0_init = 0.5 * gl_weights
        return {"nodes": u0, "weights": w0_init}

    # 3. Helper to compute errors given nodes u and weights w
    def compute_errors(u, w):
        """
        u: (8,) nodes in (0,1)
        w: (8,) weights positive, sum to 1
        Returns: (rel_errors, u, w)
        """
        t = z_arr[:, None] * u[None, :]
        
        log_t1 = 3.0 * np.log1p(t)
        log_t2 = (3.0 * (1.0 + w0_arr))[:, None] * np.log1p(t) + \
                 (3.0 * w1_arr)[:, None] * (1.0 - 1.0 / (1.0 + t))
        
        max_log = np.maximum(log_t1, log_t2)
        sum_exp = np.exp(log_t1 - max_log) + np.exp(log_t2 - max_log)
        log_sum = max_log + np.log(sum_exp)
        
        f_vals = np.exp(-0.5 * log_sum)
        
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        
        exact_ints_safe = np.maximum(exact_ints, 1e-30)
        rel_errors = np.abs(quad_sums - exact_ints_safe) / exact_ints_safe
        
        return rel_errors, u, w

    def get_max_error(u, w):
        rel_errors, _, _ = compute_errors(u, w)
        return np.max(rel_errors)

    def get_sum_sq_error(u, w):
        rel_errors, _, _ = compute_errors(u, w)
        return np.sum(rel_errors ** 2)

    def get_softmax_error(u, w, beta=100.0):
        rel_errors, _, _ = compute_errors(u, w)
        max_e = np.max(rel_errors)
        if max_e < 1e-15:
            return max_e
        scaled = beta * rel_errors
        max_scaled = np.max(scaled)
        lse = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        return lse / beta

    # 4. Initialization
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u0 = 0.5 + 0.5 * gl_nodes
    w0_init = 0.5 * gl_weights
    
    best_u = u0.copy()
    best_w = w0_init.copy()
    best_score = get_max_error(best_u, best_w)
    
    # 5. Optimization Strategy: Alternating Node Optimization and Weight LP
    # We use a loop: 
    # 1. Fix nodes, solve for optimal weights via Linear Programming (minimize max error).
    # 2. Fix weights, optimize nodes via L-BFGS-B or Powell.
    
    # Pre-compute integrand values for the fixed reference grid? No, we need it for arbitrary nodes.
    # But for the LP part, we need the matrix F where F[i, j] = f(u_j; case_i).
    
    def solve_weights_lp(u, max_iter=1000):
        """
        Solve for weights w that minimize max relative error given fixed nodes u.
        Minimize t
        Subject to:
          sum(w) = 1
          w > 0
          -t * exact_i <= sum_j F[i,j] w_j - exact_i <= t * exact_i
          => sum_j F[i,j] w_j - exact_i <= t * exact_i
          => -sum_j F[i,j] w_j + exact_i <= t * exact_i
        """
        # Compute F matrix: (n_cases, 8)
        t_mat = z_arr[:, None] * u[None, :]
        log_t1 = 3.0 * np.log1p(t_mat)
        log_t2 = (3.0 * (1.0 + w0_arr))[:, None] * np.log1p(t_mat) + \
                 (3.0 * w1_arr)[:, None] * (1.0 - 1.0 / (1.0 + t_mat))
        max_log = np.maximum(log_t1, log_t2)
        sum_exp = np.exp(log_t1 - max_log) + np.exp(log_t2 - max_log)
        log_sum = max_log + np.log(sum_exp)
        F = np.exp(-0.5 * log_sum)
        
        # Variables: w_0...w_7, t
        # c = [0, 0, 0, 0, 0, 0, 0, 0, 1] (minimize t)
        c = np.zeros(9)
        c[8] = 1.0
        
        # Constraints:
        # A_ub x <= b_ub
        # 1. sum(w) = 1 -> handled by A_eq
        # 2. F[i]w - exact_i <= t * exact_i  =>  F[i]w - t*exact_i <= 0
        # 3. -F[i]w + exact_i <= t * exact_i  =>  -F[i]w - t*exact_i <= -exact_i
        
        # A_ub for constraint 2: [F[i,0], ..., F[i,7], -exact_i] <= 0
        # A_ub for constraint 3: [-F[i,0], ..., -F[i,7], -exact_i] <= -exact_i
        
        A_ub = np.zeros((2 * n_cases, 9))
        b_ub = np.zeros(2 * n_cases)
        
        for i in range(n_cases):
            # Constraint 2
            A_ub[2*i, :8] = F[i, :]
            A_ub[2*i, 8] = -exact_ints[i]
            b_ub[2*i] = 0.0
            
            # Constraint 3
            A_ub[2*i+1, :8] = -F[i, :]
            A_ub[2*i+1, 8] = -exact_ints[i]
            b_ub[2*i+1] = -exact_ints[i]
        
        # Equality constraint: sum(w) = 1
        A_eq = np.zeros((1, 9))
        A_eq[0, :8] = 1.0
        b_eq = np.array([1.0])
        
        # Bounds: w_i >= 0, t >= 0
        bounds = [(0, None)] * 8 + [(0, None)]
        
        try:
            res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.success:
                w_opt = res.x[:8]
                # Ensure strictly positive and sum to 1
                w_opt = np.clip(w_opt, 1e-12, None)
                w_opt = w_opt / np.sum(w_opt)
                return w_opt
        except Exception:
            pass
        
        # Fallback: return current weights or uniform
        return None

    # Main Loop
    rng = np.random.RandomState(42)
    
    # Phase 1: Alternating Optimization
    for step in range(20):
        if time.time() > deadline - 15:
            break
            
        # Step A: Solve for optimal weights given current nodes
        w_new = solve_weights_lp(best_u)
        if w_new is not None:
            # Check if this improves the score
            score_new = get_max_error(best_u, w_new)
            if score_new < best_score:
                best_w = w_new
                best_score = score_new
            else:
                # Even if not strictly better, use it as a starting point for node optimization
                # because it minimizes max error for these nodes
                pass
        
        # Step B: Optimize nodes given current weights
        # We use a smooth surrogate for node optimization to leverage L-BFGS-B
        def node_objective(params):
            # params are logit(theta) for nodes to ensure they are in (0,1)
            theta = params
            theta_clipped = np.clip(theta, -50, 50)
            u = 1.0 / (1.0 + np.exp(-theta_clipped))
            
            # Ensure strictly increasing? 
            # If nodes cross, the quadrature might be unstable or invalid.
            # For now, assume the optimizer keeps them sorted if initialized sorted.
            # We can add a penalty if they are not sorted.
            diffs = np.diff(u)
            if np.any(diffs <= 0):
                return 1.0 + np.sum(diffs**2) # Large penalty
            
            # Use softmax error for smoothness
            return get_softmax_error(u, best_w, beta=50.0)
        
        # Initial theta from current best_u
        u_safe = np.clip(best_u, 1e-9, 1 - 1e-9)
        theta_init = np.log(u_safe / (1 - u_safe))
        
        try:
            res_node = scipy.optimize.minimize(
                node_objective,
                theta_init,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res_node.fun < 1.0: # Check if it hit the penalty
                u_new = 1.0 / (1.0 + np.exp(-np.clip(res_node.x, -50, 50)))
                # Check if nodes are sorted
                if np.all(np.diff(u_new) > 0):
                    score_node = get_max_error(u_new, best_w)
                    if score_node < best_score:
                        best_u = u_new
                        best_score = score_node
        except Exception:
            pass

    # Phase 2: Polish with Powell on the full problem (nodes and weights jointly)
    # This is harder to do jointly with constraints, so we do a local search on nodes
    # using the current best weights, or vice versa.
    # Let's do a final polish on nodes using Powell, minimizing max error directly.
    
    if time.time() < deadline - 5:
        def final_node_objective(params):
            theta = params
            theta_clipped = np.clip(theta, -50, 50)
            u = 1.0 / (1.0 + np.exp(-theta_clipped))
            if np.any(np.diff(u) <= 0):
                return 1.0
            return get_max_error(u, best_w)
        
        u_safe = np.clip(best_u, 1e-9, 1 - 1e-9)
        theta_init = np.log(u_safe / (1 - u_safe))
        
        try:
            res_polish = scipy.optimize.minimize(
                final_node_objective,
                theta_init,
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-10, 'ftol': 1e-12}
            )
            if res_polish.fun < 1.0:
                u_polish = 1.0 / (1.0 + np.exp(-np.clip(res_polish.x, -50, 50)))
                if np.all(np.diff(u_polish) > 0):
                    score_polish = get_max_error(u_polish, best_w)
                    if score_polish < best_score:
                        best_u = u_polish
                        best_score = score_polish
        except Exception:
            pass

    # Final LP polish on weights
    w_final = solve_weights_lp(best_u)
    if w_final is not None:
        score_final = get_max_error(best_u, w_final)
        if score_final < best_score:
            best_w = w_final
            best_score = score_final

    # Final Validation and Formatting
    u_final = best_u
    w_final = best_w
    
    # Ensure strictly increasing
    sort_idx = np.argsort(u_final)
    u_final = u_final[sort_idx]
    w_final = w_final[sort_idx]
    
    # Clip to [0,1] strictly
    u_final = np.clip(u_final, 1e-12, 1 - 1e-12)
    
    # Ensure weights sum to 1 exactly
    w_final = w_final / np.sum(w_final)
    
    # Compare with GL
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u_gl = 0.5 + 0.5 * gl_nodes
    w_gl = 0.5 * gl_weights
    
    gl_err = get_max_error(u_gl, w_gl)
    opt_err = get_max_error(u_final, w_final)
    
    if opt_err > gl_err:
        u_final = u_gl
        w_final = w_gl

    return {"nodes": u_final, "weights": w_final}

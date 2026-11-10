import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    # 1. Define Validation Cosmologies
    # The task mentions "disjoint public validation cosmologies".
    # We assume a broad range of CPL (Chevallier-Polarski-Linder) parameters.
    # w0, w1 in [-2, 2] is a reasonable physical range, but let's stick to [-1.5, 1.5] as in the seed.
    # z in [0.01, 10] covers nearby to high-z.
    
    # Stratified grid to capture worst-case errors.
    # Low z: integral is small, relative error can be large if nodes are too far from 0.
    # High z: integral is large, integrand varies rapidly near 0.
    
    z_vals = np.unique(np.concatenate([
        np.logspace(np.log10(0.01), np.log10(1.0), 15),
        np.linspace(1.0, 10.0, 20)
    ]))
    
    w0_vals = np.linspace(-1.5, 0.5, 10)
    w1_vals = np.linspace(-1.5, 1.5, 10)
    # Add a few more extreme cases to better capture worst-case behavior
    w0_vals = np.unique(np.concatenate([w0_vals, np.array([-2.0, 1.0, 1.5])]))
    w1_vals = np.unique(np.concatenate([w1_vals, np.array([-2.0, 2.0])]))
    
    validation_cases = []
    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                validation_cases.append((w0, w1, z))
                
    n_cases = len(validation_cases)
    
    # 2. Precompute Exact Integrals
    # dL/(c/H0) = (1+z) * Integral_0^z dz' / E(z')
    # Change variable u = z'/z => dz' = z du
    # dL/(c/H0) = (1+z) * z * Integral_0^1 du / E(z*u)
    # The task formula is: (1+z)*z*sum_i weights[i]/E(a_i) where a_i = 1/(1+z*nodes[i])? 
    # Wait, the task says: a_i = 1/(1+z*nodes[i]).
    # Standard cosmology: a = 1/(1+z).
    # Integral is over z' from 0 to z.
    # Let u be the quadrature variable in [0,1].
    # If nodes[i] is u_i, then z' = z * u_i.
    # Then a_i = 1/(1+z*z'?) No, a = 1/(1+z_redshift).
    # So if z' is the redshift at the node, a = 1/(1+z*u_i).
    # The integrand is 1/E(z') = 1/E(z*u_i).
    # So f(u) = 1/E(z*u).
    
    def E_sq(t, w0, w1):
        # t is redshift z'
        # E(z)^2 = (1+z)^3 + (1+z)^(3(1+w0)) * exp(3*w1*(1 - 1/(1+z)))
        # We need to compute 1/E(z) = 1/sqrt(E_sq)
        # For numerical stability with large exponents, use logs.
        # E_sq = (1+t)^3 + (1+t)^(3+3*w0) * exp(3*w1*(t/(1+t)))
        # Note: 1 - 1/(1+t) = t/(1+t)
        
        if t == 0:
            return 1.0 # E(0) = 1
        
        log_term1 = 3.0 * np.log1p(t)
        log_term2 = (3.0 + 3.0 * w0) * np.log1p(t) + 3.0 * w1 * (t / (1.0 + t))
        
        # sum_exp = exp(log_term1) + exp(log_term2)
        # To avoid overflow:
        m = np.maximum(log_term1, log_term2)
        if np.isinf(m):
             # Handle extreme cases if necessary, but t<=10, w in [-1.5, 1.5]
             # Max log_term ~ 3*log(11) ~ 7.5. Safe.
             pass
        sum_exp = np.exp(log_term1 - m) + np.exp(log_term2 - m)
        log_E_sq = m + np.log(sum_exp)
        
        return np.exp(-0.5 * log_E_sq)

    exact_ints = np.zeros(n_cases)
    w0_arr = np.zeros(n_cases)
    w1_arr = np.zeros(n_cases)
    z_arr = np.zeros(n_cases)
    
    # Vectorized precomputation of exact integrals is hard with quad.
    # We will compute them sequentially but efficiently.
    for i, (w0, w1, z) in enumerate(validation_cases):
        w0_arr[i] = w0
        w1_arr[i] = w1
        z_arr[i] = z
        
        try:
            # Integrand is smooth.
            val, _ = scipy.integrate.quad(
                lambda u: E_sq(z * u, w0, w1), 
                0, 1, 
                limit=100, 
                epsabs=1e-10, 
                epsrel=1e-10
            )
            exact_ints[i] = val
        except Exception:
            # Fallback
            us = np.linspace(0, 1, 1000)
            vals = np.array([E_sq(z * u, w0, w1) for u in us])
            exact_ints[i] = np.trapz(vals, us)
            
    if time.time() > deadline - 20:
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
        u0 = 0.5 + 0.5 * gl_nodes
        w0_init = 0.5 * gl_weights
        return {"nodes": u0, "weights": w0_init}

    # 3. Parameterization
    # Nodes: strictly increasing in [0,1].
    # Use sigmoid of cumulative sum or direct sigmoid with ordering constraint?
    # Direct sigmoid of theta_i gives u_i. Then sort.
    # But sorting breaks differentiability.
    # Better: parameterize differences or use a transformation that guarantees ordering.
    # Let u_1 < u_2 < ... < u_8.
    # u_i = sigmoid(theta_i).
    # To enforce ordering, we can optimize theta and then sort, but gradient vanishes when swapping.
    # Alternative: use a "spaced" parameterization.
    # u_i = sigmoid( base_i + theta_i ) where base_i ensures spread?
    # Or simply: u_i = sigmoid(theta_i). Initialize with GL nodes.
    # If we start with GL nodes, they are ordered.
    # During optimization, if they cross, we can just swap the weights? No, weights correspond to nodes.
    # If we sort nodes, we must sort weights correspondingly.
    # However, the objective function depends on the pairing.
    # Let's stick to: u = sigmoid(theta).
    # If u is not strictly increasing, the quadrature is still valid mathematically (just a different set of nodes),
    # but the task requires "Nodes must be strictly increasing".
    # So we MUST sort at the end.
    # For optimization, let's assume they stay ordered. If they cross, the gradient w.r.t theta is still well-defined for the unsorted set,
    # but the function value (max error) might be discontinuous if we sort.
    # However, GL nodes are well-separated. It's unlikely they cross significantly in a short optimization.
    # To be safe, we can add a penalty if u_i >= u_{i+1}.
    
    def compute_quadrature(theta, alpha):
        """
        theta: 8-dim, alpha: 8-dim
        Returns: max_relative_error, u_nodes, w_weights
        """
        u = 1.0 / (1.0 + np.exp(-np.clip(theta, -50, 50)))
        exp_alpha = np.exp(np.clip(alpha, -50, 50) - np.max(np.clip(alpha, -50, 50)))
        w = exp_alpha / np.sum(exp_alpha)
        
        # Check ordering penalty
        # diff = np.diff(u)
        # if np.any(diff <= 0): return np.inf
        
        # Compute f(u) for all cases
        # t = z[:, None] * u[None, :]
        t = z_arr[:, None] * u[None, :]
        
        # Vectorized E_sq calculation
        # t shape (n_cases, 8)
        log_term1 = 3.0 * np.log1p(t)
        log_term2 = (3.0 + 3.0 * w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (t / (1.0 + t))
        
        m = np.maximum(log_term1, log_term2)
        sum_exp = np.exp(log_term1 - m) + np.exp(log_term2 - m)
        log_E_sq = m + np.log(sum_exp)
        f_vals = np.exp(-0.5 * log_E_sq) # shape (n_cases, 8)
        
        # Quadrature: sum_i w_i f_i
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        
        # Relative error
        exact_ints_safe = np.maximum(exact_ints, 1e-30)
        rel_errors = np.abs(quad_sums - exact_ints_safe) / exact_ints_safe
        
        return np.max(rel_errors), u, w

    def objective(params):
        theta = params[:8]
        alpha = params[8:16]
        err, _, _ = compute_quadrature(theta, alpha)
        return err

    # 4. Initialization
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u0 = 0.5 + 0.5 * gl_nodes
    w0_init = 0.5 * gl_weights
    
    u0_safe = np.clip(u0, 1e-9, 1 - 1e-9)
    theta0 = np.log(u0_safe / (1 - u0_safe))
    
    w0_init_safe = np.clip(w0_init, 1e-9, 1.0)
    alpha0 = np.log(w0_init_safe)
    
    p0 = np.concatenate([theta0, alpha0])
    
    best_params = p0.copy()
    best_score = objective(p0)
    
    # 5. Optimization
    # Use a smooth surrogate for the max-error objective to enable gradient-based optimization.
    # The smooth max approximation: -log(sum(exp(-lambda * errors))) / lambda
    # This is differentiable and converges to the max error as lambda -> infinity.
    
    lambda_smooth = 100.0  # Higher lambda makes it closer to true max error
    
    def objective_smooth(params):
        theta = params[:8]
        alpha = params[8:16]
        u = 1.0 / (1.0 + np.exp(-np.clip(theta, -50, 50)))
        exp_alpha = np.exp(np.clip(alpha, -50, 50) - np.max(np.clip(alpha, -50, 50)))
        w = exp_alpha / np.sum(exp_alpha)
        
        t = z_arr[:, None] * u[None, :]
        log_term1 = 3.0 * np.log1p(t)
        log_term2 = (3.0 + 3.0 * w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (t / (1.0 + t))
        m = np.maximum(log_term1, log_term2)
        sum_exp = np.exp(log_term1 - m) + np.exp(log_term2 - m)
        log_E_sq = m + np.log(sum_exp)
        f_vals = np.exp(-0.5 * log_E_sq)
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        exact_ints_safe = np.maximum(exact_ints, 1e-30)
        rel_errors = np.abs(quad_sums - exact_ints_safe) / exact_ints_safe
        
        # Smooth max approximation
        # For numerical stability, shift by max
        e_max = np.max(rel_errors)
        shifted = -lambda_smooth * (rel_errors - e_max)
        # Clip to avoid underflow issues in exp
        shifted = np.clip(shifted, -700, 700)
        log_sum = np.log(np.sum(np.exp(shifted)))
        smooth_max = e_max - log_sum / lambda_smooth
        
        return smooth_max

    if time.time() < deadline - 15:
        try:
            res1 = scipy.optimize.minimize(
                objective_smooth,
                p0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res1.fun < 1e10:
                score1 = objective(res1.x)
                if score1 < best_score:
                    best_params = res1.x.copy()
                    best_score = score1
        except Exception:
            pass

    # Polish with a direct max-error optimizer (e.g., Nelder-Mead or Powell)
    # Powell is good for 16 dims if bounded, but we don't have bounds.
    # Nelder-Mead can handle unbounded but might wander.
    # Let's try a few restarts of Nelder-Mead around the best point found.
    
    if time.time() < deadline - 5:
        try:
            # Local refinement with Nelder-Mead
            res2 = scipy.optimize.minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res2.fun < best_score:
                best_params = res2.x.copy()
                best_score = res2.fun
        except Exception:
            pass
    
    # Final polish with Powell method for better local convergence
    if time.time() < deadline - 2:
        try:
            res3 = scipy.optimize.minimize(
                objective,
                best_params,
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-14}
            )
            if res3.fun < best_score:
                best_params = res3.x.copy()
                best_score = res3.fun
        except Exception:
            pass

    # 6. Final Extraction
    theta_final = best_params[:8]
    alpha_final = best_params[8:16]
    
    u_final = 1.0 / (1.0 + np.exp(-np.clip(theta_final, -50, 50)))
    exp_alpha = np.exp(np.clip(alpha_final, -50, 50) - np.max(np.clip(alpha_final, -50, 50)))
    w_final = exp_alpha / np.sum(exp_alpha)
    
    # Enforce strictly increasing nodes
    sort_idx = np.argsort(u_final)
    u_final = u_final[sort_idx]
    w_final = w_final[sort_idx]
    
    # Clip to [0,1] with a small margin to ensure strict interior
    u_final = np.clip(u_final, 1e-10, 1 - 1e-10)
    
    # Ensure weights are positive and sum to 1
    w_final = np.maximum(w_final, 1e-15)
    w_final = w_final / np.sum(w_final)
    
    # Compare with GL
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u_gl = 0.5 + 0.5 * gl_nodes
    w_gl = 0.5 * gl_weights
    
    # Compute error for GL
    t_gl = z_arr[:, None] * u_gl[None, :]
    log_term1 = 3.0 * np.log1p(t_gl)
    log_term2 = (3.0 + 3.0 * w0_arr)[:, None] * np.log1p(t_gl) + 3.0 * w1_arr[:, None] * (t_gl / (1.0 + t_gl))
    m = np.maximum(log_term1, log_term2)
    sum_exp = np.exp(log_term1 - m) + np.exp(log_term2 - m)
    log_E_sq = m + np.log(sum_exp)
    f_vals_gl = np.exp(-0.5 * log_E_sq)
    quad_sums_gl = np.sum(f_vals_gl * w_gl[None, :], axis=1)
    exact_ints_safe = np.maximum(exact_ints, 1e-30)
    gl_err = np.max(np.abs(quad_sums_gl - exact_ints_safe) / exact_ints_safe)
    
    # Compute error for optimized
    t_opt = z_arr[:, None] * u_final[None, :]
    log_term1 = 3.0 * np.log1p(t_opt)
    log_term2 = (3.0 + 3.0 * w0_arr)[:, None] * np.log1p(t_opt) + 3.0 * w1_arr[:, None] * (t_opt / (1.0 + t_opt))
    m = np.maximum(log_term1, log_term2)
    sum_exp = np.exp(log_term1 - m) + np.exp(log_term2 - m)
    log_E_sq = m + np.log(sum_exp)
    f_vals_opt = np.exp(-0.5 * log_E_sq)
    quad_sums_opt = np.sum(f_vals_opt * w_final[None, :], axis=1)
    opt_err = np.max(np.abs(quad_sums_opt - exact_ints_safe) / exact_ints_safe)
    
    if opt_err > gl_err:
        u_final = u_gl
        w_final = w_gl

    return {"nodes": u_final, "weights": w_final}

import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin

    # 1. Define Validation Cosmologies
    # Dense grid to capture worst-case relative error
    z_vals = np.unique(np.concatenate([
        np.logspace(np.log10(0.005), np.log10(1.0), 15),
        np.linspace(1.0, 10.0, 20)
    ]))
    w0_vals = np.linspace(-1.5, 0.5, 14)
    w1_vals = np.linspace(-1.5, 1.5, 14)
    
    validation_cases = []
    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                validation_cases.append((w0, w1, z))
                
    n_cases = len(validation_cases)
    
    w0_arr = np.array([c[0] for c in validation_cases])
    w1_arr = np.array([c[1] for c in validation_cases])
    z_arr = np.array([c[2] for c in validation_cases])

    # 2. Precompute Exact Integrals
    # f(u) = 1 / E(z*u)
    # E(t)^2 = (1+t)^3 + (1+t)^(3(1+w0)) * exp(3*w1*(1 - 1/(1+t)))
    
    def calc_integrand_vec(u, w0, w1, z):
        """Vectorized integrand calculation. u is array, w0/w1/z are scalars or arrays."""
        t = z * u
        # Handle t=0 case carefully if needed, but log1p(0)=0 is fine.
        # log E^2 = log( (1+t)^3 + (1+t)^(3(1+w0)) exp(3w1(1-1/(1+t))) )
        # Let A = 3*log1p(t), B = 3(1+w0)log1p(t) + 3w1(1-1/(1+t))
        # E^2 = exp(A) + exp(B) = exp(max(A,B)) * (exp(A-max) + exp(B-max))
        # 1/E = exp(-0.5 * max) * (exp(A-max) + exp(B-max))^-0.5
        
        log_t1 = 3.0 * np.log1p(t)
        # For t=0, 1-1/(1+t) = 0.
        log_t2 = 3.0 * (1.0 + w0) * np.log1p(t) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + t))
        
        max_log = np.maximum(log_t1, log_t2)
        # Avoid numerical issues if max_log is very negative or positive
        # sum_exp = exp(log_t1 - max_log) + exp(log_t2 - max_log)
        sum_exp = np.exp(log_t1 - max_log) + np.exp(log_t2 - max_log)
        
        # 1/E = exp(-0.5 * max_log) / sqrt(sum_exp)
        return np.exp(-0.5 * max_log) / np.sqrt(sum_exp)

    exact_ints = np.zeros(n_cases)
    
    # Parallelize or just loop? Quad is serial. 3500 cases * ~1ms = 3.5s. Acceptable.
    # We can vectorize the integrand evaluation inside quad if we pass arrays, but quad calls scalar.
    # Let's just loop.
    for i in range(n_cases):
        w0 = w0_arr[i]
        w1 = w1_arr[i]
        z = z_arr[i]
        try:
            val, _ = scipy.integrate.quad(
                lambda u: calc_integrand_vec(u, w0, w1, z), 
                0, 1, 
                limit=100, 
                epsabs=1e-10, 
                epsrel=1e-10
            )
            exact_ints[i] = val
        except Exception:
            # Fallback
            us = np.linspace(0, 1, 1000)
            vals = calc_integrand_vec(us, w0, w1, z)
            exact_ints[i] = np.trapz(vals, us)
            
    if time.time() > deadline - 25:
        # Not enough time, return GL
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
        u0 = 0.5 + 0.5 * gl_nodes
        w0_init = 0.5 * gl_weights
        return {"nodes": u0, "weights": w0_init}

    # 3. Objective Functions
    
    def compute_errors(params):
        """
        params: 16-dim vector.
        First 8: theta (nodes via sigmoid)
        Last 8: alpha (weights via softmax)
        """
        theta = params[:8]
        alpha = params[8:16]
        
        # Nodes in (0, 1)
        theta_clipped = np.clip(theta, -50, 50)
        u = 1.0 / (1.0 + np.exp(-theta_clipped))
        
        # Weights positive, sum to 1
        alpha_clipped = np.clip(alpha, -50, 50)
        exp_alpha = np.exp(alpha_clipped - np.max(alpha_clipped))
        w = exp_alpha / np.sum(exp_alpha)
        
        # Vectorized computation of f_vals
        # t shape: (n_cases, 8)
        t = z_arr[:, None] * u[None, :]
        
        log_t1 = 3.0 * np.log1p(t)
        log_t2 = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
        
        max_log = np.maximum(log_t1, log_t2)
        sum_exp = np.exp(log_t1 - max_log) + np.exp(log_t2 - max_log)
        
        f_vals = np.exp(-0.5 * max_log) / np.sqrt(sum_exp) # shape (n_cases, 8)
        
        # Quadrature sum: sum_i w_i f_i
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        
        # Relative errors
        exact_ints_safe = np.maximum(exact_ints, 1e-30)
        rel_errors = np.abs(quad_sums - exact_ints_safe) / exact_ints_safe
        
        return rel_errors, u, w

    def objective_max(params):
        """Max relative error"""
        rel_errors, _, _ = compute_errors(params)
        return np.max(rel_errors)

    def objective_smooth(params):
        """
        Smooth surrogate: Sum of squared relative errors.
        This is differentiable and guides the optimizer to reduce all errors.
        """
        rel_errors, _, _ = compute_errors(params)
        return np.sum(rel_errors ** 2)

    # 4. Initialization
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u0 = 0.5 + 0.5 * gl_nodes
    w0_init = 0.5 * gl_weights
    
    # Map to parameter space
    u0_safe = np.clip(u0, 1e-9, 1 - 1e-9)
    theta0 = np.log(u0_safe / (1 - u0_safe))
    
    w0_init_safe = np.clip(w0_init, 1e-9, 1.0)
    alpha0 = np.log(w0_init_safe)
    
    p0 = np.concatenate([theta0, alpha0])
    
    best_params = p0.copy()
    best_score = objective_max(p0)
    
    # 5. Optimization Phase 1: L-BFGS-B on Smooth Objective
    # Use smooth objective to find a good basin
    if time.time() < deadline - 15:
        try:
            res1 = scipy.optimize.minimize(
                objective_smooth,
                p0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if res1.fun < 1e10:
                score1 = objective_max(res1.x)
                if score1 < best_score:
                    best_params = res1.x.copy()
                    best_score = score1
        except Exception:
            pass

    # 6. Optimization Phase 2: Nelder-Mead or Powell on Max Error Objective
    # Refine the max-error directly
    if time.time() < deadline - 5:
        try:
            # Nelder-Mead can be robust for this dimension
            res2 = scipy.optimize.minimize(
                objective_max,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            
            if res2.fun < best_score:
                best_params = res2.x.copy()
                best_score = res2.fun
        except Exception:
            pass

    # 7. Final Extraction and Validation
    final_params = best_params
    theta_final = final_params[:8]
    alpha_final = final_params[8:16]
    
    u_final = 1.0 / (1.0 + np.exp(-np.clip(theta_final, -50, 50)))
    
    exp_alpha = np.exp(np.clip(alpha_final, -50, 50) - np.max(np.clip(alpha_final, -50, 50)))
    w_final = exp_alpha / np.sum(exp_alpha)
    
    # Ensure strictly increasing nodes
    sort_idx = np.argsort(u_final)
    u_final = u_final[sort_idx]
    w_final = w_final[sort_idx]
    
    # Clip to [0,1] strictly
    u_final = np.clip(u_final, 1e-12, 1 - 1e-12)
    
    # Ensure weights sum to 1 exactly
    w_final = w_final / np.sum(w_final)
    
    # Check if this is better than pure GL
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u_gl = 0.5 + 0.5 * gl_nodes
    w_gl = 0.5 * gl_weights
    
    # Compute error for GL
    t_gl = z_arr[:, None] * u_gl[None, :]
    log_t1 = 3.0 * np.log1p(t_gl)
    log_t2 = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t_gl) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t_gl))
    max_log = np.maximum(log_t1, log_t2)
    sum_exp = np.exp(log_t1 - max_log) + np.exp(log_t2 - max_log)
    f_vals_gl = np.exp(-0.5 * max_log) / np.sqrt(sum_exp)
    quad_sums_gl = np.sum(f_vals_gl * w_gl[None, :], axis=1)
    exact_ints_safe = np.maximum(exact_ints, 1e-30)
    gl_err = np.max(np.abs(quad_sums_gl - exact_ints_safe) / exact_ints_safe)
    
    # Compute error for optimized
    t_opt = z_arr[:, None] * u_final[None, :]
    log_t1 = 3.0 * np.log1p(t_opt)
    log_t2 = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t_opt) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t_opt))
    max_log = np.maximum(log_t1, log_t2)
    sum_exp = np.exp(log_t1 - max_log) + np.exp(log_t2 - max_log)
    f_vals_opt = np.exp(-0.5 * max_log) / np.sqrt(sum_exp)
    quad_sums_opt = np.sum(f_vals_opt * w_final[None, :], axis=1)
    opt_err = np.max(np.abs(quad_sums_opt - exact_ints_safe) / exact_ints_safe)
    
    if opt_err > gl_err:
        u_final = u_gl
        w_final = w_gl

    return {"nodes": u_final, "weights": w_final}

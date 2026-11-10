import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    # Start timer
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit

    # Define a grid of cosmological parameters for validation
    # w0 in [-1.2, 0.5], w1 in [-1.5, 1.5]
    w0_vals = np.linspace(-1.2, 0.5, 10)
    w1_vals = np.linspace(-1.5, 1.5, 10)
    
    # Define a grid of redshifts for evaluation
    z_vals = np.linspace(0.05, 3.0, 15)
    
    # Precompute exact integrals for all combinations
    # I_exact = int_0^1 du / E(z*u)
    # E(t) = (1+t)^(3(1+w0)) * exp(3*w1*(1 - 1/(1+t)))
    
    exact_ints = []
    model_params = []
    
    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                def integrand(u, w0=w0, w1=w1, z=z):
                    t = z * u
                    log_E = 3.0 * (1.0 + w0) * np.log1p(t) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + t))
                    return np.exp(-log_E)
                
                try:
                    val, _ = scipy.integrate.quad(integrand, 0, 1, limit=100)
                    exact_ints.append(val)
                    model_params.append((w0, w1, z))
                except Exception:
                    us = np.linspace(0, 1, 501)
                    ts = z * us
                    log_E = 3.0 * (1.0 + w0) * np.log1p(ts) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + ts))
                    vals = np.exp(-log_E)
                    val = np.trapz(vals, us)
                    exact_ints.append(val)
                    model_params.append((w0, w1, z))

    exact_ints = np.array(exact_ints)
    
    # Extract arrays for vectorized objective
    w0_arr = np.array([p[0] for p in model_params])
    w1_arr = np.array([p[1] for p in model_params])
    z_arr = np.array([p[2] for p in model_params])
    
    def params_to_nodes_weights(params):
        theta = params[:8]
        alpha = params[8:16]
        
        u = 1.0 / (1.0 + np.exp(-theta))
        
        alpha_max = np.max(alpha)
        exp_alpha = np.exp(alpha - alpha_max)
        w = exp_alpha / np.sum(exp_alpha)
        
        return u, w

    def objective(params):
        u, w = params_to_nodes_weights(params)
        
        t = z_arr[:, None] * u[None, :]
        
        log_E = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
        
        f_vals = np.exp(-log_E)
        
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        
        return np.max(rel_errors)

    def objective_smooth(params):
        u, w = params_to_nodes_weights(params)
        
        t = z_arr[:, None] * u[None, :]
        
        log_E = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
        
        f_vals = np.exp(-log_E)
        
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        
        # Use log-sum-exp approximation to max for smoothness
        beta = 100.0
        max_err = np.max(rel_errors)
        if max_err > 0:
            smooth_max = max_err - np.log(np.sum(np.exp(-beta * rel_errors / max_err))) / beta
        else:
            smooth_max = 0.0
        
        return smooth_max

    # Initial guess: Gauss-Legendre nodes and weights on [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u0 = 0.5 + 0.5 * gl_nodes
    w0_init = 0.5 * gl_weights
    
    u0_safe = np.clip(u0, 1e-9, 1-1e-9)
    theta0 = np.log(u0_safe / (1 - u0_safe))
    
    w0_init_safe = np.clip(w0_init, 1e-9, 1.0)
    alpha0 = np.log(w0_init_safe)
    
    p0 = np.concatenate([theta0, alpha0])
    
    best_params = p0
    best_score = objective(p0)
    
    # Multi-start optimization
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 10:
            break
            
        if restart == 0:
            current_p = p0.copy()
        else:
            # Add random perturbation
            current_p = best_params + np.random.randn(16) * 0.1
        
        # Stage 1: Smooth optimization with L-BFGS-B
        try:
            res1 = scipy.optimize.minimize(
                objective_smooth, 
                current_p, 
                method='L-BFGS-B', 
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if time.time() - start_time < time_limit - 5:
                score1 = objective(res1.x)
                if score1 < best_score:
                    best_params = res1.x
                    best_score = score1
        except Exception:
            pass
        
        # Stage 2: Powell on raw objective
        if time.time() - start_time < time_limit - 5:
            try:
                res2 = scipy.optimize.minimize(
                    objective, 
                    best_params, 
                    method='Powell', 
                    options={'maxiter': 50, 'xtol': 1e-8, 'ftol': 1e-10}
                )
                
                if time.time() - start_time < time_limit - 2:
                    score2 = objective(res2.x)
                    if score2 < best_score:
                        best_params = res2.x
                        best_score = score2
            except Exception:
                pass

    # Final processing
    u_final, w_final = params_to_nodes_weights(best_params)
    
    # Ensure strict increasing order of nodes
    sort_idx = np.argsort(u_final)
    u_sorted = u_final[sort_idx]
    w_sorted = w_final[sort_idx]
    
    # Normalize weights
    w_norm = w_sorted / np.sum(w_sorted)
    
    return {"nodes": u_sorted, "weights": w_norm}

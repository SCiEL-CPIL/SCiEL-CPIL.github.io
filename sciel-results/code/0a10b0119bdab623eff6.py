import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit

    # Define a grid of cosmological parameters for validation
    # w0 in [-1.2, 0.5], w1 in [-1.5, 1.5]
    w0_vals = np.linspace(-1.2, 0.5, 15)
    w1_vals = np.linspace(-1.5, 1.5, 15)
    
    # Define a grid of redshifts for evaluation
    # The error should be minimized across a range of z
    # Include lower z to capture behavior near 0, and higher z for tail
    z_vals = np.concatenate([
        np.linspace(0.01, 0.1, 5),
        np.linspace(0.2, 1.0, 10),
        np.linspace(1.5, 3.0, 10),
        np.linspace(3.5, 6.0, 5)
    ])
    
    # Precompute exact integrals for all combinations
    # I_exact = int_0^1 du / E(z*u)
    # E(t) = (1+t)^(3(1+w0)) * exp(3*w1*(1 - 1/(1+t)))
    # 1/E(t) = exp(-log_E)
    
    exact_ints = []
    model_params = []
    
    # Vectorize the exact integral computation where possible, but quad is scalar
    # We will loop, but it's fast enough for 15*15*30 = 6750 calls
    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                def integrand(u, w0=w0, w1=w1, z=z):
                    t = z * u
                    # Use log1p for precision near t=0
                    log_E = 3.0 * (1.0 + w0) * np.log1p(t) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + t))
                    return np.exp(-log_E)
                
                # Use quad for high accuracy reference
                try:
                    val, _ = scipy.integrate.quad(integrand, 0, 1, limit=100, epsabs=1e-12, epsrel=1e-12)
                    exact_ints.append(val)
                    model_params.append((w0, w1, z))
                except Exception:
                    # Fallback
                    us = np.linspace(0, 1, 10001)
                    ts = z * us
                    log_E = 3.0 * (1.0 + w0) * np.log1p(ts) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + ts))
                    vals = np.exp(-log_E)
                    val = np.trapz(vals, us)
                    exact_ints.append(val)
                    model_params.append((w0, w1, z))

    exact_ints = np.array(exact_ints)
    
    # Extract w0, w1, z arrays for vectorized objective
    w0_arr = np.array([p[0] for p in model_params])
    w1_arr = np.array([p[1] for p in model_params])
    z_arr = np.array([p[2] for p in model_params])
    
    # Parameterization
    # 16 parameters: 8 for nodes (theta), 8 for weights (alpha)
    # u_i = sigmoid(theta_i)
    # w_i = softmax(alpha_i)
    
    def params_to_nodes_weights(params):
        theta = params[:8]
        alpha = params[8:16]
        
        # Nodes
        u = 1.0 / (1.0 + np.exp(-theta))
        
        # Weights
        # Shift alpha for numerical stability
        alpha_max = np.max(alpha)
        exp_alpha = np.exp(alpha - alpha_max)
        w = exp_alpha / np.sum(exp_alpha)
        
        return u, w

    def compute_errors(params):
        u, w = params_to_nodes_weights(params)
        
        # Calculate quadrature sum for all models
        # Create arrays of shape (n_models, 8)
        t = z_arr[:, None] * u[None, :]
        
        # Calculate E(t)
        log_E = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
        
        f_vals = np.exp(-log_E) # 1/E(t)
        
        # Quadrature sum: sum_i w_i f(t_i)
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        
        # Relative error
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        
        return rel_errors

    def objective(params):
        rel_errors = compute_errors(params)
        return np.max(rel_errors)

    def objective_smooth(params):
        # Use a soft-max or similar to make it differentiable for BFGS/L-BFGS-B
        rel_errors = compute_errors(params)
        
        # Soft max with temperature
        temp = 0.01
        return np.log(np.sum(np.exp(rel_errors / temp))) * temp

    # Initial guess: Gauss-Legendre nodes and weights on [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u0 = 0.5 + 0.5 * gl_nodes
    w0_init = 0.5 * gl_weights
    
    # Inverse transform for parameters
    u0_safe = np.clip(u0, 1e-9, 1-1e-9)
    theta0 = np.log(u0_safe / (1 - u0_safe))
    
    w0_init_safe = np.clip(w0_init, 1e-9, 1.0)
    alpha0 = np.log(w0_init_safe)
    
    p0 = np.concatenate([theta0, alpha0])
    
    best_params = p0
    best_score = objective(p0)
    
    # Strategy:
    # 1. Try L-BFGS-B on the smooth objective to get close to a local minimum
    # 2. Refine with Powell on the raw max objective
    
    try:
        # Step 1: L-BFGS-B on smooth objective
        res1 = scipy.optimize.minimize(
            objective_smooth, 
            p0, 
            method='L-BFGS-B', 
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if time.time() - start_time < time_limit - 10:
            if objective(res1.x) < best_score:
                best_params = res1.x
                best_score = objective(res1.x)
                
        # Step 2: Powell on raw objective starting from best so far
        res2 = scipy.optimize.minimize(
            objective, 
            best_params, 
            method='Powell', 
            options={'maxiter': 200, 'xtol': 1e-9, 'ftol': 1e-12}
        )
        
        if time.time() - start_time < time_limit - 5:
            if res2.fun < best_score:
                best_params = res2.x
                best_score = res2.fun
                
    except Exception:
        pass

    # Check time
    if time.time() - start_time > time_limit - 2:
        u_final, w_final = params_to_nodes_weights(best_params)
        sort_idx = np.argsort(u_final)
        u_sorted = u_final[sort_idx]
        w_sorted = w_final[sort_idx]
        w_norm = w_sorted / np.sum(w_sorted)
        return {"nodes": u_sorted, "weights": w_norm}

    u_final, w_final = params_to_nodes_weights(best_params)
    
    # Ensure strict increasing order of nodes
    sort_idx = np.argsort(u_final)
    u_sorted = u_final[sort_idx]
    w_sorted = w_final[sort_idx]
    
    # Normalize weights to sum to 1 exactly
    w_norm = w_sorted / np.sum(w_sorted)
    
    return {"nodes": u_sorted, "weights": w_norm}

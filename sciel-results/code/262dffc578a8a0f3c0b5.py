import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit

    # Define a grid of cosmological parameters for validation
    # We increase the resolution slightly to better capture the worst-case error
    w0_vals = np.linspace(-1.2, 0.5, 20)
    w1_vals = np.linspace(-1.5, 1.5, 20)
    
    # Define grid for z
    # Include a wider range and higher density at high z where errors often peak
    z_vals = np.concatenate([
        np.linspace(0.01, 0.2, 10),
        np.linspace(0.3, 1.0, 15),
        np.linspace(1.2, 3.0, 15),
        np.linspace(3.2, 6.0, 10),
        np.linspace(6.5, 10.0, 5)
    ])
    
    # Precompute exact integrals for all combinations
    # I_exact = int_0^1 du / E(z*u)
    # E(t)^2 = (1+t)^3 + (1+t)^(3(1+w0)) * exp(3*w1*(1 - 1/(1+t)))
    # 1/E(t) = 1 / sqrt((1+t)^3 + (1+t)^(3(1+w0)) * exp(3*w1*(1 - 1/(1+t))))
    
    exact_ints = []
    model_params = []
    
    # Vectorized computation of exact integrals is difficult with quad, so we loop
    # But we can vectorize the integrand evaluation if we use a fixed grid for quad? No, quad is adaptive.
    # We will stick to the loop but optimize the inner function.
    
    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                def integrand(u, w0=w0, w1=w1, z=z):
                    t = z * u
                    # Use log for precision
                    term1 = 3.0 * np.log1p(t)
                    term2 = 3.0 * (1.0 + w0) * np.log1p(t) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + t))
                    # To avoid overflow in exp, use log-sum-exp
                    max_term = np.maximum(term1, term2)
                    log_sum = max_term + np.log(np.exp(term1 - max_term) + np.exp(term2 - max_term))
                    return np.exp(-0.5 * log_sum)
                
                # Use quad for high accuracy reference
                try:
                    val, _ = scipy.integrate.quad(integrand, 0, 1, limit=100, epsabs=1e-12, epsrel=1e-12)
                    exact_ints.append(val)
                    model_params.append((w0, w1, z))
                except Exception:
                    # Fallback
                    us = np.linspace(0, 1, 10001)
                    ts = z * us
                    term1 = 3.0 * np.log1p(ts)
                    term2 = 3.0 * (1.0 + w0) * np.log1p(ts) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + ts))
                    max_term = np.maximum(term1, term2)
                    log_sum = max_term + np.log(np.exp(term1 - max_term) + np.exp(term2 - max_term))
                    vals = np.exp(-0.5 * log_sum)
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

    def objective(params):
        u, w = params_to_nodes_weights(params)
        
        # Calculate quadrature sum for all models
        # Create arrays of shape (n_models, 8)
        t = z_arr[:, None] * u[None, :]
        
        # Calculate E(t)
        term1 = 3.0 * np.log1p(t)
        term2 = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
        max_term = np.maximum(term1, term2)
        log_sum = max_term + np.log(np.exp(term1 - max_term) + np.exp(term2 - max_term))
        f_vals = np.exp(-0.5 * log_sum) # 1/E(t)
        
        # Quadrature sum: sum_i w_i f(t_i)
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        
        # Relative error
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        
        # Return the max error
        return np.max(rel_errors)

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
    # Use Differential Evolution for global search.
    # It is robust to local minima and handles box constraints.
    # We need to define bounds for theta and alpha.
    # theta: sigmoid maps R to (0,1). To cover most of [0,1], theta in [-10, 10] is sufficient.
    # alpha: log-weights. Weights sum to 1, so max weight is 1, min is >0.
    # If one weight is 1, others are 0. log(0) is -inf.
    # Let's bound alpha to [-10, 10]. exp(10) ~ 22000, exp(-10) ~ 4e-5.
    # This allows weights down to ~1e-4 if normalized, which is fine.
    
    bounds = [(-10, 10)] * 16
    
    try:
        # Differential Evolution
        # maxiter: 50
        # popsize: 15 (default is 15)
        # tol: 1e-8
        # mutation: (0.5, 1.0)
        # recombination: 0.9
        # seed: 42
        
        # Check time before starting DE
        if time.time() - start_time < time_limit - 10:
            res_de = scipy.optimize.differential_evolution(
                objective, 
                bounds, 
                maxiter=50, 
                popsize=15, 
                tol=1e-8,
                seed=42,
                polish=True
            )
            
            if res_de.fun < best_score:
                best_params = res_de.x
                best_score = res_de.fun
            
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

    # Refine with Powell on the best found so far
    if time.time() - start_time < time_limit - 5:
        try:
            res_powell = scipy.optimize.minimize(
                objective, 
                best_params, 
                method='Powell', 
                options={'maxiter': 100, 'xtol': 1e-9, 'ftol': 1e-12}
            )
            if res_powell.fun < best_score:
                best_params = res_powell.x
                best_score = res_powell.fun
        except Exception:
            pass

    u_final, w_final = params_to_nodes_weights(best_params)
    
    # Ensure strict increasing order of nodes
    sort_idx = np.argsort(u_final)
    u_sorted = u_final[sort_idx]
    w_sorted = w_final[sort_idx]
    
    # Normalize weights to sum to 1 exactly
    w_norm = w_sorted / np.sum(w_sorted)
    
    # Final check for validity
    # Nodes strictly increasing in [0,1]
    if not np.all(np.diff(u_sorted) > 0):
        # Fallback to Gauss-Legendre if sorting failed or something went wrong
        u_sorted, w_norm = np.polynomial.legendre.leggauss(8)
        u_sorted = 0.5 + 0.5 * u_sorted
        w_norm = 0.5 * w_norm

    return {"nodes": u_sorted, "weights": w_norm}

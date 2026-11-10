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
    w0_vals = np.linspace(-1.2, 0.5, 12)
    w1_vals = np.linspace(-1.5, 1.5, 12)
    
    # Define a grid of redshifts for evaluation
    # The error should be minimized across a range of z
    z_vals = np.linspace(0.05, 3.0, 25)
    
    # Precompute exact integrals for all combinations
    # I_exact = int_0^1 du / E(z*u)
    # E(t) = (1+t)^(3(1+w0)) * exp(3*w1*(1 - 1/(1+t)))
    
    exact_ints = []
    model_params = []
    
    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                # If z is very small, integral is ~ 1
                # E(z*u) for small z*u is ~ 1
                # We need to handle z=0 separately or start from small z
                # The formula for dL has factor z, so relative error is well defined for z>0
                
                def integrand(u, w0=w0, w1=w1, z=z):
                    t = z * u
                    # E(t)
                    log_E = 3.0 * (1.0 + w0) * np.log1p(t) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + t))
                    return np.exp(-log_E)
                
                # Use quad for high accuracy reference
                try:
                    val, _ = scipy.integrate.quad(integrand, 0, 1, limit=100)
                    exact_ints.append(val)
                    model_params.append((w0, w1, z))
                except Exception:
                    # Fallback to fixed quadrature if quad fails (unlikely)
                    us = np.linspace(0, 1, 1001)
                    ts = z * us
                    log_E = 3.0 * (1.0 + w0) * np.log1p(ts) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + ts))
                    vals = np.exp(-log_E)
                    val = np.trapz(vals, us)
                    exact_ints.append(val)
                    model_params.append((w0, w1, z))

    exact_ints = np.array(exact_ints)
    n_models = len(exact_ints)
    
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
        # u shape (8,) -> (1, 8)
        # z shape (n_models,) -> (n_models, 1)
        # t = z[:, None] * u[None, :]
        
        t = z_arr[:, None] * u[None, :]
        
        # Calculate E(t)
        # log_E = 3(1+w0)ln(1+t) + 3w1(1 - 1/(1+t))
        log_E = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
        
        f_vals = np.exp(-log_E) # 1/E(t)
        
        # Quadrature sum: sum_i w_i f(t_i)
        # w shape (8,) -> (1, 8)
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        
        # Relative error
        # err = |quad - exact| / exact
        # exact_ints shape (n_models,)
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        
        # Max relative error
        return np.max(rel_errors)

    # Initial guess: Gauss-Legendre nodes and weights on [0,1]
    # Transform from [-1,1] to [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u0 = 0.5 + 0.5 * gl_nodes
    w0_init = 0.5 * gl_weights
    
    # Inverse transform for parameters
    # u = sigmoid(theta) -> theta = log(u / (1-u))
    # Clamp u to avoid inf
    u0_safe = np.clip(u0, 1e-9, 1-1e-9)
    theta0 = np.log(u0_safe / (1 - u0_safe))
    
    # w = softmax(alpha) -> alpha = log(w)
    # Clamp w
    w0_init_safe = np.clip(w0_init, 1e-9, 1.0)
    alpha0 = np.log(w0_init_safe)
    
    p0 = np.concatenate([theta0, alpha0])
    
    # Optimization
    best_params = p0
    best_score = objective(p0)
    
    try:
        # Use Powell for robustness with non-smooth max objective
        # Reduce maxiter slightly to ensure we finish in time, but enough to converge
        result = scipy.optimize.minimize(
            objective, 
            p0, 
            method='Powell', 
            options={'maxiter': 150, 'xtol': 1e-8, 'ftol': 1e-10}
        )
        
        # Check time
        if time.time() - start_time < time_limit:
            if result.fun < best_score:
                best_params = result.x
                best_score = result.fun
            else:
                # Keep initial if result is worse
                best_params = p0
                
    except Exception as e:
        # If optimization fails, stick with initial Gauss-Legendre
        best_params = p0

    # Time check before final processing
    if time.time() - start_time > time_limit - 5:
        # Emergency exit with best known
        u_final, w_final = params_to_nodes_weights(best_params)
        # Ensure strict increasing order
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
    
    # Normalize weights to sum to 1 exactly (floating point error)
    w_norm = w_sorted / np.sum(w_sorted)
    
    return {"nodes": u_sorted, "weights": w_norm}

# EVOLVE-BLOCK-END

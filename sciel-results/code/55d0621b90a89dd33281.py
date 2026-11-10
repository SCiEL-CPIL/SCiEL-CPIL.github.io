import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    time_limit = 85.0

    # 1. Define Cosmology Grid
    w0_vals = np.linspace(-1.2, 0.5, 10)
    w1_vals = np.linspace(-1.5, 1.5, 10)
    z_vals = np.logspace(np.log10(0.02), np.log10(6.0), 20)
    
    # Flatten
    w0_grid, w1_grid, z_grid = np.meshgrid(w0_vals, w1_vals, z_vals, indexing='ij')
    w0_flat = w0_grid.flatten()
    w1_flat = w1_grid.flatten()
    z_flat = z_grid.flatten()
    n_models = len(z_flat)

    # 2. Compute Exact Integrals
    exact_ints = np.zeros(n_models)
    for i in range(n_models):
        w0 = w0_flat[i]
        w1 = w1_flat[i]
        z = z_flat[i]
        
        def integrand(u):
            t = z * u
            # Use the prompt's specific integrand definition
            log_E = 3.0 * (1.0 + w0) * np.log1p(t) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + t))
            return np.exp(-log_E)
            
        try:
            val, _ = scipy.integrate.quad(integrand, 0, 1, limit=100, epsabs=1e-12, epsrel=1e-12)
            exact_ints[i] = val
        except Exception:
            # Fallback
            us = np.linspace(0, 1, 10000)
            ts = z * us
            log_E = 3.0 * (1.0 + w0) * np.log1p(ts) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + ts))
            vals = np.exp(-log_E)
            exact_ints[i] = np.trapz(vals, us)

    # 3. Define Function to Evaluate Nodes
    def evaluate_nodes(nodes):
        nodes = np.sort(nodes)
        # Ensure strict increase and bounds
        nodes = np.clip(nodes, 1e-6, 1-1e-6)
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-7
        
        # Compute integrand values at nodes for all models
        # t = z * nodes
        # Shape: (n_models, 8)
        t = z_flat[:, None] * nodes[None, :]
        log_E = 3.0 * (1.0 + w0_flat)[:, None] * np.log1p(t) + 3.0 * w1_flat[:, None] * (1.0 - 1.0 / (1.0 + t))
        f_vals = np.exp(-log_E)
        
        # Solve LP for weights
        n_nodes = 8
        c = np.zeros(n_nodes + 1)
        c[n_nodes] = 1.0
        
        A_ub = np.zeros((2 * n_models, n_nodes + 1))
        b_ub = np.zeros(2 * n_models)
        
        # Vectorized construction of A_ub and b_ub might be faster but loop is fine for 200*2 rows
        for i in range(n_models):
            A_ub[2*i, :n_nodes] = f_vals[i, :]
            A_ub[2*i, n_nodes] = -1.0
            b_ub[2*i] = exact_ints[i]
            
            A_ub[2*i+1, :n_nodes] = -f_vals[i, :]
            A_ub[2*i+1, n_nodes] = -1.0
            b_ub[2*i+1] = -exact_ints[i]
        
        A_eq = np.zeros((1, n_nodes + 1))
        A_eq[0, :n_nodes] = 1.0
        b_eq = np.array([1.0])
        
        bounds = [(0, None)] * n_nodes + [(0, None)]
        
        try:
            res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                                         bounds=bounds, method='highs')
            if res.success:
                return res.x[n_nodes] # Return epsilon
            else:
                return 1e10
        except Exception:
            return 1e10

    # 4. Global Optimization with Differential Evolution
    bounds = [(1e-5, 1-1e-5)] * 8
    
    # Time check
    if time.time() - start_time < time_limit - 20:
        try:
            # Use a seed for reproducibility if needed, but random is fine
            res_de = scipy.optimize.differential_evolution(
                evaluate_nodes, 
                bounds, 
                maxiter=100, 
                popsize=15, 
                tol=1e-5, 
                seed=42,
                init='latinhypercube'
            )
            best_nodes = res_de.x
            best_err = res_de.fun
        except Exception:
            # Fallback to Gauss-Legendre
            gl_nodes, _ = np.polynomial.legendre.leggauss(8)
            best_nodes = 0.5 + 0.5 * gl_nodes
            best_err = evaluate_nodes(best_nodes)
    else:
        gl_nodes, _ = np.polynomial.legendre.leggauss(8)
        best_nodes = 0.5 + 0.5 * gl_nodes
        best_err = evaluate_nodes(best_nodes)

    # 5. Local Refinement with Nelder-Mead
    if time.time() - start_time < time_limit - 5:
        try:
            # Sort nodes for NM start
            start_nodes = np.sort(best_nodes)
            res_nm = scipy.optimize.minimize(
                evaluate_nodes, 
                start_nodes, 
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res_nm.fun < best_err:
                best_nodes = res_nm.x
                best_err = res_nm.fun
        except Exception:
            pass

    # 6. Final Weights Calculation
    # Re-solve LP for the final nodes to get clean weights
    final_nodes = np.sort(best_nodes)
    final_nodes = np.clip(final_nodes, 1e-6, 1-1e-6)
    for i in range(1, 8):
        if final_nodes[i] <= final_nodes[i-1]:
            final_nodes[i] = final_nodes[i-1] + 1e-7
            
    t = z_flat[:, None] * final_nodes[None, :]
    log_E = 3.0 * (1.0 + w0_flat)[:, None] * np.log1p(t) + 3.0 * w1_flat[:, None] * (1.0 - 1.0 / (1.0 + t))
    f_vals = np.exp(-log_E)
    
    n_nodes = 8
    c = np.zeros(n_nodes + 1)
    c[n_nodes] = 1.0
    A_ub = np.zeros((2 * n_models, n_nodes + 1))
    b_ub = np.zeros(2 * n_models)
    for i in range(n_models):
        A_ub[2*i, :n_nodes] = f_vals[i, :]
        A_ub[2*i, n_nodes] = -1.0
        b_ub[2*i] = exact_ints[i]
        A_ub[2*i+1, :n_nodes] = -f_vals[i, :]
        A_ub[2*i+1, n_nodes] = -1.0
        b_ub[2*i+1] = -exact_ints[i]
        
    A_eq = np.zeros((1, n_nodes + 1))
    A_eq[0, :n_nodes] = 1.0
    b_eq = np.array([1.0])
    bounds_lp = [(0, None)] * n_nodes + [(0, None)]
    
    try:
        res_final = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                                           bounds=bounds_lp, method='highs')
        if res_final.success:
            weights = res_final.x[:n_nodes]
        else:
            # Fallback weights
            _, gl_w = np.polynomial.legendre.leggauss(8)
            weights = 0.5 * gl_w
    except Exception:
        _, gl_w = np.polynomial.legendre.leggauss(8)
        weights = 0.5 * gl_w

    # Clean up weights
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)
    
    return {"nodes": final_nodes, "weights": weights}

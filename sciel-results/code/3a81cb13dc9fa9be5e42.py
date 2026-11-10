# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio optimization on a 10x10 grid.
    
    Strategy:
    1. Use the product-of-marginals structure P[y, x] = R[y] * C[x].
       This reduces the search space from 100 dimensions to 20 dimensions.
       The entropies of projections decompose into sums over the marginals,
       allowing for efficient evaluation.
    2. Use Differential Evolution for robust global optimization of the
       20-dimensional space.
    3. Refine the best candidate using L-BFGS-B.
    4. Finally, perform a full 100-dimensional local refinement using 
       multiplicative weights (mirror descent) to escape the product-form 
       manifold and capture any non-separable structure.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    # Pre-compute grid coordinates
    y_idx = np.arange(n)
    x_idx = np.arange(n)
    
    # Pre-compute projection values for each slope
    # P[y, x] -> value = x + r*y
    # Note: In P[y, x], the first index is y and the second is x.
    # So for a cell (y, x), the value is x + r*y.
    proj_m1 = x_idx[None, :] - y_idx[:, None]  # x - y
    proj_0 = x_idx[None, :]                    # x
    proj_1 = x_idx[None, :] + y_idx[:, None]   # x + y
    proj_2 = x_idx[None, :] + 2 * y_idx[:, None] # x + 2y

    def compute_entropies_from_marginals(R, C):
        """
        Compute H_{-1}, H_0, H_1, H_2 given marginals R (for Y) and C (for X).
        P[y, x] = R[y] * C[x]
        """
        P = np.outer(R, C)
        
        def entropy_of_projection(proj_vals, P):
            # Pool cells with equal projection value
            # Using np.bincount for speed
            # We need to map projection values to non-negative integers
            min_val = np.min(proj_vals)
            max_val = np.max(proj_vals)
            size = int(max_val - min_val) + 1
            idx = (proj_vals - min_val).astype(np.int32).ravel()
            weights = P.ravel()
            counts = np.bincount(idx, weights=weights, minlength=size)
            probs = counts[counts > 1e-15]
            if len(probs) == 0:
                return 0.0
            return -np.sum(probs * np.log2(probs))
        
        D = entropy_of_projection(proj_m1, P)
        H0 = entropy_of_projection(proj_0, P)
        H1 = entropy_of_projection(proj_1, P)
        H2 = entropy_of_projection(proj_2, P)
        M = max(H0, H1, H2)
        
        if M <= 1e-9:
            return -1e10, P
        return D / M, P

    def objective_product(params):
        """
        Objective function for the product form.
        params: [log_R_0, ..., log_R_9, log_C_0, ..., log_C_9]
        """
        if time.time() > deadline - 10:
            return 1000.0

        log_R = params[:n]
        log_C = params[n:]
        
        # Convert to probabilities via softmax-like normalization
        R = np.exp(log_R - np.max(log_R))
        R /= np.sum(R)
        
        C = np.exp(log_C - np.max(log_C))
        C /= np.sum(C)
        
        ratio, _ = compute_entropies_from_marginals(R, C)
        return -ratio

    # 1. Global Optimization with Differential Evolution
    # Bounds for log probabilities. Since we normalize, the absolute scale doesn't matter,
    # but the relative differences do. A range of [-5, 5] is usually sufficient.
    bounds = [(-5.0, 5.0)] * (2 * n)
    
    best_ratio = -np.inf
    best_R = np.ones(n) / n
    best_C = np.ones(n) / n
    
    # Run DE
    try:
        res_de = differential_evolution(
            objective_product, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-8, 
            polish=False, # We will polish manually
            popsize=15
        )
        if res_de.fun < -best_ratio:
            best_ratio = -res_de.fun
            log_R = res_de.x[:n]
            log_C = res_de.x[n:]
            best_R = np.exp(log_R - np.max(log_R))
            best_R /= np.sum(best_R)
            best_C = np.exp(log_C - np.max(log_C))
            best_C /= np.sum(best_C)
    except Exception as e:
        pass

    # 2. Refinement with L-BFGS-B
    # Use the best point from DE as starting point
    if best_ratio > -np.inf:
        p0 = np.concatenate([np.log(best_R + 1e-15), np.log(best_C + 1e-15)])
    else:
        p0 = np.zeros(2 * n)
        
    try:
        res_lbfgs = minimize(
            objective_product, 
            p0, 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res_lbfgs.fun < -best_ratio:
            best_ratio = -res_lbfgs.fun
            log_R = res_lbfgs.x[:n]
            log_C = res_lbfgs.x[n:]
            best_R = np.exp(log_R - np.max(log_R))
            best_R /= np.sum(best_R)
            best_C = np.exp(log_C - np.max(log_C))
            best_C /= np.sum(best_C)
    except Exception as e:
        pass

    # Also try a few random restarts for L-BFGS-B if time permits
    rng = np.random.default_rng(42)
    for _ in range(5):
        if time.time() > deadline - 60:
            break
        p0_rand = rng.uniform(-3, 3, 2 * n)
        try:
            res_rand = minimize(
                objective_product, 
                p0_rand, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res_rand.fun < -best_ratio:
                best_ratio = -res_rand.fun
                log_R = res_rand.x[:n]
                log_C = res_rand.x[n:]
                best_R = np.exp(log_R - np.max(log_R))
                best_R /= np.sum(best_R)
                best_C = np.exp(log_C - np.max(log_C))
                best_C /= np.sum(best_C)
        except Exception as e:
            pass

    # Construct P from best marginals
    P_best = np.outer(best_R, best_C)
    
    # 3. Full 100-dim local refinement
    # Use multiplicative weights update (mirror descent) to refine P directly.
    # This allows P to deviate from the product form if beneficial.
    
    def compute_ratio_full(P):
        def entropy_of_projection(proj_vals, P):
            min_val = np.min(proj_vals)
            max_val = np.max(proj_vals)
            size = int(max_val - min_val) + 1
            idx = (proj_vals - min_val).astype(np.int32).ravel()
            weights = P.ravel()
            counts = np.bincount(idx, weights=weights, minlength=size)
            probs = counts[counts > 1e-15]
            if len(probs) == 0:
                return 0.0
            return -np.sum(probs * np.log2(probs))
        
        D = entropy_of_projection(proj_m1, P)
        H0 = entropy_of_projection(proj_0, P)
        H1 = entropy_of_projection(proj_1, P)
        H2 = entropy_of_projection(proj_2, P)
        M = max(H0, H1, H2)
        if M <= 1e-9:
            return -1e10
        return D / M

    # Gradient-based refinement on log P
    # We want to maximize Ratio(P).
    # Let L = log P. P = exp(L) / sum(exp(L)).
    # We can use L-BFGS-B on the 100 parameters of log P.
    
    def objective_full(log_P_flat):
        if time.time() > deadline - 5:
            return 1000.0
        log_P = log_P_flat.reshape(n, n)
        P = np.exp(log_P - np.max(log_P))
        P /= np.sum(P)
        ratio = compute_ratio_full(P)
        return -ratio

    # Initial guess for full refinement is the product form P
    log_P_init = np.log(P_best + 1e-15).ravel()
    
    try:
        res_full = minimize(
            objective_full, 
            log_P_init, 
            method='L-BFGS-B', 
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-7}
        )
        if res_full.fun < -best_ratio:
            log_P_opt = res_full.x.reshape(n, n)
            P_opt = np.exp(log_P_opt - np.max(log_P_opt))
            P_opt /= np.sum(P_opt)
            ratio_opt = compute_ratio_full(P_opt)
            if ratio_opt > best_ratio:
                best_ratio = ratio_opt
                P_best = P_opt
    except Exception as e:
        pass

    # Final validation
    P_best = np.clip(P_best, 0, None)
    s = np.sum(P_best)
    if s > 1e-9:
        P_best = P_best / s
    else:
        P_best = np.ones((n, n)) / (n * n)

    return P_best

# EVOLVE-BLOCK-END

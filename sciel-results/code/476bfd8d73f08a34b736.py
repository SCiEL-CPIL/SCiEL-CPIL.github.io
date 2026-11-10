import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio D/M where:
    D = H(x - y)
    M = max(H(x), H(x + y), H(x + 2y))
    
    Strategy:
    1. Use Differential Evolution (DE) for global exploration.
       DE is a population-based global optimizer that is robust to local minima,
       unlike the multi-start local SLSQP used in the parent.
    2. Use the best result from DE as a starting point for SLSQP local refinement.
    3. Ensure strict time management to stay within the 630s limit.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 610.0  # Safety margin before 630s kill

    # Grid coordinates
    ys, xs = np.mgrid[0:n, 0:n]
    
    # Flatten for binning
    d_val = (xs - ys).flatten()
    m0_val = xs.flatten()
    m1_val = (xs + ys).flatten()
    m2_val = (xs + 2 * ys).flatten()
    
    # Unique values and inverse mapping
    d_vals, d_inv = np.unique(d_val, return_inverse=True)
    m0_vals, m0_inv = np.unique(m0_val, return_inverse=True)
    m1_vals, m1_inv = np.unique(m1_val, return_inverse=True)
    m2_vals, m2_inv = np.unique(m2_val, return_inverse=True)
    
    num_d = len(d_vals)
    num_m0 = len(m0_vals)
    num_m1 = len(m1_vals)
    num_m2 = len(m2_vals)

    def compute_entropies_from_p(p_flat):
        p_d = np.bincount(d_inv, weights=p_flat, minlength=num_d)
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=num_m2)
        
        def shannon(p_dist):
            p_nonzero = p_dist[p_dist > 1e-15]
            if len(p_nonzero) == 0:
                return 0.0
            return -np.sum(p_nonzero * np.log2(p_nonzero))
        
        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)
        
        M = max(H_m0, H_m1, H_m2)
        return H_d, M

    def get_score(p_flat):
        H_d, M = compute_entropies_from_p(p_flat)
        if M < 1e-9:
            return -1e9
        return H_d / M

    def objective(p_flat):
        """
        Objective function for minimizers.
        We minimize -D/M.
        """
        H_d, M = compute_entropies_from_p(p_flat)
        if M < 1e-9:
            return 1e9
        return -H_d / M

    # --- Phase 1: Global Search with Differential Evolution ---
    # DE requires bounds. We use (0, 1) for each probability, 
    # but we must ensure the sum is 1. 
    # DE doesn't handle equality constraints well directly.
    # We can use a penalty method or transform the space.
    # A common trick: Optimize over 99 variables, set the 100th to 1 - sum(others).
    # However, this might lead to negative values if the sum of 99 > 1.
    # Better: Use a simplex projection inside the objective or use a parameterization.
    
    # Let's use a simpler approach: 
    # DE on 100 variables with bounds [0, 1].
    # In the objective, we clip and normalize. 
    # This effectively searches the simplex.
    
    def de_objective(p_flat):
        # Clip to ensure non-negativity
        p_clip = np.clip(p_flat, 0.0, 1.0)
        s = np.sum(p_clip)
        if s < 1e-9:
            return 1e9
        p_norm = p_clip / s
        return objective(p_norm)

    bounds = [(0.0, 1.0) for _ in range(100)]
    
    # DE parameters
    # popsize=15, maxiter=200, tol=1e-8
    # This is a heavy but global search.
    try:
        # Use a seed for reproducibility in the search logic, though DE is stochastic
        res_de = differential_evolution(
            de_objective, 
            bounds, 
            seed=42,
            maxiter=100, # Limit iterations to save time
            popsize=20,   # Small population for speed
            tol=1e-6,
            polish=False, # We will polish manually later
            init='sobol'  # Better initial distribution
        )
        
        p_de = res_de.x
        p_de = np.clip(p_de, 0.0, None)
        p_de /= np.sum(p_de)
        
    except Exception as e:
        # Fallback if DE fails or times out
        p_de = np.ones(100) / 100.0

    if time.time() > deadline - 50.0:
        # Not enough time for local polish, return best DE result
        P = p_de.reshape((n, n))
        P = np.clip(P, 0.0, None)
        s = np.sum(P)
        if s > 1e-9:
            P = P / s
        else:
            P = np.ones((n, n)) / (n * n)
        return P

    # --- Phase 2: Local Refinement with SLSQP ---
    # Use the best DE result as the starting point for SLSQP.
    # Also try a few other high-quality starts (uniform, diagonal) to ensure robustness.
    
    starts = [p_de]
    starts.append(np.ones(100) / 100.0)
    
    # Add a structured start: "Anti-diagonal" (maximizes H(X-Y))
    p_adiag = np.zeros(100)
    for i in range(10):
        p_adiag[i*10+(9-i)] = 1.0/10.0
    starts.append(p_adiag)
    
    best_score = -1e9
    p_best = p_de
    
    constraints = [{'type': 'eq', 'fun': lambda p: np.sum(p) - 1.0}]
    bounds_slsqp = [(0.0, 1.0) for _ in range(100)]
    
    for start_p in starts:
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(
                objective, 
                start_p, 
                method='SLSQP', 
                bounds=bounds_slsqp, 
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            
            p_res = res.x
            p_res = np.clip(p_res, 0.0, None)
            s = np.sum(p_res)
            if s < 1e-9: continue
            p_res /= s
            
            score = get_score(p_res)
            
            if score > best_score:
                best_score = score
                p_best = p_res
                
        except Exception as e:
            pass

    # Final validation and normalization
    P = p_best.reshape((n, n))
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
        
    return P

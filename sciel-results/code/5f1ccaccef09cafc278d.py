import numpy as np
import time
import scipy.optimize as opt
import itertools

def shannon_entropy(probs):
    probs = probs[probs > 1e-15]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def calc_entropies(P):
    # P is 10x10, P[y, x]
    n = 10
    X = np.arange(n)
    Y = np.arange(n)
    X_grid, Y_grid = np.meshgrid(X, Y, indexing='xy')
    
    # D = X - Y
    D_vals = X_grid - Y_grid
    # M components: X, X+Y, X+2Y
    X_vals = X_grid
    S1_vals = X_grid + Y_grid
    S2_vals = X_grid + 2 * Y_grid
    
    # Helper to get entropy of a linear projection
    def proj_entropy(vals):
        unique_vals, inverse = np.unique(vals, return_inverse=True)
        counts = np.bincount(inverse, weights=P.ravel())
        probs = counts / counts.sum()
        return shannon_entropy(probs)
        
    H_D = proj_entropy(D_vals)
    H_X = proj_entropy(X_vals)
    H_S1 = proj_entropy(S1_vals)
    H_S2 = proj_entropy(S2_vals)
    
    M = max(H_X, H_S1, H_S2)
    return H_D, M

def objective(u):
    # u is a vector of 100 real numbers
    # Convert to probabilities using softmax
    u = np.array(u)
    u -= np.max(u) # for numerical stability
    P = np.exp(u)
    P_sum = np.sum(P)
    if P_sum < 1e-15:
        return 0.0
    P = P / P_sum
    P = P.reshape(10, 10)
    
    H_D, M = calc_entropies(P)
    if M < 1e-9:
        return 0.0
    return H_D / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0 # 10s safety margin
    
    n = 10
    # Initial best
    P_best = np.ones((n, n)) / (n * n)
    score_best = objective(P_best.ravel())
    
    # Strategy: Differential Evolution
    # Bounds for u (log-probs): [-20, 20]
    bounds = [(-20, 20)] * (n * n)
    
    # DE might be slow for 100 dimensions. 
    # Let's try a smaller population or fewer generations, or use a local polish.
    
    # First, try a few random restarts with local optimization (L-BFGS-B)
    # which is faster per iteration.
    
    def local_optimize(u_init, time_limit):
        # Maximize objective, so minimize -objective
        res = opt.minimize(lambda x: -objective(x), u_init, method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 1000, 'disp': False, 'maxfun': 10000})
        if res.success:
            P_res = np.exp(res.x)
            P_res = P_res / np.sum(P_res)
            P_res = P_res.reshape(n, n)
            score_res = objective(res.x)
            return P_res, score_res
        return None, -np.inf

    # 1. Uniform start
    u_uniform = np.zeros(n * n)
    P_loc, score_loc = local_optimize(u_uniform, 5)
    if score_loc > score_best:
        P_best = P_loc
        score_best = score_loc
        
    # 2. Random sparse starts
    # Try to find structures where support is small
    for i in range(20):
        if time.time() > deadline - 30:
            break
        # Random initialization
        u_rand = np.random.uniform(-10, 10, n * n)
        # Bias towards sparse? 
        # Just run local opt
        P_loc, score_loc = local_optimize(u_rand, 5)
        if score_loc > score_best:
            P_best = P_loc
            score_best = score_loc
            
    # 3. Differential Evolution for global search if time permits
    # DE is expensive. 100 dims.
    # Let's run DE with small population for remaining time.
    if time.time() < deadline - 60:
        # Use a subset of the best found points or random points as initial population?
        # DE generates its own.
        # We need to be careful with time.
        # Let's try a quick DE run.
        try:
            # maxiter is tricky with time. We'll just run a fixed small number of iterations 
            # and check time.
            # Actually, DE in scipy doesn't have a direct time limit.
            # We can wrap the objective to raise an exception or stop.
            # Simpler: Just do a few random restarts with DE? No, DE is the global search.
            
            # Let's try a manual DE-like loop or just trust the local opts + random restarts.
            # Given 100 dimensions, DE might struggle to converge in 500s if pop size is large.
            # Pop size 10 * 100 = 1000. 1000 evaluations per generation.
            # Each eval is ~1ms? 1s per gen. 500 gens = 500s.
            # This is feasible.
            
            def de_objective(x):
                return -objective(x)
                
            # Run DE
            # Use tol to stop early if converged
            res = opt.differential_evolution(de_objective, bounds, maxiter=500, 
                                             popsize=10, tol=1e-5, 
                                             polish=True, seed=42)
            
            if res.success or res.fun < -score_best:
                P_res = np.exp(res.x)
                P_res = P_res / np.sum(P_res)
                P_res = P_res.reshape(n, n)
                score_res = -res.fun
                if score_res > score_best:
                    P_best = P_res
                    score_best = score_res
                    
        except Exception:
            pass

    # Final polish with L-BFGS-B on the best found
    if time.time() < deadline - 10:
        u_best = np.log(P_best.ravel() + 1e-15)
        P_loc, score_loc = local_optimize(u_best, 5)
        if score_loc > score_best:
            P_best = P_loc
            score_best = score_loc

    # Ensure validity
    P_best = np.clip(P_best, 0, None)
    P_sum = np.sum(P_best)
    if P_sum < 1e-9:
        return np.ones((n, n)) / (n * n)
    P_best = P_best / P_sum
    
    return P_best

import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """
    Compute the score: L2^2 / (L1 * Sup(autoconv))
    Assumes h is the vector of heights.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max)) / (DELTA**2)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_score = _score_raw(best_h)
    
    # Objective to minimize: - (sum(h^2) / max(conv(h,h)))
    # Subject to sum(h)=1, h>=0
    
    def neg_obj(h):
        l1 = np.sum(h)
        if l1 < 1e-12: return 0
        l2_sq = np.sum(h**2)
        hc = np.convolve(h, h, mode='full')
        mc = np.max(hc)
        if mc < 1e-12: return 0
        return -(l2_sq / mc)
        
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
    ]
    
    bounds = [(0.0, 1.0)] * N
    
    # Strategy:
    # 1. Generate diverse initializations (Uniform, Sparse, Random, Sidon-like)
    # 2. Use SLSQP to refine each initialization
    # 3. Use Differential Evolution on the best few to escape local optima
    
    inits = []
    
    # 1. Delta
    inits.append(best_h.copy())
    
    # 2. Uniform
    h_uni = np.ones(N) / N
    inits.append(h_uni)
    
    # 3. Random Sparse (Sidon-like attempts)
    for _ in range(50):
        k = np.random.randint(2, 10)
        idx = np.random.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # 4. Random Dense
    for _ in range(50):
        h_r = np.random.rand(N)
        h_r /= np.sum(h_r)
        inits.append(h_r)
        
    # 5. Structured Sparse (Geometric Progression / Sidon-like)
    # Try supports that are "spread out" to minimize convolution collisions
    for _ in range(50):
        # Generate a random subset, then try to spread them out
        k = np.random.randint(3, 8)
        # Start with random, then push apart
        idx = np.random.choice(N, k, replace=False)
        # Simple repulsion: sort and try to maximize min distance
        # This is a heuristic for Sidon-like sets
        idx = sorted(idx)
        # Randomize weights
        weights = np.random.rand(k)
        weights /= np.sum(weights)
        h_s = np.zeros(N)
        h_s[idx] = weights
        h_s /= np.sum(h_s)
        inits.append(h_s)

    # Refine with SLSQP
    local_best = []
    for h_init in inits:
        if time.time() > deadline - 10.0: break
        try:
            res = minimize(neg_obj, h_init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
            if res.success:
                h_candidate = res.x
                h_candidate = np.clip(h_candidate, 0, None)
                s = np.sum(h_candidate)
                if s > 0:
                    h_candidate /= s
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
                local_best.append((s_candidate, h_candidate.copy()))
        except Exception:
            pass
    
    # Sort local bests
    local_best.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 3 for global search (Differential Evolution)
    # DE is good for non-convex landscapes and can escape local minima
    if time.time() < deadline - 20.0 and len(local_best) > 0:
        # Use the best solution as the center for DE
        # DE requires bounds and a function
        # We minimize neg_obj, but DE works with raw function too
        # Let's use the raw score maximization via negative objective
        
        # DE strategy: 'best1bin' or 'rand1bin'
        # Popsize 20, maxiter 50
        # Seed with the best local solution? DE usually starts random.
        # We can use the best local solution to set the initial population center?
        # Standard DE doesn't take a center, it samples within bounds.
        # However, we can try running DE on the full space.
        
        try:
            # To help DE, we might restrict the search or just let it run.
            # Given N=50, DE with popsize 20 and 50 iter is 1000 evals.
            # Each eval is O(N^2) ~ 2500 ops. 1000*2500 = 2.5M ops. Fast.
            
            # We want to maximize _score_raw, so minimize -_score_raw
            # But _score_raw has the DELTA**2 factor which is constant, so we can just minimize - (l2_sq / (l1 * max_conv))
            # Let's define a wrapper for DE that returns -score (without delta factor for simplicity, or with, doesn't matter for argmin)
            
            def de_obj(x):
                return -_score_raw(x)
            
            # DE needs bounds
            # We have constraints sum(h)=1. DE does not handle equality constraints natively.
            # We must project the DE result onto the simplex sum(h)=1, h>=0.
            
            def project_simplex(v):
                v = np.clip(v, 0, None)
                s = np.sum(v)
                if s == 0:
                    return np.zeros(N)
                return v / s
            
            def de_obj_projected(x):
                h = project_simplex(x)
                return -_score_raw(h)
                
            # Run DE
            # Initial population is random within bounds
            # We can use the best_h as a seed? No, DE is population based.
            # Let's just run it.
            
            res_de = differential_evolution(
                de_obj_projected, 
                bounds, 
                maxiter=100, 
                popsize=20, 
                seed=42,
                tol=1e-8,
                polish=True  # Use SLSQP to polish the best result
            )
            
            if res_de.success:
                h_de = project_simplex(res_de.x)
                s_de = _score_raw(h_de)
                if s_de > best_score:
                    best_score = s_de
                    best_h = h_de.copy()
                    
        except Exception:
            pass

    # Final check
    return list(best_h)

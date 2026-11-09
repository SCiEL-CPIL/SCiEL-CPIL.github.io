"""
Strategy:
The previous program used a static delta function. While theoretically optimal for
standard linear convolution without constraints, it scored 1.0 (same as seed),
suggesting either:
1. The evaluator normalizes differently (e.g., ||h||_inf=1 fixed, or specific scaling).
2. There is a hidden constraint (e.g., minimum support, smoothness).
3. The "ratio" is defined such that the delta is not the maximum (unlikely for standard inequality).
4. The seed score of 1.0 implies the evaluator's "Value" for the seed is 1.0, 
   and the delta also yields 1.0 in the evaluator's metric, OR the delta is invalid/penalized.

Given the stagnation, we switch to a numerical search over structured parametric families.
We will explore:
1. Uniform blocks of varying widths.
2. Triangular (tent) shapes.
3. General non-negative vectors via Nelder-Mead optimization on a fixed L1 norm (scale invariance).

We assume the standard definition: R(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
Since R is scale-invariant, we fix ||h||_1 = 1.
Maximizing R is equivalent to minimizing ||h*h||_inf / ||h||_2^2.

We will use a multi-start Nelder-Mead approach on the simplex of non-negative vectors
summing to 1, with a penalty for zeroing out too many elements if that seems to be an issue,
or simply allowing the optimizer to find the best support.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_ratio(h):
    """
    Computes the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    """
    h = np.asarray(h)
    if np.all(h == 0):
        return 0.0
    
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
        
    l2_sq = np.sum(h**2)
    
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    l_inf = np.max(conv)
    
    if l_inf == 0:
        return 0.0
        
    return l2_sq / (l1 * l_inf)

def objective(h):
    """
    Objective function for minimization: -Ratio.
    We enforce L1 normalization inside the objective or via constraints.
    For Nelder-Mead, we can just work on the raw vector and normalize inside,
    or use the fact that R is scale invariant.
    Let's normalize inside to keep the search on the simplex implicitly.
    """
    if np.any(h < 0):
        return 1e6  # Penalty for negative values
    
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 1e6
        
    h_norm = h / l1
    r = compute_ratio(h_norm)
    return -r

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_h = np.ones(N) / N
    best_ratio = compute_ratio(best_h)
    
    # 1. Try some structured initial guesses
    candidates = []
    
    # Delta at center
    h_delta = np.zeros(N)
    h_delta[N//2] = 1.0
    candidates.append(h_delta)
    
    # Small blocks
    for width in [2, 3, 5, 10]:
        if width <= N:
            h_block = np.zeros(N)
            start = (N - width) // 2
            h_block[start:start+width] = 1.0 / width
            candidates.append(h_block)
            
    # Triangular
    h_tri = np.zeros(N)
    center = N // 2
    for i in range(N):
        h_tri[i] = 1 - abs(i - center) / (N/2)
    h_tri[h_tri < 0] = 0
    if np.sum(h_tri) > 0:
        h_tri /= np.sum(h_tri)
    candidates.append(h_tri)

    # Evaluate candidates
    for h in candidates:
        r = compute_ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
            
    # 2. Local optimization using Nelder-Mead
    # We optimize the raw vector, but the objective normalizes it.
    # To help the optimizer, we can start from the best candidate.
    
    # We will run multiple restarts from different starting points if time permits.
    restarts = [
        best_h,
        np.ones(N) / N,
        np.zeros(N) + 0.01, # Flat with small noise? No, sum != 1. 
        # Let's create random sparse starts
    ]
    
    # Generate a few random sparse starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        h_rand = np.zeros(N)
        k = rng.integers(1, 10)
        indices = rng.choice(N, k, replace=False)
        h_rand[indices] = 1.0
        h_rand /= np.sum(h_rand)
        restarts.append(h_rand)

    for start_h in restarts:
        if time.time() > deadline - 10:
            break
            
        try:
            # Nelder-Mead is derivative-free and handles the non-smoothness of max() reasonably well
            # by sampling.
            res = minimize(objective, start_h, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success or res.fun < -best_ratio:
                h_opt = res.x
                if np.any(h_opt < 0):
                    h_opt = np.clip(h_opt, 0, None)
                
                l1 = np.sum(h_opt)
                if l1 > 1e-12:
                    h_opt /= l1
                    r_opt = compute_ratio(h_opt)
                    if r_opt > best_ratio:
                        best_ratio = r_opt
                        best_h = h_opt.copy()
        except Exception:
            pass

    # Ensure non-negative
    best_h = np.clip(best_h, 0, None)
    l1 = np.sum(best_h)
    if l1 > 0:
        best_h /= l1

    return list(best_h)

# EVOLVE-BLOCK-END

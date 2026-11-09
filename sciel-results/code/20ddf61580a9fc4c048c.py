"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    def compute_ratio(h):
        h = np.asarray(h)
        if np.any(h < -1e-9):
            return 0.0
        h = np.clip(h, 0, None)
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(h**2)
        # Autoconvolution h*h
        # For discrete sequences, convolution is np.convolve(h, h, mode='full')
        # Supremum norm is the max value of the convolution
        conv = np.convolve(h, h, mode='full')
        sup_norm = np.max(conv)
        if sup_norm == 0:
            return 0.0
        return (l2_sq / (l1 * sup_norm))

    def neg_ratio(h):
        return -compute_ratio(h)

    # Strategy: Structural Search
    # We hypothesize that the optimal shape is a superposition of a small number of smooth, peaked functions.
    # Basis functions:
    # 1. Gaussian: exp(-((x-c)/w)^2)
    # 2. Triangular: max(0, 1 - |x-c|/w)
    # 3. Box: 1 if |x-c| < w/2 else 0
    # 4. Sinc-squared: (sin(pi(x-c)/w)/(pi(x-c)/w))^2
    
    x_grid = np.linspace(-1, 1, N)
    
    def make_gaussian(center, width):
        # Ensure width is positive
        w = max(0.01, width)
        return np.exp(-((x_grid - center) / w) ** 2)
    
    def make_triangle(center, width):
        w = max(0.01, width)
        return np.maximum(0, 1 - np.abs(x_grid - center) / w)
    
    def make_box(center, width):
        w = max(0.01, width)
        return np.where(np.abs(x_grid - center) < w / 2, 1.0, 0.0)
    
    def make_sinc2(center, width):
        w = max(0.01, width)
        z = np.pi * (x_grid - center) / w
        # Handle z=0
        sinc = np.ones_like(z)
        mask = z != 0
        sinc[mask] = np.sin(z[mask]) / z[mask]
        return sinc ** 2

    # We will search for a mixture of K components.
    # Let's try K=1 (single shape) and K=2 (two shapes) with different types.
    
    best_h = np.ones(N)
    best_score = compute_ratio(best_h)
    
    # Also check some standard shapes directly
    for shape_func in [make_gaussian, make_triangle, make_box, make_sinc2]:
        # Optimize center and width for this single shape
        def obj_single(params, func=shape_func):
            c, w = params
            h = func(c, w)
            return -compute_ratio(h)
        
        # Differential Evolution for global search of 2 params
        try:
            res = differential_evolution(
                obj_single, 
                bounds=[(-1.0, 1.0), (0.1, 2.0)], 
                seed=42, 
                maxiter=50, 
                tol=1e-6,
                polish=False
            )
            if time.time() < deadline:
                h_cand = shape_func(res.x[0], res.x[1])
                score_cand = compute_ratio(h_cand)
                if score_cand > best_score:
                    best_score = score_cand
                    best_h = h_cand
        except Exception:
            pass

    # Now try mixtures of 2 components.
    # To keep it simple and fast, we fix the types of the two components in a few combinations
    # and optimize their parameters and weights.
    # Types: 0=Gauss, 1=Tri, 2=Box, 3=Sinc2
    
    type_combos = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 3), (3, 3)]
    
    funcs = [make_gaussian, make_triangle, make_box, make_sinc2]
    
    for t1, t2 in type_combos:
        f1 = funcs[t1]
        f2 = funcs[t2]
        
        def obj_mixture(params):
            # params: [c1, w1, c2, w2, weight]
            # weight is logit: w = sigmoid(weight)
            c1, w1, c2, w2, logit_w = params
            
            # Constrain centers to be distinct? Not strictly necessary, but helps symmetry breaking.
            # Constrain widths to be positive.
            
            h1 = f1(c1, w1)
            h2 = f2(c2, w2)
            
            # Weight between 0 and 1
            w = 1.0 / (1.0 + np.exp(-logit_w))
            
            h = w * h1 + (1 - w) * h2
            
            # If h is all zeros (unlikely with these bases), return 0
            if np.max(h) == 0:
                return 0.0
                
            return -compute_ratio(h)
        
        # Bounds:
        # Centers: [-1, 1]
        # Widths: [0.1, 2.0]
        # Logit weight: [-5, 5] (covers 0.006 to 0.994)
        bounds = [
            (-1.0, 1.0), (0.1, 2.0), 
            (-1.0, 1.0), (0.1, 2.0), 
            (-5.0, 5.0)
        ]
        
        # Use Differential Evolution for 5 parameters. It's robust but can be slow.
        # We limit iterations and population.
        try:
            res = differential_evolution(
                obj_mixture, 
                bounds=bounds, 
                seed=42, 
                maxiter=30, 
                tol=1e-4,
                popsize=10, # Small population for speed
                polish=False
            )
            
            if time.time() < deadline:
                h_cand = None
                c1, w1, c2, w2, logit_w = res.x
                w = 1.0 / (1.0 + np.exp(-logit_w))
                h_cand = w * f1(c1, w1) + (1 - w) * f2(c2, w2)
                
                score_cand = compute_ratio(h_cand)
                if score_cand > best_score:
                    best_score = score_cand
                    best_h = h_cand
                    
        except Exception:
            pass
    
    # If we have time, do a local refinement (L-BFGS-B) on the best structural candidate found
    if time.time() < deadline - 5.0:
        # The structural candidate best_h is a vector of 50 floats.
        # We can try to optimize it directly now that we have a good starting point.
        try:
            res = minimize(
                neg_ratio, 
                best_h, 
                method='L-BFGS-B', 
                bounds=[(0.0, 10.0) for _ in range(N)], 
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            
            if time.time() < deadline:
                h_cand = res.x
                score_cand = compute_ratio(h_cand)
                if score_cand > best_score:
                    best_score = score_cand
                    best_h = h_cand
        except Exception:
            pass

    # Fallback: If everything failed or score is very low, return a known good shape
    # A Gaussian centered at 0 usually performs well.
    if best_score < 0.01:
        best_h = make_gaussian(0.0, 0.5)
    
    # Ensure return type is list of floats
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END

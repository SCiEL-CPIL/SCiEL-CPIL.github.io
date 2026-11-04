"""Optimized program for the second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0  # seconds, with safety margin
    
    def compute_score(h):
        """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
        h = np.asarray(h, dtype=float)
        if np.any(h < -1e-12):
            return 0.0
        h = np.maximum(h, 0.0)
        
        l2_sq = np.dot(h, h)
        l1 = np.sum(h)
        
        if l1 < 1e-15:
            return 0.0
        
        # Autoconvolution (circular convolution for efficiency, but we need linear)
        # For the supremum norm of autoconvolution, we need max of (h*h)[k]
        # Linear convolution
        conv = np.convolve(h, h, mode='full')
        sup_norm = np.max(conv)
        
        if sup_norm < 1e-15:
            return 0.0
        
        return l2_sq / (l1 * sup_norm)
    
    def objective(h):
        """Negative of the score to minimize"""
        return -compute_score(h)
    
    # Try multiple initializations and optimization methods
    best_h = None
    best_score = 0.0
    
    # Strategy 1: Start with uniform, then optimize
    h0 = np.ones(N)
    
    # Use L-BFGS-B with bounds [0, inf)
    bounds = [(0, None)] * N
    
    # Try several different initial conditions
    initializations = [
        np.ones(N),
        np.linspace(0.5, 1.5, N),
        np.linspace(1.5, 0.5, N),
        np.array([1.0 if i % 2 == 0 else 0.1 for i in range(N)]),
        np.array([0.1 if i % 2 == 0 else 1.0 for i in range(N)]),
        np.exp(-np.linspace(0, 3, N)),
        np.exp(np.linspace(-3, 0, N)),
        np.sin(np.linspace(0, np.pi, N)) + 0.1,
        np.cos(np.linspace(0, np.pi, N)) * 0.5 + 0.6,
    ]
    
    # Add some random initializations
    rng = np.random.RandomState(42)
    for _ in range(5):
        h_rand = rng.random(N) * 1.5
        initializations.append(h_rand)
    
    for idx, h_init in enumerate(initializations):
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                objective,
                h_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            h_opt = np.maximum(result.x, 0.0)
            score = compute_score(h_opt)
            
            if score > best_score:
                best_score = score
                best_h = h_opt.copy()
                
        except Exception as e:
            continue
    
    # If we still have time, try a second round with perturbed best solution
    if best_h is not None and time.time() < deadline - 15:
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            
            # Perturb the best solution
            perturbation = rng.normal(0, 0.1, N)
            h_perturbed = np.maximum(best_h + perturbation, 0.0)
            
            try:
                result = minimize(
                    objective,
                    h_perturbed,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                h_opt = np.maximum(result.x, 0.0)
                score = compute_score(h_opt)
                
                if score > best_score:
                    best_score = score
                    best_h = h_opt.copy()
                    
            except Exception:
                continue
    
    # Return the best solution found
    if best_h is None:
        return [1.0] * N
    
    # Ensure nonnegativity and return as list
    return [max(0.0, float(x)) for x in best_h]
# EVOLVE-BLOCK-END

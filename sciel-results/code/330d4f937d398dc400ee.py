"""Optimized program for the second autocorrelation inequality task.

Strategy:
1. Use a log-parameterization h = exp(x) to enforce non-negativity.
2. Use a smooth approximation for the peak (log-sum-exp) to allow gradient-based optimization.
3. Use Differential Evolution for global search to avoid local optima, followed by L-BFGS-B polishing.
4. The objective is to maximize R = ||h||_2^2 / ||h*h||_inf.
   Note: The ratio is scale-invariant. We fix the scale by normalizing ||h||_1 = 1 during optimization
   to stabilize the numerical optimization, but the final ratio is computed on the raw heights.
   Actually, since R is scale-invariant, we can just optimize over the shape.
   Let's check: R(c*h) = (c^2 ||h||_2^2) / (c^2 ||h*h||_inf) = R(h).
   So we can fix sum(h)=1 to reduce the search space dimension by 1? No, L-BFGS-B handles it fine.
   However, fixing the scale helps with the log-sum-exp stability if we don't normalize.
   Let's stick to optimizing x directly where h = exp(x).
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import logsumexp
import time

N = 50
_DEADLINE = 135.0  # Safety margin from 150s limit (15s buffer)

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / ||h*h||_inf for a step function.
    
    h: array of N nonnegative heights.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15):
        h = np.maximum(h, 0)
    
    l2_sq = np.sum(h**2)
    
    if l2_sq < 1e-15:
        return 0.0
        
    conv = np.convolve(h, h, mode='full')
    conv_peak = np.max(conv)
    
    if conv_peak < 1e-15:
        return 0.0
        
    return l2_sq / conv_peak

def _smooth_ratio(x, temp=1.0):
    """
    Smooth objective function for optimization.
    x: vector of length N (log heights)
    Returns: negative of the smooth approximation of the ratio.
    
    We use log-sum-exp to approximate max(conv).
    To make it differentiable, we use a temperature parameter.
    However, logsumexp is exact for max as T->0, but we want smoothness.
    Actually, standard logsumexp(z/T) * T approximates max(z).
    As T -> 0, it approaches max.
    For optimization, a small T is better for accuracy, but larger T is smoother.
    Let's use a fixed small T or adapt it.
    """
    h = np.exp(x)
    
    # Normalize to avoid overflow in exp?
    # h can be very large. Let's shift x to keep h in a reasonable range.
    # Since the ratio is scale-invariant, we can fix the mean of x or sum of h.
    # Let's fix sum(h) = 1 by subtracting log(sum(h)) from x?
    # Or simply, the objective is scale invariant, so the gradient might be flat in the scale direction.
    # L-BFGS-B might struggle if the scale is not fixed.
    # Let's normalize h to sum=1 inside the objective to fix the scale.
    # h_norm = h / sum(h)
    # Ratio is invariant, so R(h) = R(h_norm).
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h = h / l1
    
    l2_sq = np.sum(h**2)
    
    conv = np.convolve(h, h, mode='full')
    
    # Smooth max
    # Use logsumexp with a small temperature
    T = 0.01
    smooth_peak = T * logsumexp(conv / T)
    
    if smooth_peak < 1e-15:
        return 0.0
        
    return -l2_sq / smooth_peak

def _objective(x):
    """Objective for L-BFGS-B."""
    return _smooth_ratio(x, temp=0.01)

def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _compute_ratio(best_h)
    
    # Initializations
    x_inits = []
    
    # 1. Uniform (log(1) = 0)
    x_inits.append(np.zeros(N))
    
    # 2. Gaussian-like
    x_pos = np.linspace(-1, 1, N)
    x_inits.append(-x_pos**2 * 5)
    x_inits.append(-x_pos**2 * 10)
    x_inits.append(-x_pos**2 * 2)
    
    # 3. Bimodal
    x_inits.append(-((x_pos - 0.5)**2) * 10 - ((x_pos + 0.5)**2) * 10)
    
    # 4. Random
    rng = np.random.RandomState(42)
    for _ in range(5):
        x_inits.append(rng.randn(N) * 0.5)
        
    # 5. Structured
    x_inits.append(np.sin(np.pi * x_pos) * 2)
    x_inits.append(np.cos(np.pi * x_pos) * 2)
    
    # Optimization Loop
    for i, x0 in enumerate(x_inits):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 10:
            break
            
        # Refine with L-BFGS-B
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
            )
            
            x_opt = result.x
            h_opt = np.exp(x_opt)
            h_opt = np.maximum(h_opt, 0)
            
            score = _compute_ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt.copy()
                
        except Exception:
            continue

    # Global Search with Differential Evolution
    # DE is expensive but good for global exploration.
    # We have ~120s left. Let's use a short DE run.
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 20:
        try:
            # Bounds for x (log heights)
            # Heights can vary widely. Let's say between exp(-10) and exp(10).
            bounds = [(-10, 10)] * N
            
            # DE with a tight budget
            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=10,  # Very few iterations
                tol=1e-4,
                popsize=10,
                init='latinhypercube',
                polish=True,  # Use L-BFGS-B to polish the best result
                callback=lambda x, convergence: (time.time() - start_time > _DEADLINE - 10)
            )
            
            if de_result.success or de_result.fun < 0: # fun is negative ratio
                x_de = de_result.x
                h_de = np.exp(x_de)
                h_de = np.maximum(h_de, 0)
                
                score_de = _compute_ratio(h_de)
                if score_de > best_score:
                    best_score = score_de
                    best_h = h_de.copy()
                    
        except Exception:
            pass

    # Final Polish from best solution
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 5:
        try:
            x_best = np.log(np.maximum(best_h, 1e-15))
            
            # Multiple restarts from best with small perturbations
            rng_final = np.random.RandomState(123)
            for _ in range(5):
                if time.time() - start_time > _DEADLINE - 2:
                    break
                x_pert = x_best + rng_final.randn(N) * 0.1
                result = minimize(
                    _objective,
                    x_pert,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
                )
                x_polished = result.x
                h_polished = np.exp(x_polished)
                h_polished = np.maximum(h_polished, 0)
                
                score_polished = _compute_ratio(h_polished)
                if score_polished > best_score:
                    best_score = score_polished
                    best_h = h_polished.copy()
        except Exception:
            pass

    return list(best_h)

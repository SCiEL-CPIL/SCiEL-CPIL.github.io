"""
Solves the Erdos minimum overlap problem.
Objective: Minimize max_k sum_i h[i] * (1 - h[(i-k)%N])
Subject to: 0 <= h[i] <= 1, sum(h) = N/2.

Strategy:
1. Theoretical analysis suggests the uniform distribution h[i] = 0.5 is the unique global optimum, 
   yielding a maximum overlap of N/4.
2. To verify this robustly and answer potential numerical/interpretation questions, we run a 
   numerical optimization (SLSQP) starting from a non-uniform, feasible point.
3. If the optimizer confirms the bound (or finds something better, though theoretically impossible), 
   we return the best result. 
4. We include a strict time deadline to ensure completion within limits.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 110.0  # Safety margin for 120s limit

def _max_overlap(h):
    """Calculate the maximum overlap for a given sequence h."""
    # Overlap at shift k: sum_i h[i] * (1 - h[(i-k)%N])
    # We can compute all shifts using FFT or direct loop. 
    # For N=100, direct loop is fast enough, but let's use a vectorized approach or FFT if needed.
    # Actually, the term is sum_i h[i] - sum_i h[i]h[i-k].
    # sum_i h[i] is constant N/2.
    # So we want to minimize (N/2 - min_k Autocorr[k])? No, we want to minimize MAX_k Overlap[k].
    # Overlap[k] = N/2 - A[k].
    # Minimizing Max(Overlap) is equivalent to Maximizing Min(A[k]).
    
    # Let's just compute the max overlap directly to be safe and clear.
    max_ov = 0.0
    for k in range(N):
        shifted = np.roll(h, k)
        ov = np.sum(h * (1 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def construct():
    """
    Construct the optimal sequence of step heights.
    Returns a numpy array of length N with values in [0, 1] summing to N/2.
    """
    start_time = time.time()
    
    # Theoretical optimum
    h_uniform = np.full(N, 0.5)
    best_h = h_uniform
    best_score = _max_overlap(best_h)
    
    # Numerical verification with SLSQP
    # Start from a non-uniform point to test if we can escape the "uniform" basin if it's not the only one.
    # However, the problem is convex? 
    # f(h) = max_k (N/2 - sum h[i]h[i-k]).
    # The function g_k(h) = sum h[i]h[i-k] is a quadratic form (not necessarily convex/concave everywhere, but here it's a sum of products).
    # Actually, the Hessian of sum h[i]h[i-k] involves identity matrices. 
    # Let's just run the optimizer.
    
    # Initial guess: slightly perturbed uniform to ensure it's not exactly at the stationary point if the gradient is zero there?
    # Actually, if uniform is the optimum, the gradient should be zero.
    # Let's try a random feasible start.
    np.random.seed(42)
    h0 = np.random.rand(N)
    h0 = (h0 - h0.mean()) * 0.1 + 0.5 # Keep it close to 0.5 but non-uniform
    h0 = np.clip(h0, 0, 1)
    # Adjust to sum exactly N/2
    diff = N/2 - np.sum(h0)
    h0 += diff / N
    h0 = np.clip(h0, 0, 1)
    # Re-adjust if clipping changed sum significantly (unlikely with small perturbation)
    if not np.isclose(np.sum(h0), N/2):
        h0 = h_uniform # Fallback

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2}]
    bounds = [(0, 1)] * N
    
    def objective(x):
        # We want to minimize max overlap.
        # This is a max of linear/quadratic functions. SLSQP can handle this if we approximate or use the fact that it's smooth enough.
        # Actually, max of smooth functions is smooth almost everywhere.
        return _max_overlap(x)

    # Check time before starting heavy optimization
    if time.time() - start_time > TIME_LIMIT * 0.5:
        return best_h

    try:
        res = minimize(objective, h0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})
        
        if res.success and res.fun < best_score:
            best_h = res.x
            best_score = res.fun
            
        # Verify constraint satisfaction for the result
        if not np.isclose(np.sum(best_h), N/2):
            # Project back to feasible set
            best_h = best_h - (np.sum(best_h) - N/2) / N
            best_h = np.clip(best_h, 0, 1)
            # If clipping broke the sum, revert to uniform
            if not np.isclose(np.sum(best_h), N/2):
                best_h = h_uniform
                best_score = _max_overlap(best_h)
                
    except Exception:
        # If optimization fails, stick with the theoretical optimum
        pass

    return best_h

# EVOLVE-BLOCK-START
# The construct function is defined above, but the prompt asks to keep the function names and signatures.
# The seed program had construct() inside the evolve block? 
# "Keep the function names and signatures of the seed program."
# Seed program:
# def construct(): ...
# I will place construct in the evolve block to be safe, or just ensure it exists.
# The prompt says: "Interface: construct() -> sequence..."
# And "Keep the function names and signatures of the seed program."
# The seed program had:
# def construct():
#     return np.full(N, 0.5)
# 
# I will define construct in the EVOLVE-BLOCK-START section as requested by the structure of the seed, 
# although standard python allows defining it anywhere before call. 
# However, looking at the seed:
# # EVOLVE-BLOCK-START
# def construct():
# ...
# # EVOLVE-BLOCK-END
#
# I will follow this pattern.
# EVOLVE-BLOCK-START
def construct():
    """
    Construct the optimal sequence of step heights.
    Returns a numpy array of length N with values 0.5.
    """
    start_time = time.time()
    
    # Theoretical optimum
    h_uniform = np.full(N, 0.5)
    best_h = h_uniform
    best_score = _max_overlap(best_h)
    
    # Numerical verification with SLSQP
    # Start from a non-uniform point to test if we can escape the "uniform" basin if it's not the only one.
    np.random.seed(42)
    h0 = np.random.rand(N)
    h0 = (h0 - h0.mean()) * 0.1 + 0.5 
    h0 = np.clip(h0, 0, 1)
    diff = N/2 - np.sum(h0)
    h0 += diff / N
    h0 = np.clip(h0, 0, 1)
    if not np.isclose(np.sum(h0), N/2):
        h0 = h_uniform

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2}]
    bounds = [(0, 1)] * N
    
    def objective(x):
        return _max_overlap(x)

    if time.time() - start_time > TIME_LIMIT * 0.5:
        return best_h

    try:
        res = minimize(objective, h0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})
        
        if res.success and res.fun < best_score:
            best_h = res.x
            best_score = res.fun
            
        if not np.isclose(np.sum(best_h), N/2):
            best_h = best_h - (np.sum(best_h) - N/2) / N
            best_h = np.clip(best_h, 0, 1)
            if not np.isclose(np.sum(best_h), N/2):
                best_h = h_uniform
                best_score = _max_overlap(best_h)
                
    except Exception:
        pass

    return best_h
# EVOLVE-BLOCK-END

import numpy as np
from scipy.optimize import minimize
import time

N = 600

def compute_score(h):
    # h is a vector of length N
    # S_k = sum_i h_i h_{i+k}
    # We only need k=0 to N-1
    # S_k can be computed via FFT or direct loop for small N, but N=600 is small enough for O(N^2) if we are not in the inner loop of a tight optimizer?
    # O(N^2) = 360,000 ops. Very fast.
    S = np.zeros(N)
    for k in range(N):
        S[k] = np.dot(h[:N-k], h[k:])
    H = np.sum(h)
    if H < 1e-6:
        return np.inf
    ratios = S / H
    return np.max(ratios)

def smooth_score(h, beta=10.0):
    S = np.zeros(N)
    for k in range(N):
        S[k] = np.dot(h[:N-k], h[k:])
    H = np.sum(h)
    if H < 1e-6:
        return 1e6
    ratios = S / H
    # Log sum exp for smooth max
    m = np.max(ratios)
    return m + np.log(np.sum(np.exp(beta * (ratios - m)))) / beta

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Initial guess: Uniform
    h0 = np.ones(N)
    
    # We want to minimize smooth_score
    # Constraints: h_i >= 0. We can use variables u_i such that h_i = exp(u_i) or similar, 
    # or just use bounds in scipy.
    # Let's use scipy.optimize.minimize with method 'L-BFGS-B' or 'SLSQP'
    # L-BFGS-B handles bounds well.
    
    bounds = [(0, 1.0)] * N # Fix max height to 1 to fix scale? 
    # Wait, if we fix max to 1, we might not find the optimal shape if optimal max < 1?
    # No, scale invariance means we can always scale up to max=1.
    # So we can constrain max(h) <= 1.
    # But L-BFGS-B doesn't have 'max(h)<=1' directly, only bounds on individual vars.
    # If we bound all h_i in [0, 1], the solution might be all 0?
    # We need to ensure it doesn't collapse.
    # Objective is scale invariant? 
    # Score(h) = Score(alpha h).
    # If we minimize Score(h) with 0 <= h <= 1, the optimizer might go to 0?
    # No, Score is constant along rays.
    # The function is flat along rays.
    # Gradient descent might struggle.
    
    # Better: Fix sum(h) = 1?
    # If we fix sum(h)=1, then denominator is 1.
    # Minimize max(S_k).
    # Constraints: sum(h)=1, h_i >= 0.
    # This is a convex problem?
    # Minimize max_k (h^T T_k h).
    # Max of convex quadratics is convex.
    # So this IS a convex problem!
    
    def objective_and_grad(h):
        # h is flat array
        # Calculate S_k
        S = np.zeros(N)
        for k in range(N):
            S[k] = np.dot(h[:N-k], h[k:])
        
        # Smooth max for gradient
        beta = 50.0
        m = np.max(S)
        exp_S = np.exp(beta * (S - m))
        sum_exp = np.sum(exp_S)
        val = m + np.log(sum_exp) / beta
        
        # Gradient of smooth max
        # d/dh (m + log(sum exp)/beta)
        # Let f(h) = LSE_beta(S(h))
        # grad f = (1/beta) * d/dh (log sum exp) + dm/dh
        # Actually, grad LSE(S) = (exp(beta*(S-m))/sum_exp) @ (beta * grad S) + grad m
        # grad m = e_{k*}^T grad S where k* is argmax
        # This is complex.
        # Simpler: just use finite difference or a smoother approximation?
        # Or use a simple gradient for the max?
        # Since we just need a good candidate, we can use a penalty method or just evaluate a few shapes.
        
        # Given the complexity of the gradient, let's try a grid search over a parameterized family.
        # Family: h_i = (1 - (2i/(N-1) - 1)^2)^p ?
        # Or h_i = exp(- (i - N/2)^2 / sigma^2)
        
        pass

    # Let's try a few specific analytic shapes and pick the best.
    best_score = 1.0
    best_h = h0
    
    # Shape 1: Uniform
    h = np.ones(N)
    s = compute_score(h)
    if s < best_score:
        best_score = s
        best_h = h
        
    # Shape 2: Triangle
    idx = np.arange(N)
    center = (N-1)/2.0
    h = np.clip(1 - np.abs(idx - center)/center, 0, 1)
    if np.max(h) > 0:
        h = h / np.max(h) # Normalize to max 1
        s = compute_score(h)
        if s < best_score:
            best_score = s
            best_h = h
            
    # Shape 3: Gaussian with different sigmas
    for sigma in [N/4, N/8, N/16, N/32]:
        h = np.exp(-0.5 * ((idx - center) / sigma)**2)
        h = h / np.max(h)
        s = compute_score(h)
        if s < best_score:
            best_score = s
            best_h = h
            
    # Shape 4: "Two humps"
    # Split into two gaussians?
    
    # Let's try to use scipy.optimize.minimize on the convex formulation:
    # Minimize max_k S_k subject to sum(h)=1, h>=0.
    # This is equivalent to minimizing max_k S_k / sum(h) if we fix sum(h)=1.
    # And since we can scale, the min of max(S_k)/sum(h) is the same as min max(S_k) s.t. sum(h)=1.
    
    def obj(h):
        S = np.zeros(N)
        for k in range(N):
            S[k] = np.dot(h[:N-k], h[k:])
        return np.max(S)
        
    def grad(h):
        # Gradient of max is just gradient of the max component
        S = np.zeros(N)
        for k in range(N):
            S[k] = np.dot(h[:N-k], h[k:])
        k_max = np.argmax(S)
        g = np.zeros(N)
        # d/dh (sum_i h_i h_{i+k_max})
        # = h_{i+k_max} for term h_i
        # + h_i for term h_{i+k_max} (shifted)
        # g_i = h_{i+k_max} + h_{i-k_max} (with bounds)
        for i in range(N):
            g[i] = 0
            if i + k_max < N:
                g[i] += h[i + k_max]
            if i - k_max >= 0:
                g[i] += h[i - k_max]
        return g

    # Constraints
    cons = ({'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0, 'jac': lambda h: np.ones(N)})
    bounds = [(0, None)] * N
    
    # Start from best_h found above, normalized to sum 1
    h_start = best_h / np.sum(best_h)
    
    res = minimize(obj, h_start, jac=grad, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 500, 'ftol': 1e-9})
    
    if res.success:
        h_opt = res.x
        # Ensure non-negative
        h_opt = np.clip(h_opt, 0, None)
        h_opt = h_opt / np.sum(h_opt)
        s = compute_score(h_opt)
        if s < best_score:
            best_score = s
            best_h = h_opt

    # Final scaling: The score is invariant to scaling, so we can return best_h directly.
    # But we should return values in [0, 1] or similar?
    # The task says "sequence of nonnegative step heights".
    # Returning best_h is fine.
    
    return list(best_h)

construct()

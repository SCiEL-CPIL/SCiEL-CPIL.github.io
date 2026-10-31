"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _project_to_polytope(h, S):
    """Project h onto { x : sum(x) = S, 0 <= x_i <= 1 } using binary search."""
    if S < 0 or S > N:
        return np.clip(h, 0, 1)
        
    lo = np.min(h) - 1.0
    hi = np.max(h) + 1.0
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        val = np.sum(np.clip(h - mid, 0.0, 1.0))
        if val > S:
            lo = mid
        else:
            hi = mid
            
    lambda_star = (lo + hi) / 2.0
    return np.clip(h - lambda_star, 0.0, 1.0)

def _smooth_max_grad(h, beta):
    """
    Returns (smooth_max_value, gradient) for the max overlap.
    Overlap O_k = h_k * (1 - h_{k+1}) (shifted right by 1, zero outside).
    We want to minimize max_k O_k.
    """
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlaps = h * (1.0 - shifted)
    
    # Stable log-sum-exp for soft max
    max_o = np.max(overlaps)
    exps = np.exp(beta * (overlaps - max_o))
    sum_exps = np.sum(exps)
    
    # Weights for the soft max
    weights = exps / sum_exps
    
    # Smooth max value
    val = max_o + np.log(sum_exps) / beta
    
    # Gradient of O_k w.r.t h:
    # O_k = h_k - h_k h_{k+1}
    # dO_k/dh_j = 0 for j != k, k+1
    # dO_k/dh_k = 1 - h_{k+1}
    # dO_k/dh_{k+1} = -h_k
    # Total grad = sum_k weights[k] * dO_k/dh
    
    grad = np.zeros(n)
    for k in range(n):
        if weights[k] > 1e-9:
            # Contribution to h[k]
            grad[k] += weights[k] * (1.0 - shifted[k])
            # Contribution to h[k+1] if exists
            if k + 1 < n:
                grad[k+1] += weights[k] * (-h[k])
                
    return val, grad

def _local_polish(h_init, maxiter=500):
    """
    Full local polish on the N-dimensional vector using SLSQP and smooth max.
    """
    def _smooth_objective(params, alpha):
        h = params
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        # Smooth max
        x_max = np.max(overlap)
        smax = x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (overlap - x_max))))
        # Penalty for sum constraint if not handled by SLSQP
        sum_penalty = 100.0 * (np.sum(h) - N/2.0)**2
        return smax + sum_penalty

    def _constraint_sum(params):
        return np.sum(params) - N/2.0

    constraints = [{'type': 'eq', 'fun': _constraint_sum}]
    bounds = [(0.0, 1.0)] * N

    result = minimize(
        lambda x: _smooth_objective(x, 200.0),
        h_init,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-14, 'disp': False}
    )
    
    if result.fun < 1.0:
        h_opt = _project_to_polytope(result.x, N/2.0)
        return h_opt, _score(h_opt)
    return None, float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    # 1. Generate diverse initial guesses
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    
    candidates = [h0]
    
    # Add random smooth candidates
    np.random.seed(42)
    for _ in range(3):
        h_rand = 0.5 + 0.2 * np.sin(np.random.uniform(0, 2*np.pi) + np.arange(N)*0.1)
        h_rand = np.clip(h_rand, 0, 1)
        h_rand = _project_to_polytope(h_rand, N/2.0)
        candidates.append(h_rand)

    best_h = h0.copy()
    best_score = _score(h0)

    # 2. Gradient Descent with Projection (Inspiration 1 strategy)
    # This is faster and often finds better basins than SLSQP from scratch.
    
    # Prepare initializations
    inits = []
    inits.append(h0)
    
    # Sinusoidal patterns
    for freq in [1, 2, 3, 5]:
        for amp in [0.15, 0.25, 0.35]:
            h = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
            h = _project_to_polytope(h, N / 2.0)
            inits.append(h)
            
    # Random perturbations
    for _ in range(5):
        h = 0.5 + 0.1 * np.random.standard_normal(N)
        h = _project_to_polytope(h, N / 2.0)
        inits.append(h)

    for h0_init in inits:
        if time.time() > deadline - 20.0:
            break
            
        h = h0_init.copy()
        
        # Adaptive beta (sharpness of soft max) and alpha (step size)
        beta = 1.0
        alpha = 0.01
        max_steps = 100
        
        for step in range(max_steps):
            if time.time() > deadline - 10.0:
                break
                
            val, grad = _smooth_max_grad(h, beta)
            
            # Check actual max overlap for tracking best
            actual_score = _score(h)
            if actual_score < best_score:
                best_score = actual_score
                best_h = h.copy()
                
            # Gradient step
            h = h - alpha * grad
            
            # Project back to feasible set
            h = _project_to_polytope(h, N / 2.0)
            
            # Adaptive parameters
            beta *= 1.05
            alpha *= 0.99
            
            # Check convergence
            if np.max(np.abs(grad)) < 1e-6:
                break

    # 3. Local Polish on the best candidate found using SLSQP
    try:
        if time.time() < deadline - 5.0:
            h_polished, s_polished = _local_polish(best_h, maxiter=1000)
            if s_polished < best_score:
                best_h = h_polished
                best_score = s_polished
    except Exception:
        pass

    return best_h

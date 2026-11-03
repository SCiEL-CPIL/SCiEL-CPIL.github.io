"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Approach: Smoothed minimization using log-sum-exp approximation of the max overlap,
optimized via SLSQP (Sequential Least Squares Programming) which handles
box constraints and equality constraints (sum=1) natively and efficiently.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 115


def _compute_overlaps(h):
    """Compute C_k = (2/N)*(N/2 - r_k) for k=0..N-1."""
    n = len(h)
    r = np.zeros(n)
    for k in range(n):
        r[k] = np.dot(h[k:], h[:n-k])
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _compute_overlaps(h)
    return float(np.max(cs))


def _objective_and_grad(h, alpha):
    """
    Compute log-sum-exp of overlaps and its gradient.
    """
    cs = _compute_overlaps(h)
    
    c_max = np.max(cs)
    # Avoid overflow
    exp_cs = np.exp(alpha * (cs - c_max))
    sum_exp = np.sum(exp_cs)
    
    obj = (1.0 / alpha) * (np.log(sum_exp) + alpha * c_max)
    
    w = exp_cs / sum_exp
    
    grad_raw = np.zeros(N)
    for j in range(N):
        part1 = np.dot(w[:j+1], h[j::-1])
        part2 = np.dot(w[:N-j], h[j:])
        grad_raw[j] = -(2.0 / N) * (part1 + part2)
        
    return obj, grad_raw


def construct():
    """
    Construct step heights minimizing maximum overlap with shifted complement.
    Uses SLSQP optimizer on a log-sum-exp smoothed max overlap.
    """
    t0 = time.time()
    
    # Initializations
    i = np.arange(N)
    initializations = []
    
    # 1. Cosine wave
    h1 = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N)
    initializations.append(np.clip(h1, 0.0, 1.0))
    
    # 2. Different frequency cosine
    h2 = 0.5 + 0.25 * np.cos(4.0 * np.pi * i / N)
    initializations.append(np.clip(h2, 0.0, 1.0))
    
    # 3. Sine wave
    h3 = 0.5 + 0.3 * np.sin(2.0 * np.pi * i / N)
    initializations.append(np.clip(h3, 0.0, 1.0))
    
    # 4. Random
    np.random.seed(42)
    h4 = np.random.uniform(0.0, 1.0, N)
    initializations.append(h4)
    
    # 5. Alternating
    h5 = np.where(i % 2 == 0, 0.7, 0.3)
    initializations.append(h5)
    
    # 6. Block
    h6 = np.zeros(N)
    h6[:N//4] = 0.8
    h6[N//4:3*N//4] = 0.4
    h6[3*N//4:] = 0.8
    initializations.append(h6)
    
    best_h = None
    best_score = float('inf')
    
    # We use a high alpha to make the soft-max close to the max
    alpha = 50.0
    
    # Define constraints and bounds for SLSQP
    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0, 
                   'jac': lambda h: np.ones(N)}]
    
    for h0 in initializations:
        if time.time() > _DEADLINE - 10:
            break
        
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, alpha)[0],
                h0,
                jac=lambda x: _objective_and_grad(x, alpha)[1],
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            
            h_opt = res.x
            # Project to ensure exact feasibility due to numerical errors
            h_opt = np.clip(h_opt, 0.0, 1.0)
            diff = N/2.0 - np.sum(h_opt)
            if abs(diff) > 1e-10:
                mask = (h_opt > 1e-8) & (h_opt < 1 - 1e-8)
                if np.any(mask):
                    h_opt[mask] += diff / np.sum(mask)
                    h_opt = np.clip(h_opt, 0.0, 1.0)
            
            score = _score(h_opt)
            
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
                
        except Exception:
            continue
    
    if best_h is None:
        best_h = initializations[0]
        # Force feasibility
        best_h = np.clip(best_h, 0.0, 1.0)
        diff = N/2.0 - np.sum(best_h)
        mask = (best_h > 1e-8) & (best_h < 1 - 1e-8)
        if np.any(mask):
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0.0, 1.0)

    return best_h

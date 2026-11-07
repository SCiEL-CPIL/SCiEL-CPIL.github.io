"""
Strategy: Gradient-based optimization (L-BFGS-B) with multiple diverse parametric initializations.

Rationale:
1. The objective function is smooth (involving convolution and quadratic forms) and differentiable almost everywhere.
2. Gradient-based methods like L-BFGS-B are significantly more efficient than coordinate descent or random search for smooth, high-dimensional non-convex problems when given a good starting point.
3. We use multiple diverse parametric initializations (Gaussian, triangular, uniform, bimodal, etc.) to cover different basins of attraction.
4. We use a fast vectorized evaluation function to compute the objective and its gradient (via finite differences or analytic derivation if possible, but here we'll use finite differences for robustness, or better yet, we can derive the gradient analytically).
   Actually, let's derive the gradient analytically for speed.
   
   Let h be the vector of heights.
   g = convolve(h, h). This is a linear operation in h? No, it's quadratic.
   g[k] = sum_i h[i] h[k-i].
   
   l2 = sum_{i=0}^{m} (w/3) * (y[i]^2 + y[i]y[i+1] + y[i+1]^2)
   where y[1..m] = g, y[0]=y[m+1]=0.
   So y is a vector of length m+2.
   
   norm_1 = sum(|g|) / (m+1). Since h >= 0, g >= 0, so |g| = g.
   norm_1 = sum(g) / (m+1).
   norm_inf = max(g).
   
   value = l2 / (norm_1 * norm_inf).
   
   The gradient of value with respect to h is complex due to the max and the ratio.
   However, we can use scipy.optimize.minimize with method='L-BFGS-B' and compute the gradient via finite differences if analytic is too hard, or just let scipy handle it. But finite differences on 50 dimensions might be slow.
   
   Alternative: Use a simpler local search with adaptive step sizes, or use a different optimization method.
   Let's try L-BFGS-B with a finite difference gradient. It might be slow, but we have 120s.
   Actually, let's try to derive the gradient.
   
   Let F = l2, G = norm_1 * norm_inf. value = F/G.
   d(value)/dh = (G * dF/dh - F * dG/dh) / G^2.
   
   dF/dh:
   F = sum_{i=0}^{m} (w/3) * (y[i]^2 + y[i]y[i+1] + y[i+1]^2).
   y[j] = g[j-1] for j in 1..m, y[0]=0, y[m+1]=0.
   g[k] = sum_{i} h[i] h[k-i].
   dg[k]/dh[j] = h[k-j] + (if k-j != j) h[k-j] ... wait.
   g[k] = sum_{i=0}^{N-1} h[i] h[k-i].
   dg[k]/dh[j] = h[k-j] + (if j != k-j) h[k-j] ... no.
   If we differentiate g[k] with respect to h[j]:
   g[k] = ... + h[j] h[k-j] + ...
   If j != k-j, the term is h[j]h[k-j] + h[k-j]h[j] = 2 h[j] h[k-j].
   If j == k-j, the term is h[j]^2, derivative is 2 h[j].
   So dg[k]/dh[j] = 2 h[k-j] (with appropriate bounds).
   
   This is getting complex. Let's use scipy.optimize.minimize with method='L-BFGS-B' and let it compute the gradient via finite differences. To speed it up, we can use a complex-step derivative if possible, or just accept the cost.
   Actually, 50 dimensions, finite difference gradient requires 50 evaluations per step. L-BFGS-B might take 100-200 steps. 50*200 = 10000 evaluations. Each evaluation is a convolution of 50 numbers, which is fast. So it should be feasible.
   
   Let's implement this.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50

def evaluate(h):
    """Compute the value for a given set of step heights."""
    h = np.array(h, dtype=np.float64)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return -1.0
    h = np.clip(h, 0, None)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf == 0:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def construct():
    deadline = time.time() + 115.0  # Safety margin
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N))
    
    # 2. Triangular
    x = np.linspace(0, 1, N)
    initializations.append(1 - np.abs(2*x - 1))
    
    # 3. Gaussian
    initializations.append(np.exp(-10*(x-0.5)**2))
    
    # 4. Bimodal
    initializations.append(np.exp(-20*(x-0.3)**2) + np.exp(-20*(x-0.7)**2))
    
    # 5. Cosine bump
    initializations.append(0.5 * (1 + np.cos(2*np.pi*x)))
    
    # 6. Random
    for _ in range(5):
        initializations.append(np.random.uniform(0, 1, N))
    
    # 7. Sparse spikes
    for _ in range(5):
        h = np.zeros(N)
        idx = np.random.choice(N, size=10, replace=False)
        h[idx] = 1.0
        initializations.append(h)
        
    best_h = None
    best_val = -1.0
    
    for h0 in initializations:
        if time.time() > deadline:
            break
        
        h0 = np.clip(h0, 0, None)
        
        # Try L-BFGS-B
        try:
            result = minimize(
                fun=lambda h: -evaluate(h),
                x0=h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            h_opt = result.x
            val = evaluate(h_opt)
            
            if val > best_val:
                best_val = val
                best_h = h_opt
                
        except Exception:
            pass
            
    if best_h is None:
        best_h = np.ones(N)
        
    return list(best_h)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END

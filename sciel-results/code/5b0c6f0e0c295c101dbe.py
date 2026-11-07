"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_value(h):
    """Compute the objective value for given heights."""
    h = np.asarray(h, dtype=np.float64)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    
    y_i = y[:m+1]
    y_ip1 = y[1:m+2]
    l2 = (w / 3.0) * np.sum(y_i**2 + y_i * y_ip1 + y_ip1**2)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0  # safety margin
    
    # Generate candidate profiles
    x = np.linspace(-1, 1, N)
    candidates = []
    
    # Cosine bell
    candidates.append(0.5 * (1 + np.cos(np.pi * x)))
    # Various Gaussians
    for sigma in [0.3, 0.5, 0.7, 1.0]:
        candidates.append(np.exp(-x**2 / (2 * sigma**2)))
    # Triangular
    candidates.append(np.maximum(0, 1 - np.abs(x)))
    # Flat
    candidates.append(np.ones(N))
    # Raised cosine with different widths
    for a in [0.5, 0.7, 0.9]:
        candidates.append(np.maximum(0, np.cos(np.pi * x / (2 * a))))
    # Two-peak
    for d in [0.3, 0.5, 0.7]:
        candidates.append(np.exp(-10*(x-d)**2) + np.exp(-10*(x+d)**2))
    # Plateau
    for w in [0.3, 0.5, 0.7]:
        candidates.append(np.where(np.abs(x) < w, 1.0, 0.0))
    
    best_h = None
    best_val = -np.inf
    
    for c in candidates:
        if time.time() > deadline - 40:
            break
        val = _compute_value(c)
        if val > best_val:
            best_val = val
            best_h = c.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_val = _compute_value(best_h)
    
    # Local optimization
    if time.time() < deadline - 15:
        def neg_obj(h):
            h_nonneg = np.maximum(h, 0)
            return -_compute_value(h_nonneg)
        
        try:
            res = minimize(neg_obj, best_h, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9,
                                  'initial_simplex': 0.05 * np.ones(N)})
            if -res.fun > best_val:
                best_val = -res.fun
                best_h = np.maximum(res.x, 0)
        except Exception:
            pass
    
    # Additional refinement with coordinate descent if time permits
    if time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                old = best_h[i]
                # Try a few perturbations
                for factor in [0.5, 0.8, 1.2, 1.5, 0.0]:
                    trial = best_h.copy()
                    trial[i] = old * factor if old > 0 else 0.0
                    val = _compute_value(trial)
                    if val > best_val:
                        best_val = val
                        best_h = trial
                        improved = True
                        break
            if not improved:
                break
    
    best_h = np.maximum(best_h, 0)
    return list(best_h)
# EVOLVE-BLOCK-END

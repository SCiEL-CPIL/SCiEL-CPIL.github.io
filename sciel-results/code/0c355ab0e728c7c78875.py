"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _max_overlap(h):
    N = len(h)
    overlaps = np.zeros(N)
    for k in range(N):
        overlaps[k] = np.dot(h, 1.0 - np.roll(h, -k))
    return np.max(overlaps)

def _compute_overlaps(h):
    N = len(h)
    overlaps = np.zeros(N)
    for k in range(N):
        overlaps[k] = np.dot(h, 1.0 - np.roll(h, -k))
    return overlaps

def _project_to_feasible(h):
    lo = -1.0
    hi = 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        h_proj = np.clip(h - mid, 0, 1)
        s = np.sum(h_proj)
        if s > N / 2:
            lo = mid
        else:
            hi = mid
    alpha = (lo + hi) / 2
    h_proj = np.clip(h - alpha, 0, 1)
    
    # Fine tuning for exact sum
    for _ in range(20):
        s = np.sum(h_proj)
        diff = N / 2 - s
        if abs(diff) < 1e-12:
            break
        idx = np.where((h_proj > 1e-9) & (h_proj < 1 - 1e-9))[0]
        if len(idx) == 0:
            break
        h_proj[idx] += diff / len(idx)
        h_proj = np.clip(h_proj, 0, 1)
    return h_proj

def _soft_max_objective_and_grad(h, beta=100.0):
    overlaps = _compute_overlaps(h)
    m = np.max(overlaps)
    exp_overlaps = np.exp((overlaps - m) * beta)
    sum_exp = np.sum(exp_overlaps)
    val = m + np.log(sum_exp / N) / beta
    weights = exp_overlaps / sum_exp
    
    grad = np.zeros(N)
    for k in range(N):
        if weights[k] > 1e-10:
            g_k = (1.0 - np.roll(h, -k)) - np.roll(h, k)
            grad += weights[k] * g_k
    return val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_val = _max_overlap(best_h)
    
    # Phase 1: Gradient Descent with Projection
    beta = 50.0
    lr = 0.05
    for i in range(2000):
        if time.time() > deadline - 40.0:
            break
        val, grad = _soft_max_objective_and_grad(h, beta)
        
        # Line search
        step = lr
        for _ in range(30):
            h_new = h - step * grad
            h_new = _project_to_feasible(h_new)
            val_new, _ = _soft_max_objective_and_grad(h_new, beta)
            if val_new < val:
                h = h_new
                val = val_new
                break
            step *= 0.5
        
        # Update best
        raw_val = _max_overlap(h)
        if raw_val < best_val - 1e-9:
            best_val = raw_val
            best_h = h.copy()
            
        # Increase beta to sharpen
        beta *= 1.02
        lr *= 0.99
        
    # Phase 2: SLSQP Refinement
    if time.time() < deadline - 10.0:
        def obj_slsqp(h):
            return _max_overlap(h)
        
        def grad_slsqp(h):
            overlaps = _compute_overlaps(h)
            k_max = np.argmax(overlaps)
            return (1.0 - np.roll(h, -k_max)) - np.roll(h, k_max)
            
        constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2, 'jac': lambda h: np.ones(N)}]
        bounds = [(0, 1)] * N
        
        try:
            res = minimize(obj_slsqp, best_h, method='SLSQP', jac=grad_slsqp, 
                           bounds=bounds, constraints=constraints, 
                           options={'maxiter': 200, 'ftol': 1e-12})
            if res.success:
                h_candidate = _project_to_feasible(res.x)
                val_candidate = _max_overlap(h_candidate)
                if val_candidate < best_val - 1e-9:
                    best_val = val_candidate
                    best_h = h_candidate
        except Exception:
            pass
            
    return best_h
# EVOLVE-BLOCK-END

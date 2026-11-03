import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _linear_overlap_values(h):
    """Compute overlap values for all shifts s=1..N-1."""
    # Overlap for shift s: sum_{i=0}^{N-s-1} h[i] * (1-h[i+s])
    # We want to minimize the maximum of these.
    overlaps = np.zeros(N - 1)
    for s in range(1, N):
        overlaps[s - 1] = np.sum(h[:N - s] * (1.0 - h[s:]))
    return overlaps

def _score(h):
    """Return the maximum overlap (lower is better)."""
    return np.max(_linear_overlap_values(h))

def _fix_sum(h, target=None):
    """Adjust h to have sum exactly N/2, keeping values in [0,1] as much as possible."""
    if target is None:
        target = N / 2.0
    current_sum = h.sum()
    diff = target - current_sum
    if abs(diff) < 1e-12:
        return h
    # Simple projection: distribute diff evenly, then clip and fix residual
    h = h + diff / N
    h = np.clip(h, 0.0, 1.0)
    residual = target - h.sum()
    if abs(residual) > 1e-12:
        if residual > 0:
            # Add to smallest elements
            idx = np.argsort(h)
            for i in idx:
                if residual <= 0:
                    break
                room = 1.0 - h[i]
                add = min(room, residual)
                h[i] += add
                residual -= add
        else:
            # Subtract from largest elements
            idx = np.argsort(h)[::-1]
            for i in idx:
                if residual >= 0:
                    break
                room = h[i]
                sub = min(room, -residual)
                h[i] -= sub
                residual += sub
    return h

def _smooth_max(values, temperature=0.1):
    """Smooth maximum via log-sum-exp."""
    max_val = np.max(values)
    return max_val + temperature * np.log(np.sum(np.exp((values - max_val) / temperature)))

def _smooth_max_grad(values, temperature=0.1):
    """Gradient of smooth max."""
    max_val = np.max(values)
    exp_vals = np.exp((values - max_val) / temperature)
    weights = exp_vals / np.sum(exp_vals)
    return weights

def _objective_and_grad(h, temperature=0.1):
    """Objective: smooth max of overlaps + penalty for sum constraint."""
    overlaps = _linear_overlap_values(h)
    obj = _smooth_max(overlaps, temperature)
    
    # Gradient of smooth max w.r.t. overlaps
    grad_overlaps = _smooth_max_grad(overlaps, temperature)
    
    # Chain rule: grad_h = sum_s grad_overlaps[s] * d(overlap_s)/d(h)
    grad_h = np.zeros(N)
    for s in range(1, N):
        idx = s - 1
        w = grad_overlaps[idx]
        # overlap_s = sum_{i=0}^{N-s-1} h[i]*(1-h[i+s])
        # d/dh[i] for i < N-s: (1-h[i+s])
        # d/dh[i] for i >= s and i < N: -h[i-s]
        grad_h[:N - s] += w * (1.0 - h[s:])
        grad_h[s:] += w * (-h[:N - s])
    
    # Penalty for sum constraint
    sum_penalty = 100.0 * (h.sum() - N / 2.0)
    grad_h += 100.0 / N * np.ones(N)
    
    return obj + sum_penalty, grad_h

def construct():
    deadline = time.time() + 110.0
    np.random.seed(42)
    
    # Initialize with Thue-Morse
    h = np.zeros(N)
    for i in range(N):
        cnt = bin(i).count('1')
        h[i] = 1.0 if cnt % 2 == 1 else 0.0
    h = _fix_sum(h)
    
    best_h = h.copy()
    best_score = _score(best_h)
    
    temperature = 0.1
    
    # Use L-BFGS-B with constraints
    while time.time() < deadline:
        # Try to improve with gradient descent
        def fun(x):
            obj, grad = _objective_and_grad(x, temperature)
            return obj, grad
        
        # Use a few restarts with different temperatures
        for temp in [0.05, 0.1, 0.2]:
            if time.time() > deadline:
                break
            
            result = minimize(fun, h, method='L-BFGS-B', jac=True,
                            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.x is not None:
                h_new = _fix_sum(result.x)
                score = _score(h_new)
                if score < best_score - 1e-9:
                    best_h = h_new.copy()
                    best_score = score
                    h = h_new.copy()
        
        # If no improvement, try random perturbations
        if time.time() > deadline:
            break
        
        improved = False
        for _ in range(50):
            if time.time() > deadline:
                break
            i = np.random.randint(N)
            h_trial = h.copy()
            h_trial[i] = np.clip(h_trial[i] + np.random.uniform(-0.2, 0.2), 0.0, 1.0)
            h_trial = _fix_sum(h_trial)
            score = _score(h_trial)
            if score < best_score - 1e-9:
                best_h = h_trial.copy()
                best_score = score
                h = h_trial.copy()
                improved = True
        
        if not improved:
            # Larger perturbation
            for _ in range(5):
                i = np.random.randint(N)
                h[i] = np.random.uniform(0, 1)
            h = _fix_sum(h)
            score = _score(h)
            if score < best_score:
                best_h = h.copy()
                best_score = score
    
    return best_h

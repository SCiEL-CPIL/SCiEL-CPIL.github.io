import numpy as np
from scipy.optimize import minimize

N = 100

def compute_overlaps(h):
    """Compute all cyclic shift overlaps of h with 1-h."""
    overlaps = np.empty(N)
    for t in range(N):
        overlaps[t] = np.sum(h * (1.0 - np.roll(h, t)))
    return overlaps

def construct():
    def objective(h):
        h_clipped = np.clip(h, 0, 1)
        overlaps = compute_overlaps(h_clipped)
        # Smooth max approximation with adaptive beta
        beta = 100.0
        m = np.max(overlaps)
        # Use log-sum-exp for numerical stability
        smooth_max = m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta
        
        # Penalty for sum constraint (tighter)
        sum_val = np.sum(h_clipped)
        sum_penalty = 5000.0 * (sum_val - N / 2.0) ** 2
        
        return smooth_max + sum_penalty

    def objective_and_grad(h):
        h_clipped = np.clip(h, 0, 1)
        overlaps = compute_overlaps(h_clipped)
        beta = 100.0
        m = np.max(overlaps)
        exp_terms = np.exp(beta * (overlaps - m))
        smooth_max = m + np.log(np.sum(exp_terms)) / beta
        
        # Gradient of smooth max w.r.t. overlaps
        grad_overlaps = exp_terms / np.sum(exp_terms)
        
        # Gradient of overlaps w.r.t. h
        # overlap[t] = sum_i h[i] * (1 - h[(i+t)%N])
        # d(overlap[t])/dh[j] = (1 - h[(j+t)%N]) - h[(j-t)%N]
        grad_h = np.zeros(N)
        for t in range(N):
            for j in range(N):
                grad_h[j] += grad_overlaps[t] * ((1.0 - h_clipped[(j + t) % N]) - h_clipped[(j - t) % N])
        
        # Sum penalty gradient
        sum_val = np.sum(h_clipped)
        sum_grad = 2.0 * 5000.0 * (sum_val - N / 2.0) * np.ones(N)
        
        # Only gradient where not clipped
        mask = (h > 0) & (h < 1)
        grad_h = grad_h * mask + sum_grad * mask
        
        return smooth_max + 5000.0 * (sum_val - N / 2.0) ** 2, grad_h

    # Try multiple initializations
    best_h = None
    best_score = np.inf
    
    initializations = []
    # Constant
    initializations.append(np.full(N, 0.5))
    # Sinusoidal with different frequencies
    for freq in [1, 2, 3, 5]:
        for amp in [0.1, 0.2, 0.3, 0.4]:
            x = np.full(N, 0.5)
            x += amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
            x = np.clip(x, 0, 1)
            initializations.append(x)
    # Random
    rng = np.random.RandomState(42)
    for _ in range(5):
        x = rng.uniform(0.3, 0.7, N)
        initializations.append(x)
    
    bounds = [(0.0, 1.0)] * N
    
    for x0 in initializations:
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            h = np.clip(res.x, 0, 1)
            # Evaluate actual score
            overlaps = compute_overlaps(h)
            score = np.max(overlaps)
            if score < best_score:
                best_score = score
                best_h = h.copy()
        except Exception:
            continue
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    # Ensure sum is exactly N/2
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h * (N / 2.0) / s
    best_h = np.clip(best_h, 0, 1)
    
    # Fine-tune sum to be exactly N/2
    diff = N / 2.0 - np.sum(best_h)
    idx = np.argsort(best_h)[::-1]
    for i in idx:
        if diff > 0:
            add = min(diff, 1.0 - best_h[i])
            best_h[i] += add
            diff -= add
        elif diff < 0:
            sub = min(-diff, best_h[i])
            best_h[i] -= sub
            diff += sub
        else:
            break
            
    return best_h

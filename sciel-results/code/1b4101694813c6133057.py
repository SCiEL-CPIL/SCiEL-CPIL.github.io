import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def _negative_ratio(h):
    """Negative of the ratio for minimization."""
    h = np.maximum(h, 0)  # enforce nonnegativity
    if np.sum(h) < 1e-15:
        return 0.0
    return -compute_ratio(h)

def _gradient_ratio(h, eps=1e-7):
    """Numerical gradient of the negative ratio."""
    grad = np.zeros_like(h)
    for i in range(N):
        h_plus = h.copy()
        h_plus[i] += eps
        h_minus = h.copy()
        h_minus[i] = max(h[i] - eps, 0)
        grad[i] = (_negative_ratio(h_plus) - _negative_ratio(h_minus)) / (2*eps)
    return grad

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    best = np.zeros(N, dtype=np.float64)
    best[0] = 1.0
    best_score = compute_ratio(best)

    # Generate diverse initial candidates
    candidates = []
    
    # Single spike
    h = np.zeros(N); h[0] = 1.0
    candidates.append(h.copy())
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Well-separated peaks (Sidon-like)
    for k in range(3, 10):
        h = np.zeros(N)
        # Choose k well-separated positions
        positions = np.linspace(0, N-1, k, dtype=int)
        for p in positions:
            h[p] = 1.0
        candidates.append(h.copy())
    
    # Random sparse vectors
    np.random.seed(42)
    for _ in range(20):
        h = np.zeros(N)
        num_nonzero = np.random.randint(3, 15)
        indices = np.random.choice(N, num_nonzero, replace=False)
        h[indices] = 1.0
        candidates.append(h.copy())
    
    # Evaluate all candidates
    for c in candidates:
        if time.time() > deadline - 30.0:
            break
        sc = compute_ratio(c)
        if sc > best_score:
            best_score = sc
            best = c.copy()

    # Gradient-based refinement from best candidate
    if time.time() < deadline - 10.0:
        h0 = best.copy()
        
        # Use L-BFGS-B with nonnegativity constraints
        bounds = [(0.0, None)] * N
        
        def objective(x):
            if time.time() > deadline - 5.0:
                return 0.0
            x = np.maximum(x, 0)
            return _negative_ratio(x)
        
        result = minimize(
            objective, 
            h0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < _negative_ratio(best):
            h_opt = np.maximum(result.x, 0)
            sc_opt = compute_ratio(h_opt)
            if sc_opt > best_score:
                best_score = sc_opt
                best = h_opt.copy()
        
        # Try multiple restarts from different initial points
        np.random.seed(123)
        for restart in range(5):
            if time.time() > deadline - 10.0:
                break
            
            # Create a new random starting point
            h_start = np.zeros(N)
            num_nonzero = np.random.randint(5, 20)
            indices = np.random.choice(N, num_nonzero, replace=False)
            h_start[indices] = 1.0
            
            # Optimize from this start
            result = minimize(
                objective, 
                h_start, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < _negative_ratio(best):
                h_opt = np.maximum(result.x, 0)
                sc_opt = compute_ratio(h_opt)
                if sc_opt > best_score:
                    best_score = sc_opt
                    best = h_opt.copy()
        
        # Final polish: coordinate ascent
        if time.time() < deadline - 5.0:
            h = best.copy()
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                improved = False
                for i in range(N):
                    old_val = h[i]
                    # Try different values for coordinate i
                    for trial in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
                        h_trial = h.copy()
                        h_trial[i] = trial
                        sc_trial = compute_ratio(h_trial)
                        if sc_trial > best_score + 1e-10:
                            best_score = sc_trial
                            best = h_trial.copy()
                            h = h_trial.copy()
                            improved = True
                            break
                if not improved:
                    break

    return list(best)

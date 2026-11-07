"""Optimized program for Erdos minimum overlap task."""
import numpy as np
import time

N = 100
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _value(h):
    """Compute the objective value (lower is better)."""
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2.0

def _soft_max(corr, beta):
    """Smooth approximation of max via log-sum-exp."""
    m = np.max(corr)
    return m + np.log(np.sum(np.exp(beta * (corr - m)))) / beta

def _objective(h, beta):
    """Smooth objective: soft-max of correlations, scaled."""
    corr = np.correlate(h, 1.0 - h, mode="full")
    return _soft_max(corr, beta) / N * 2.0

def _project(h):
    """Project h onto feasible set: [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    # Adjust sum to N/2
    target = N / 2.0
    diff = target - np.sum(h)
    # Distribute adjustment
    if diff > 0:
        # Need to increase some elements
        room = 1.0 - h
        total_room = np.sum(room)
        if total_room > 1e-12:
            h += diff * room / total_room
        else:
            h += diff / N
    elif diff < 0:
        # Need to decrease some elements
        room = h
        total_room = np.sum(room)
        if total_room > 1e-12:
            h += diff * room / total_room
        else:
            h += diff / N
    return np.clip(h, 0.0, 1.0)

def _feasible(h):
    """Check if h is feasible."""
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-9) or np.any(h > 1.0 + 1e-9):
        return False
    if abs(np.sum(h) - N / 2.0) > 1e-9 * N:
        return False
    return True

def _gradient_descent(h0, beta_start, beta_end, lr, n_iter, time_limit):
    """Run gradient descent on the smooth objective."""
    h = h0.copy().astype(np.float64)
    h = _project(h)
    best_h = h.copy()
    best_val = _value(h)
    
    t0 = time.time()
    for i in range(n_iter):
        if time.time() - t0 > time_limit:
            break
        
        # Anneal beta
        beta = beta_start + (beta_end - beta_start) * (i / max(1, n_iter - 1))
        beta = max(beta, 1.0)
        
        # Compute gradient numerically or analytically
        # Use analytical gradient of soft-max correlation
        corr = np.correlate(h, 1.0 - h, mode="full")
        m = np.max(corr)
        exps = np.exp(beta * (corr - m))
        weights = exps / np.sum(exps)  # softmax weights over shifts
        
        # Gradient of corr[k] w.r.t. h[i]
        # corr[k] = sum_j h[j] * (1 - h[j+k]) for appropriate indices
        # d(corr[k])/d(h[i]) = (1 - h[i+k]) - h[i-k] (with boundary handling)
        # Using the full correlation: corr[k] = sum_{j} h[j] * g[j + k] where g = 1-h
        # Actually np.correlate(h, 1-h, "full") gives:
        # corr[k] = sum_j h[j] * (1-h)[j + k - (N-1)] for k in 0..2N-2
        # Let's use numerical gradient for safety
        
        eps = 1e-6
        grad = np.zeros_like(h)
        for idx in range(N):
            h_plus = h.copy()
            h_plus[idx] += eps
            f_plus = _objective(h_plus, beta)
            
            h_minus = h.copy()
            h_minus[idx] -= eps
            f_minus = _objective(h_minus, beta)
            
            grad[idx] = (f_plus - f_minus) / (2 * eps)
        
        # Update
        h_new = h - lr * grad
        h_new = _project(h_new)
        
        # Check improvement
        val_new = _value(h_new)
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
        h = h_new
        
        # Adaptive learning rate
        if i % 50 == 0:
            lr *= 0.95
    
    return best_h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE
    
    # Best solution so far
    best_h = np.full(N, 0.5)
    best_val = _value(best_h)  # 0.5
    
    # Generate diverse starting points
    rng = np.random.RandomState(42)
    
    starts = []
    # Uniform (already best)
    starts.append(np.full(N, 0.5))
    
    # Sine waves
    for freq in [1, 2, 3, 5, 8]:
        x = np.linspace(0, 2 * np.pi * freq, N)
        h = 0.5 + 0.3 * np.sin(x)
        starts.append(_project(h))
    
    # Random perturbations
    for _ in range(5):
        h = 0.5 + 0.3 * rng.randn(N)
        starts.append(_project(h))
    
    # Structured patterns
    # Alternating blocks
    h = np.zeros(N)
    h[::2] = 1.0
    h[1::2] = 0.0
    starts.append(_project(h))
    
    # Ramp
    h = np.linspace(0, 1, N)
    h = _project(h)
    starts.append(h)
    
    # Step functions
    for split in [25, 33, 50, 66, 75]:
        h = np.zeros(N)
        h[:split] = 1.0
        h[split:] = 0.0
        h = _project(h)
        starts.append(h)
    
    # Optimize each starting point
    for start in starts:
        if time.time() > deadline - 10:
            break
        
        # Try different beta values and iterations
        for beta_start, beta_end, lr, n_iter in [
            (10, 100, 0.01, 200),
            (5, 50, 0.02, 150),
            (20, 200, 0.005, 300),
        ]:
            if time.time() > deadline - 10:
                break
            
            remaining = deadline - time.time() - 5
            if remaining <= 0:
                break
            
            h_opt, val_opt = _gradient_descent(
                start, beta_start, beta_end, lr, n_iter,
                time_limit=remaining
            )
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
    
    # Final polish: high-precision local optimization
    if time.time() < deadline - 5:
        # Use smaller steps and higher beta
        for beta in [50, 100, 200]:
            if time.time() > deadline - 5:
                break
            remaining = deadline - time.time() - 3
            if remaining <= 0:
                break
            h_polish, val_polish = _gradient_descent(
                best_h, beta, beta * 2, 0.001, 100,
                time_limit=remaining
            )
            if val_polish < best_val:
                best_val = val_polish
                best_h = h_polish.copy()
    
    # Ensure feasibility
    best_h = _project(best_h)
    
    # Final check
    if not _feasible(best_h):
        best_h = np.full(N, 0.5)
    
    return best_h

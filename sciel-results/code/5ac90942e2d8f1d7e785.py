"""Optimize h to minimize max autocorrelation with (1-h)."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 120.0
SAFETY_MARGIN = 15.0

def _compute_value(h):
    """Compute the objective value: max(np.correlate(h, 1-h, mode='full')) / N * 2."""
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2.0

def _is_valid(h):
    """Check if h is valid: finite, in [0,1], sum close to N/2."""
    if not np.all(np.isfinite(h)):
        return False
    if not np.all(h >= 0.0) or not np.all(h <= 1.0):
        return False
    if abs(np.sum(h) - N / 2.0) > 1e-9 * N:
        return False
    return True

def _project_to_feasible(h):
    """Project h onto [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(100):
        diff = target - np.sum(h)
        if abs(diff) < 1e-14:
            break
        h = np.clip(h + diff / N, 0.0, 1.0)
    return h

def _gradient_descent(h, lr=0.01, max_iter=5000, tol=1e-12):
    """Gradient descent with projection for the max correlation objective."""
    h = h.copy()
    h = _project_to_feasible(h)
    
    for it in range(max_iter):
        # Compute correlation
        g = 1.0 - h
        corr = np.correlate(h, g, mode="full")
        
        # Find the max correlation index
        idx = np.argmax(corr)
        d = idx - (N - 1)  # lag
        
        # Compute gradient of the active correlation term
        # corr[idx] = sum_i h[i] * g[i+d] for valid i
        # d(corr[idx])/dh[i] = g[i+d] - h[i-d] (where both indices valid)
        grad = np.zeros(N)
        if d >= 0:
            for i in range(d, N - d):
                grad[i] = g[i + d] - h[i - d]
        else:
            dd = -d
            for i in range(dd, N):
                grad[i] = g[i + d] - h[i - d]
        
        # Project gradient descent step
        h_new = h - lr * grad
        h_new = _project_to_feasible(h_new)
        
        # Check convergence
        if np.max(np.abs(h_new - h)) < tol:
            h = h_new
            break
        h = h_new
    
    return h

def _sinusoidal_start(freqs, phases, amplitudes):
    """Create a starting point using sinusoidal components."""
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    h = np.zeros(N)
    for f, p, a in zip(freqs, phases, amplitudes):
        h += a * np.sin(f * x + p)
    h = 0.5 + h
    return _project_to_feasible(h)

def construct():
    start_time = time.time()
    deadline = TIME_LIMIT - SAFETY_MARGIN
    
    # Best solution found so far
    best_h = np.full(N, 0.5)
    best_val = _compute_value(best_h)
    
    rng = np.random.RandomState(42)
    
    # Define several sinusoidal patterns to try
    patterns = []
    
    # Pattern 1: single low frequency
    patterns.append((1, 0.0, 0.3))
    patterns.append((2, 0.0, 0.2))
    patterns.append((3, 0.0, 0.15))
    
    # Pattern 2: two frequencies
    for f1 in [1, 2]:
        for f2 in [3, 4, 5]:
            patterns.append((f1, 0.0, 0.2))
            patterns.append((f2, np.pi / 2, 0.15))
    
    # Pattern 3: three frequencies
    for f1, f2, f3 in [(1, 3, 5), (2, 4, 6), (1, 4, 7)]:
        patterns.append((f1, 0.0, 0.2))
        patterns.append((f2, np.pi / 3, 0.15))
        patterns.append((f3, 2 * np.pi / 3, 0.1))
    
    # Pattern 4: random phases
    for _ in range(10):
        f = rng.randint(1, 6)
        p = rng.uniform(0, 2 * np.pi)
        a = rng.uniform(0.1, 0.3)
        patterns.append((f, p, a))
    
    # Try each pattern with gradient descent
    for freq, phase, amp in patterns:
        if time.time() - start_time > deadline:
            break
        
        h_start = _sinusoidal_start([freq], [phase], [amp])
        h_opt = _gradient_descent(h_start, lr=0.05, max_iter=3000)
        val = _compute_value(h_opt)
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
    
    # Multi-frequency starts
    multi_patterns = [
        ([1, 3], [0, np.pi/2], [0.2, 0.15]),
        ([2, 4], [0, np.pi/3], [0.2, 0.15]),
        ([1, 2, 4], [0, np.pi/4, np.pi/2], [0.15, 0.15, 0.1]),
        ([3, 5], [0, np.pi/3], [0.15, 0.1]),
        ([1, 5], [0, 2*np.pi/5], [0.2, 0.1]),
    ]
    
    for freqs, phases, amps in multi_patterns:
        if time.time() - start_time > deadline:
            break
        
        h_start = _sinusoidal_start(freqs, phases, amps)
        h_opt = _gradient_descent(h_start, lr=0.05, max_iter=3000)
        val = _compute_value(h_opt)
        
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
    
    # Local refinement around best solution
    if time.time() - start_time < deadline:
        for trial in range(30):
            if time.time() - start_time > deadline:
                break
            
            # Random perturbation
            h_try = best_h + 0.02 * rng.randn(N)
            h_try = _project_to_feasible(h_try)
            h_opt = _gradient_descent(h_try, lr=0.02, max_iter=2000)
            val = _compute_value(h_opt)
            
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
    
    # Final polish with tighter tolerance
    if time.time() - start_time < deadline:
        h_final = _gradient_descent(best_h, lr=0.005, max_iter=5000, tol=1e-14)
        val = _compute_value(h_final)
        if val < best_val:
            best_val = val
            best_h = h_final
    
    # Ensure validity
    best_h = _project_to_feasible(best_h)
    
    # Final verification
    assert _is_valid(best_h), f"Invalid solution: sum={np.sum(best_h)}, min={np.min(best_h)}, max={np.max(best_h)}"
    
    return best_h

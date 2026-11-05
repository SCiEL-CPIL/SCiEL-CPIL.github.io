"""Optimized program for third autocorrelation inequality."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _objective(h):
    """Compute the value to minimize."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    return abs(2.0 * N * max_conv / (s * s))

def _objective_and_grad(h):
    """Objective with gradient approximation."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros(N)
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    val = abs(2.0 * N * max_conv / (s * s))
    
    # Numerical gradient
    eps = 1e-7
    grad = np.zeros(N)
    for i in range(N):
        h_plus = h.copy()
        h_plus[i] += eps
        s_plus = np.sum(h_plus)
        if abs(s_plus) < 1e-12:
            grad[i] = 0
            continue
        conv_plus = np.convolve(h_plus, h_plus)
        max_plus = np.max(conv_plus)
        val_plus = abs(2.0 * N * max_plus / (s_plus * s_plus))
        grad[i] = (val_plus - val) / eps
    return val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _objective(best_h)
    
    # Try multiple initializations
    np.random.seed(42)
    
    initializations = []
    # Constant
    initializations.append(np.ones(N))
    # Random
    for _ in range(5):
        initializations.append(np.random.randn(N))
    # Alternating with small bias
    alt = np.ones(N)
    alt[::2] = -1.0
    initializations.append(alt * 0.1 + 0.5)
    # Smooth variations
    t = np.linspace(0, 2*np.pi, N)
    initializations.append(1.0 + 0.5 * np.sin(t))
    initializations.append(1.0 + 0.5 * np.cos(t))
    
    for h0 in initializations:
        if time.time() > deadline:
            break
        
        # Normalize to have reasonable sum
        s = np.sum(h0)
        if abs(s) < 1e-6:
            h0 = h0 + 0.1 * np.ones(N)
        
        # Try L-BFGS-B optimization
        try:
            res = minimize(
                _objective,
                h0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8},
                timeout=20
            )
            if res.success or res.fun < best_val:
                if np.all(np.isfinite(res.x)):
                    s_check = np.sum(res.x)
                    if abs(s_check) > 1e-12:
                        val = _objective(res.x)
                        if val < best_val:
                            best_val = val
                            best_h = res.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # Additional local refinement with simulated annealing on best solution
    if time.time() < deadline - 10:
        h = best_h.copy()
        current_val = best_val
        T = 0.1
        for step in range(5000):
            if time.time() > deadline:
                break
            i = np.random.randint(N)
            delta = np.random.randn() * 0.1
            h_new = h.copy()
            h_new[i] += delta
            s_new = np.sum(h_new)
            if abs(s_new) < 1e-12:
                continue
            new_val = _objective(h_new)
            if new_val < current_val:
                h = h_new
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                # Accept with probability exp(-delta_E/T)
                delta_E = new_val - current_val
                if delta_E < 0:
                    accept = True
                else:
                    accept = np.random.random() < np.exp(-delta_E / T)
                if accept:
                    h = h_new
                    current_val = new_val
            T *= 0.999
    
    return best_h
# EVOLVE-BLOCK-END
